"""Pyro implementation of Attentive State-Space Models 
State-specific Gaussian plus conditional Bernoulli emissions as described in Eq. (6)."""

import torch
from torch import nn
from torch.nn import functional as F
import pyro.distributions as dist


def inverse_softplus(x):
    """Stable inverse used only for initializing unconstrained scale parameters."""
    return x + torch.log(-torch.expm1(-x))


class MixedEmissions(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config    = config
        k, c, b        = config.num_states, config.n_continuous, config.n_binary

        # randomly initialize the mean and variance of continuous emission dist
        self.means     = nn.Parameter(torch.randn(k, c) * 0.5)
        self.raw_scale = nn.Parameter(inverse_softplus(torch.ones(k, c)))

        if config.covariance == "full":
            self.lower = nn.Parameter(torch.zeros(k, c, c))
        else:
            self.register_buffer("lower", torch.zeros(k, c, c))
        
        # initialize the Bernoulli distribution with Gaussians
        self.binary_weights = nn.Parameter(torch.zeros(k, b, c))
        self.binary_bias    = nn.Parameter(torch.zeros(k, b))

    @property
    def scale_tril(self):
        diagonal = torch.diag_embed(F.softplus(self.raw_scale) + self.config.min_scale)
        return diagonal + self.lower.tril(-1) if self.config.covariance == "full" else diagonal

    @property
    def covariance(self):
        l        = self.scale_tril
        return l @ l.transpose(-1, -2)

    def continuous_distribution(self, state):
        return dist.MultivariateNormal(self.means[state], scale_tril=self.scale_tril[state])

    def binary_logits(self, continuous, state):
        return (self.binary_weights[state] * continuous.unsqueeze(-2)).sum(-1) + self.binary_bias[state]

    def binary_distribution(self, continuous, state):
        # Binary features are conditionally independent given (x_continuous,z)
        return dist.Bernoulli(logits=self.binary_logits(continuous, state)).to_event(1)

    def log_prob_all_states(self, continuous, binary):
        """Return [...,K] emission log densities for diagnostics/initialization."""
        logp = dist.MultivariateNormal(self.means, scale_tril=self.scale_tril).log_prob(continuous.unsqueeze(-2))
        if self.config.n_binary:
            logits = torch.einsum("...c,kbc->...kb", continuous, self.binary_weights) + self.binary_bias
            logp   = logp + dist.Bernoulli(logits=logits).log_prob(binary.unsqueeze(-2)).sum(-1)
        return logp

    @torch.no_grad()
    def set_gaussian(self, means, covariances):

        self.means.copy_(means)
        
        if self.config.covariance == "diag":
            covariances = torch.diag_embed(covariances.diagonal(dim1=-2, dim2=-1))
        
        l    = torch.linalg.cholesky(covariances)
        diag = l.diagonal(dim1=-2, dim2=-1)
        self.raw_scale.copy_(inverse_softplus((diag - self.config.min_scale).clamp(min=1e-4)))
        self.lower.copy_(l.tril(-1))
