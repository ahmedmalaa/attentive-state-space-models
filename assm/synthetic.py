"""Pyro implementation of Attentive State-Space Models 

This data generation process uses our paper's generative factorization, 
with an analytic attention rule as the ground-truth state dynamic. 
"""

from dataclasses import dataclass
import torch
from .data import SequenceBatch


@dataclass
class SyntheticConfig:
    n_patients: int          = 480
    max_visits: int          = 16
    min_visits: int          = 10
    observation_noise: float = 0.8
    seed: int                = 71


def true_attention(context):
    """Analytic model for the attention mechanism in the data gen process
    """
    n, t, _ = context.shape

    if t < 1:
        raise ValueError("Attention requires at least one past observation.")

    event   = context[:, -1, -1]
    decay   = 1.4 * (1 - event) + 0.10 * event
    ages    = torch.arange(t - 1, -1, -1, device=context.device, dtype=context.dtype)

    return (-decay[:, None] * ages[None]).softmax(-1)


def generate_synthetic(config=None):

    config     = config or SyntheticConfig()

    if not 1 <= config.min_visits <= config.max_visits or config.n_patients < 1 or config.observation_noise <= 0:
        raise ValueError("Require N>0, noise>0 and 1 <= min_visits <= max_visits.")

    rng        = torch.Generator().manual_seed(config.seed)
    n, total   = config.n_patients, config.max_visits
    pi         = torch.tensor([0.45, 0.35, 0.20])
    transition = torch.tensor([[0.85, 0.12, 0.03], [0.07, 0.83, 0.10], [0.04, 0.14, 0.82]])
    means      = torch.tensor([[-2.2, -1.0], [0.0, 1.7], [2.2, -0.8]])
    base       = torch.tensor([[1.0, 0.20], [0.20, 0.85]]) * config.observation_noise**2
    covariance = base[None].repeat(3, 1, 1)
    scale_tril = torch.linalg.cholesky(covariance)
    weights    = torch.tensor([[[0.15, -0.10]], [[0.25, 0.15]], [[0.15, 0.20]]])
    bias       = torch.tensor([[-1.2], [0.1], [1.1]])
    z          = torch.empty(n, total, dtype=torch.long)
    x          = torch.empty(n, total, 2)
    b          = torch.empty(n, total, 1)
    alpha      = torch.zeros(n, total, total)

    for t in range(total):
        if t == 0:
            probs           = pi.expand(n, -1)
        else:
            context         = torch.cat((x[:, :t], b[:, :t]), dim=-1)
            a               = true_attention(context)
            alpha[:, t, :t] = a
            probs           = (a[..., None] * transition[z[:, :t]]).sum(1)
        
        z[:, t] = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
        noise   = torch.randn(n, 2, generator=rng)
        x[:, t] = means[z[:, t]] + torch.einsum("nij,nj->ni", scale_tril[z[:, t]], noise)
        logits  = (weights[z[:, t]] * x[:, t, None]).sum(-1) + bias[z[:, t]]
        b[:, t] = (torch.rand(n, 1, generator=rng) < logits.sigmoid()).float()
    
    lengths = torch.randint(config.min_visits, total + 1, (n,), generator=rng)
    batch   = SequenceBatch(x, b, lengths)
    
    return {
        "batch": batch, "states": z.masked_fill(~batch.mask, -1),
        "attention": alpha * batch.mask[..., None],
        "initial_probs": pi, "transition_matrix": transition, "means": means,
        "covariance": covariance, "binary_weights": weights, "binary_bias": bias,
    }
