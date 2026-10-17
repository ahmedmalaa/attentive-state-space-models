"""Pyro implementation of Attentive State-Space Models

 This code implements the generative model and structured guide 
 desribed in equations (2)-(10) of "Attentive State-Space Modeling of Disease Progression"
 by Alaa and van der Schaar in NeurIPS 2019.
"""

import torch
from torch import nn
import pyro
import pyro.distributions as dist
from .networks import CausalSeq2SeqAttention, AttentiveGuideNetwork
from .emissions import MixedEmissions


class AttentiveStateSpaceModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config            = config
        k                      = config.num_states                              # number of hidden states are predetermined
        self.initial_logits    = nn.Parameter(torch.zeros(k))
        self.transition_logits = nn.Parameter(torch.eye(k) * 1.5)
        input_dim              = config.n_continuous + config.n_binary

        # generative model and structured inference network architectures
        self.attention         = CausalSeq2SeqAttention(input_dim, 
                                                        config.hidden_dim, 
                                                        config.num_layers)      # Architectural choice for the attention mechanism in eqs (4) and (5)   
        self.emissions         = MixedEmissions(config)                         # Emission distribution is specified in eq (6)
        self.guide_network     = AttentiveGuideNetwork(input_dim, 
                                                       config.hidden_dim, 
                                                       config.num_layers, k)

    @property
    def initial_probs(self):
        return self.initial_logits.softmax(-1)

    @property
    def transition_matrix(self):
        return self.transition_logits.softmax(-1)

    # routine checks on feature dimensions and data/model devices
    def validate_batch(self, batch):
        if batch.continuous.shape[-1] != self.config.n_continuous or batch.binary.shape[-1] != self.config.n_binary:
            raise ValueError("Feature dimensions do not match ModelConfig.")
        if batch.continuous.device != self.initial_logits.device or batch.continuous.dtype != self.initial_logits.dtype:
            raise ValueError("Data and model must use the same device and floating-point dtype.")

    def forward_probs(self, alpha_row, states):
        """Implementation of the transition model in Eq. (4): 
            sum_s alpha[t,s] * P[z_s,:], or pi for the first state.
        """
        if not states:
            return self.initial_probs.expand(alpha_row.shape[0], -1)
        past = torch.stack(states, dim=-1)
        rows = self.transition_matrix[past]     # [N, t, K]
        return (alpha_row[:, :len(states), None] * rows).sum(-2)

    def model(self, batch):
        """Model for p_theta(x,z)

        Note that we deliberately do not use pyro.markov here since all earlier 
        states can affect every later transition. 
        """
        self.validate_batch(batch)
        pyro.module("assm", self)
        
        alpha  = self.attention(batch.context) # generate attention weights from context to compute transition probabilities
        states = []

        with pyro.plate("patients", batch.n_patients):
            for t in range(batch.n_visits):
                with pyro.poutine.mask(mask=batch.mask[:, t]):
                    probs = self.forward_probs(alpha[:, t], states)
                    z     = pyro.sample(f"z_{t}", dist.Categorical(probs=probs))
                    pyro.sample(f"continuous_{t}", 
                                self.emissions.continuous_distribution(z), 
                                obs=batch.continuous[:, t])
                    if self.config.n_binary:
                        pyro.sample(f"binary_{t}", 
                                    self.emissions.binary_distribution(batch.continuous[:, t], z), 
                                    obs=batch.binary[:, t]) # the binary distribution is function of the continuous observation based on the factorization in eq (6)
                states.append(z)
        return torch.stack(states, dim=-1)

    def guide(self, batch):
        """q_phi(z|x): shared attention and P plus backwards observation context.

        Because the categorical draws of X are not differentiable, TraceGraph_ELBO will provide the
        score-function estimator and a moving-average control variate.
        """
        self.validate_batch(batch)
        pyro.module("assm", self)
        alpha      = self.attention(batch.context)
        backward_h = self.guide_network.backward(batch.context, batch.mask)
        states     = []
        
        with pyro.plate("patients", batch.n_patients):
            for t in range(batch.n_visits):
                forward = self.forward_probs(alpha[:, t], states)
                logits  = self.guide_network.logits(backward_h[:, t], forward)
                with pyro.poutine.mask(mask=batch.mask[:, t]):
                    z   = pyro.sample(f"z_{t}", dist.Categorical(logits=logits), infer={
                          "baseline": {"use_decaying_avg_baseline": True, "baseline_beta": 0.9}
                    })
                states.append(z)
        return torch.stack(states, dim=-1)

    def teacher_forced_logits(self, batch, pseudo_states):
        """Warm-start the combiner on *unsupervised* GMM assignments only.
        """
        alpha            = self.attention(batch.context).detach()
        h                = self.guide_network.backward(batch.context, batch.mask)
        previous, logits = [], []

        for t in range(batch.n_visits):
            forward = self.forward_probs(alpha[:, t], previous).detach()
            logits.append(self.guide_network.logits(h[:, t], forward))
            previous.append(pseudo_states[:, t])
        return torch.stack(logits, dim=1)
