"""Pyro implementation of Attentive State-Space Models
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:

    num_states  : int   = 3
    n_continuous: int   = 2
    n_binary    : int   = 1
    hidden_dim  : int   = 32
    num_layers  : int   = 1
    covariance  : str   = "full"
    min_scale   : float = 0.05

    def __post_init__(self):
        if min(self.num_states, self.n_continuous, self.hidden_dim, self.num_layers) < 1:
            raise ValueError("States, continuous features, hidden size and layers must be positive.")
        if self.n_binary < 0 or self.min_scale <= 0:
            raise ValueError("n_binary must be nonnegative; min_scale must be positive.")
        if self.covariance not in {"full", "diag"}:
            raise ValueError("covariance must be 'full' or 'diag'.")


@dataclass
class TrainingConfig:
    epochs: int           = 60
    batch_size: int       = 64
    learning_rate: float  = 5e-4
    num_particles: int    = 2
    clip_norm: float      = 10.0
    seed: int             = 19
    warm_start: bool      = True
    warm_start_steps: int = 250
    warm_start_lr: float  = 3e-3
    log_every: int        = 10

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.num_particles, self.log_every) < 1:
            raise ValueError("Epochs, batch size, particles and log interval must be positive.")
        if min(self.learning_rate, self.warm_start_lr, self.clip_norm) <= 0:
            raise ValueError("Learning rates and gradient clipping threshold must be positive.")
        if self.warm_start_steps < 0:
            raise ValueError("warm_start_steps cannot be negative.")
