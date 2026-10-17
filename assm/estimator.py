"""Pyro implementation of Attentive State-Space Models 
Public training/inference API wrapper over model components."""

from dataclasses import asdict
from pathlib import Path
import copy
import torch
import pyro
from .config import ModelConfig, TrainingConfig
from .model import AttentiveStateSpaceModel
from .training import initialize_from_observations, fit_svi
from . import inference


class AttentiveSSM:
    def __init__(self, config=None, device="cpu", seed=19):
        # Set the seed before network initialization, not only inside fit().
        pyro.set_rng_seed(seed)
        self.config                  = config or ModelConfig()
        self.model                   = AttentiveStateSpaceModel(self.config).to(device)
        self.history_                = []
        self.initialization_history_ = []
        self.mixture_                = None
        self.training_seconds_       = None

    def fit(self, train, validation=None, training=None):
        """Training function for the attentive state-space model
        """
        config = training or TrainingConfig()
        pyro.set_rng_seed(config.seed)
        self.model.validate_batch(train)

        if validation is not None:
            self.model.validate_batch(validation)

        if config.warm_start:
            self.mixture_, self.initialization_history_ = initialize_from_observations(self.model, train, config)

        self.initialized_state_               = copy.deepcopy(self.model.state_dict())
        self.history_, self.training_seconds_ = fit_svi(self.model, train, validation, config, {"params": {}, "constraints": {}})
        self.model.eval()
        return self

    def posterior(self, batch, num_samples=128):
        return inference.posterior(self.model, batch, num_samples)

    def filter(self, batch, num_samples=128):
        return inference.filter_prefixes(self.model, batch, num_samples)

    def forecast(self, batch, horizon=3, num_samples=128):
        return inference.forecast(self.model, batch, horizon, num_samples)

    def exact_posterior(self, batch, max_paths=100_000):
        return inference.exact_posterior(self.model, batch, max_paths)

    @torch.no_grad()
    def attention_weights(self, batch):
        self.model.validate_batch(batch)
        return self.model.attention(batch.context) * batch.mask[..., None]

    def save(self, path):
        """Save tensor-only inference checkpoint; includes config and loss history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.model.state_dict(), "history": self.history_}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        
        checkpoint         = torch.load(path, map_location=device, weights_only=True)
        estimator          = cls(ModelConfig(**checkpoint["config"]), device=device)

        estimator.model.load_state_dict(checkpoint["state_dict"])
        
        estimator.history_ = checkpoint["history"]
        estimator.model.eval()
        
        return estimator
