"""Attentive discrete state-space models implemented with PyTorch & Pyro."""

from .config import ModelConfig, TrainingConfig
from .data import SequenceBatch
from .estimator import AttentiveSSM
from .synthetic import generate_synthetic, SyntheticConfig

__all__ = ["AttentiveSSM", "ModelConfig", "TrainingConfig", "SequenceBatch", "generate_synthetic", "SyntheticConfig"]
