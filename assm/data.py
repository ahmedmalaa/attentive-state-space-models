"""Pyro implementation of Attentive State-Space Models
"""

from dataclasses import dataclass
import torch


@dataclass
class SequenceBatch:
    continuous: torch.Tensor       # [patients, visits, continuous features]
    binary: torch.Tensor           # [patients, visits, binary features], possibly size 0
    lengths: torch.Tensor          # [patients], number of valid visits

    def __post_init__(self):
        x, b, lengths = self.continuous, self.binary, self.lengths
        if x.ndim != 3 or b.ndim != 3 or x.shape[:2] != b.shape[:2]:
            raise ValueError("continuous and binary must have matching [N,T,D] shapes.")
        if min(x.shape) < 1 or lengths.shape != (x.shape[0],):
            raise ValueError("Nonempty continuous data and one length per patient are required.")
        if not x.is_floating_point() or b.dtype != x.dtype:
            raise TypeError("Observations must use the same floating-point dtype.")
        if lengths.dtype != torch.long:
            raise TypeError("lengths must be torch.long.")
        if not (x.device == b.device == lengths.device):
            raise ValueError("All batch tensors must be on the same device.")
        if torch.any((lengths < 1) | (lengths > x.shape[1])):
            raise ValueError("Every length must be in [1,T].")
        mask = self.mask
        if not torch.isfinite(x[mask]).all() or not torch.isfinite(b[mask]).all():
            raise ValueError("Valid observations must be finite; NaNs are allowed only in padding.")
        if torch.any((b[mask] != 0) & (b[mask] != 1)):
            raise ValueError("Valid binary observations must be 0 or 1.")
        self.continuous = torch.where(mask[..., None], x, torch.zeros_like(x))
        self.binary = torch.where(mask[..., None], b, torch.zeros_like(b))

    @property
    def mask(self):
        return torch.arange(self.continuous.shape[1], device=self.lengths.device)[None] < self.lengths[:, None]

    @property
    def context(self):
        return torch.cat((self.continuous, self.binary), dim=-1)

    @property
    def n_patients(self):
        return self.continuous.shape[0]

    @property
    def n_visits(self):
        return self.continuous.shape[1]

    def select(self, indices):
        return SequenceBatch(self.continuous[indices], self.binary[indices], self.lengths[indices])

    def prefix(self, stop):
        if not 1 <= stop <= self.n_visits:
            raise ValueError("stop must be in [1,T].")
        return SequenceBatch(self.continuous[:, :stop], self.binary[:, :stop], self.lengths.clamp(max=stop))

    def to(self, device):
        return SequenceBatch(self.continuous.to(device), self.binary.to(device), self.lengths.to(device))
