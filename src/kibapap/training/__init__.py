"""Training utilities: collocation point sampling and training loop."""

from kibapap.training.sampler import sample_collocation, sample_boundary, sample_initial
from kibapap.training.trainer import train_forward_pinn, TrainingConfig

__all__ = [
    "sample_collocation",
    "sample_boundary",
    "sample_initial",
    "train_forward_pinn",
    "TrainingConfig",
]
