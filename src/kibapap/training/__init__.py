"""Training utilities: collocation point sampling and training loop."""

from kibapap.training.sampler import sample_boundary, sample_collocation, sample_initial
from kibapap.training.trainer import TrainingConfig, train_forward_pinn

__all__ = [
    "TrainingConfig",
    "sample_boundary",
    "sample_collocation",
    "sample_initial",
    "train_forward_pinn",
]
