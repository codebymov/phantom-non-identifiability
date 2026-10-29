"""Neural network components: MLPs, segregated PINN, and PINN losses."""

from kibapap.models.mlp import MLP, FourierFeatures
from kibapap.models.pinn import SegregatedPINN
from kibapap.models.losses import (
    pde_residuals,
    boundary_residuals,
    initial_residuals,
    gpinn_residual_gradient,
)

__all__ = [
    "MLP",
    "FourierFeatures",
    "SegregatedPINN",
    "pde_residuals",
    "boundary_residuals",
    "initial_residuals",
    "gpinn_residual_gradient",
]
