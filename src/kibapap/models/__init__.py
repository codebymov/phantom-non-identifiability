"""Neural network components: MLPs, segregated PINN, and PINN losses."""

from kibapap.models.losses import (
    boundary_residuals,
    gpinn_residual_gradient,
    initial_residuals,
    pde_residuals,
)
from kibapap.models.mlp import MLP, FourierFeatures
from kibapap.models.pinn import SegregatedPINN

__all__ = [
    "MLP",
    "FourierFeatures",
    "SegregatedPINN",
    "boundary_residuals",
    "gpinn_residual_gradient",
    "initial_residuals",
    "pde_residuals",
]
