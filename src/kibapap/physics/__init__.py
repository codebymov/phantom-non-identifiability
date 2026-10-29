"""Physics: Luikov PDE definitions and finite-difference reference solver."""

from kibapap.physics.luikov import LuikovParams
from kibapap.physics.fd_solver import solve_luikov_fd

__all__ = ["LuikovParams", "solve_luikov_fd"]
