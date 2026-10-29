"""Physics: Luikov PDE definitions and finite-difference reference solver."""

from kibapap.physics.fd_solver import solve_luikov_fd
from kibapap.physics.luikov import LuikovParams

__all__ = ["LuikovParams", "solve_luikov_fd"]
