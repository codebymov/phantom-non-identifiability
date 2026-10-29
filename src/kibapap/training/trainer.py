"""PINN training loop. Adam warm-up followed by optional L-BFGS finetuning.

The loss combines four terms by default (PDE, BC, IC, plus optional gPINN). Weights
are scalar hyperparameters at this phase; adaptive weighting (ReLoBRaLo or Self-Adaptive
PINN) is a phase-2 extension and lives in a separate `adaptive_losses.py` once
implemented.

The training schedule used in PINN literature is Adam at lr=1e-3 with cosine decay for
~10k iters, followed by L-BFGS for 2-5k iters. Adam handles the initial steep
descent; L-BFGS polishes the residuals once we are near a good basin. Going straight to
L-BFGS from initialization is unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from kibapap.models.losses import (
    boundary_residuals,
    gpinn_residual_gradient,
    initial_residuals,
    pde_residuals,
)
from kibapap.physics.luikov import LuikovParams
from kibapap.training.sampler import sample_boundary, sample_collocation, sample_initial


@dataclass
class TrainingConfig:
    """Hyperparameters for forward-PINN training."""

    # Sampling
    n_collocation: int = 5000
    n_boundary: int = 500
    n_initial: int = 500
    tau_final: float = 2.0
    parametric: bool = False
    resample_every: int = 100  # resample collocation points every N iters

    # Loss weights (hard_ic=True makes ic loss ~1e-12 by construction; leave w_ic small as a sanity check)
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_ic: float = 1.0
    w_gpinn: float = 0.0  # set > 0 to enable gPINN loss in phase 2

    # Adam stage
    adam_iters: int = 10000
    adam_lr: float = 1e-3
    cosine_decay: bool = True

    # L-BFGS stage (set lbfgs_iters=0 to skip)
    lbfgs_iters: int = 2000
    lbfgs_history: int = 50

    # Logging
    log_every: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def _compute_loss(
    model: nn.Module,
    x_pde: torch.Tensor,
    x_bc_left: torch.Tensor,
    x_bc_right: torch.Tensor,
    x_ic: torch.Tensor,
    defaults: LuikovParams,
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward pass + loss assembly. Returns (total_loss, per_term_dict)."""
    # PDE residuals
    res_T, res_M = pde_residuals(model, x_pde, defaults)
    loss_pde = (res_T.pow(2).mean() + res_M.pow(2).mean()) * 0.5

    # BC residuals
    rTl, rMl, rTr, rMr = boundary_residuals(model, x_bc_left, x_bc_right, defaults)
    loss_bc = 0.25 * (rTl.pow(2).mean() + rMl.pow(2).mean() + rTr.pow(2).mean() + rMr.pow(2).mean())

    # IC residuals
    rT0, rM0 = initial_residuals(model, x_ic)
    loss_ic = 0.5 * (rT0.pow(2).mean() + rM0.pow(2).mean())

    total = cfg.w_pde * loss_pde + cfg.w_bc * loss_bc + cfg.w_ic * loss_ic

    if cfg.w_gpinn > 0.0:
        g_T, g_M = gpinn_residual_gradient(model, x_pde, defaults)
        loss_gpinn = 0.5 * (g_T.pow(2).mean() + g_M.pow(2).mean())
        total = total + cfg.w_gpinn * loss_gpinn
        log = {"pde": loss_pde.item(), "bc": loss_bc.item(), "ic": loss_ic.item(), "gpinn": loss_gpinn.item()}
    else:
        log = {"pde": loss_pde.item(), "bc": loss_bc.item(), "ic": loss_ic.item()}

    return total, log


def train_forward_pinn(
    model: nn.Module,
    defaults: LuikovParams,
    cfg: TrainingConfig | None = None,
    log_dir: Path | str | None = None,
) -> dict[str, list]:
    """Train a forward PINN against PDE+BC+IC residuals.

    No FD data is used here — this is the pure forward problem. Phase-3 inverse
    training is a separate function that adds a data loss term.

    Returns a history dict with keys 'iter', 'total', 'pde', 'bc', 'ic', and optionally
    'gpinn'. The model is trained in place.
    """
    cfg = cfg or TrainingConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)

    history: dict[str, list] = {"iter": [], "total": [], "pde": [], "bc": [], "ic": []}
    if cfg.w_gpinn > 0.0:
        history["gpinn"] = []

    # ---------- Adam stage ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    if cfg.cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.adam_iters)
    else:
        scheduler = None

    # Pre-sample sets that will be refreshed periodically
    def fresh_samples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_pde = sample_collocation(cfg.n_collocation, cfg.tau_final, cfg.parametric, device=device)
        x_bcl, x_bcr = sample_boundary(cfg.n_boundary, cfg.tau_final, cfg.parametric, device=device)
        x_ic = sample_initial(cfg.n_initial, cfg.parametric, device=device)
        return x_pde, x_bcl, x_bcr, x_ic

    x_pde, x_bcl, x_bcr, x_ic = fresh_samples()

    pbar = tqdm(range(cfg.adam_iters), desc="Adam")
    for it in pbar:
        if cfg.resample_every > 0 and it % cfg.resample_every == 0 and it > 0:
            x_pde, x_bcl, x_bcr, x_ic = fresh_samples()

        optimizer.zero_grad()
        total, log = _compute_loss(model, x_pde, x_bcl, x_bcr, x_ic, defaults, cfg)
        total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if it % cfg.log_every == 0:
            history["iter"].append(it)
            history["total"].append(total.item())
            for k, v in log.items():
                history[k].append(v)
            pbar.set_postfix({"total": f"{total.item():.3e}", **{k: f"{v:.2e}" for k, v in log.items()}})

    # ---------- L-BFGS stage ----------
    if cfg.lbfgs_iters > 0:
        # L-BFGS uses the most recent sample set, fixed for the duration.
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=cfg.lbfgs_iters,
            history_size=cfg.lbfgs_history,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        step_counter = {"n": 0}

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            total, log = _compute_loss(model, x_pde, x_bcl, x_bcr, x_ic, defaults, cfg)
            total.backward()
            step_counter["n"] += 1
            if step_counter["n"] % cfg.log_every == 0:
                history["iter"].append(cfg.adam_iters + step_counter["n"])
                history["total"].append(total.item())
                for k, v in log.items():
                    history[k].append(v)
            return total

        print("Switching to L-BFGS...")
        lbfgs.step(closure)

    return history
