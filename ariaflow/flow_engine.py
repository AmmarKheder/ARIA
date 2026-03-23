import torch


class FlowEngine:
    """
    Conditional Flow Matching (CFM) engine.

    Linear interpolation probability path:
        x_t = (1 - t) · x_0 + t · x_1

    Target velocity (constant along path):
        u_t(x) = x_1 - x_0

    Training loss (land-only):
        L_FM = E_{t, x_1, x_0} ‖ v_θ(x_t, t, C) − (x_1 − x_0) ‖²

    Inference:
        Euler ODE integration from x_0 ~ N(0,I) to x_1 (PM2.5 prediction)
        in N steps (N=10 sufficient for FM, unlike diffusion which needs 100+)
    """

    def __init__(self, sigma_min: float = 1e-4):
        # sigma_min: small noise floor to avoid degenerate paths
        self.sigma_min = sigma_min

    # ── Training ────────────────────────────────────────────────────────────────

    def get_train_tuple(self, x_1: torch.Tensor):
        """
        Sample one training tuple for flow matching.

        Args:
            x_1: (B, 1, H, W) normalized GHAP PM2.5 (ground truth)
        Returns:
            t:        (B,)          flow time sampled from U[0,1]
            x_0:      (B, 1, H, W) Gaussian noise
            x_t:      (B, 1, H, W) interpolated sample at time t
            target_v: (B, 1, H, W) target velocity = x_1 - x_0
        """
        B, C, H, W = x_1.shape
        device = x_1.device

        t   = torch.rand(B, device=device, dtype=x_1.dtype)
        x_0 = torch.randn_like(x_1)

        t_bchw = t.view(B, 1, 1, 1)
        x_t      = (1.0 - t_bchw) * x_0 + t_bchw * x_1
        target_v = x_1 - x_0

        return t, x_0, x_t, target_v

    def compute_loss(self, v_pred: torch.Tensor, target_v: torch.Tensor,
                     land_mask: torch.Tensor, ocean_weight: float = 0.1) -> torch.Tensor:
        """
        Flow matching MSE loss.
        - Land pixels (PM2.5 > 0): full weight
        - Ocean pixels: small penalty (ocean_weight=0.1) to prevent
          coastline artifacts and force v→0 over sea (target PM2.5 = 0).

        Args:
            v_pred:       (B, 1, H, W) predicted velocity field
            target_v:     (B, 1, H, W) target velocity x_1 - x_0
            land_mask:    (B, 1, H, W) bool — True = land pixel
            ocean_weight: weight on ocean pixels (default 0.1)
        Returns:
            scalar loss
        """
        diff_sq = (v_pred - target_v) ** 2
        ocean_mask = ~land_mask

        n_land  = land_mask.float().sum().clamp(min=1.0)
        n_ocean = ocean_mask.float().sum().clamp(min=1.0)

        land_loss  = (diff_sq * land_mask.float()).sum()  / n_land
        ocean_loss = (diff_sq * ocean_mask.float()).sum() / n_ocean

        return land_loss + ocean_weight * ocean_loss

    # ── Inference ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, model, era5, elevation_coarse, proxy_input, elevation_hires,
               lead_time, patch_center=None, wind_at_patch=None,
               n_steps: int = 20, return_trajectory: bool = False):
        """
        Euler ODE integration: x_0 → x_1 (PM2.5 prediction).

        dx/dt = v_θ(x_t, t, C)   integrated with uniform Euler steps.

        Args:
            model:    ARIAFlow instance (or EMA model)
            n_steps:  number of Euler steps (10–20 sufficient for FM)
        Returns:
            x_1_pred: (B, 1, H, W) predicted PM2.5 map (normalized log-space)
        """
        B = era5.shape[0]
        device = era5.device
        H, W = proxy_input.shape[2], proxy_input.shape[3]

        # Start from Gaussian noise
        x = torch.randn(B, 1, H, W, device=device, dtype=era5.dtype)

        dt = 1.0 / n_steps
        trajectory = [x.clone()] if return_trajectory else None

        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device, dtype=era5.dtype)
            v = model(
                era5=era5,
                elevation_coarse=elevation_coarse,
                proxy_input=proxy_input,
                x_t=x,
                elevation_hires=elevation_hires,
                t=t,
                lead_time=lead_time,
                patch_center=patch_center,
                wind_at_patch=wind_at_patch,
            )
            x = x + dt * v
            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_ensemble(self, model, n_samples: int = 10, **kwargs):
        """
        Run inference N times from different x_0 → uncertainty quantification.

        Returns:
            mean: (B, 1, H, W)  ensemble mean prediction
            std:  (B, 1, H, W)  spread / epistemic uncertainty
        """
        preds = torch.stack([self.sample(model, **kwargs) for _ in range(n_samples)])
        return preds.mean(dim=0), preds.std(dim=0)
