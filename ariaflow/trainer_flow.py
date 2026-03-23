import math
import torch
import pytorch_lightning as pl
from copy import deepcopy

from .models.model_ariaflow import ARIAFlow
from .flow_engine import FlowEngine


class ARIAFlowLightning(pl.LightningModule):
    """
    PyTorch Lightning trainer for ARIA-Flow.

    Training: flow matching loss (land-only MSE on velocity field)
    Validation: Euler ODE with EMA model → RMSE in µg/m³
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        mc = config["model"]
        tc = config["train"]
        dc = config["data"]

        self.model = ARIAFlow(
            era5_channels=mc.get("era5_channels", 72),
            global_img_size=tuple(mc.get("global_img_size", [168, 280])),
            global_patch_size=mc.get("global_patch_size", 8),
            global_embed_dim=mc.get("global_embed_dim", 768),
            global_depth=mc.get("global_depth", 8),
            global_num_heads=mc.get("global_num_heads", 12),
            local_proxy_channels=mc.get("local_proxy_channels", 18),
            local_img_size=tuple(mc.get("local_img_size", [512, 512])),
            local_patch_size=mc.get("local_patch_size", 16),
            local_embed_dim=mc.get("local_embed_dim", 512),
            local_depth=mc.get("local_depth", 6),
            local_num_heads=mc.get("local_num_heads", 8),
            cross_num_heads=mc.get("cross_num_heads", 8),
            cross_layers=mc.get("cross_layers", 2),
            mlp_ratio=mc.get("mlp_ratio", 4.0),
            drop_rate=mc.get("drop_rate", 0.1),
            drop_path=mc.get("drop_path", 0.1),
            global_region_h=mc.get("global_region_h", 7),
            global_region_w=mc.get("global_region_w", 7),
        )

        # EMA: exponential moving average of weights for stable validation
        self.ema_decay = tc.get("ema_decay", 0.9999)
        self.ema_model = deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.flow = FlowEngine()

        # Normalization: log1p(PM2.5) normalized
        self.ghap_mean = dc.get("ghap_mean", 2.416)
        self.ghap_std  = dc.get("ghap_std",  0.800)

        self._warmup_steps = tc.get("warmup_steps", 10000)
        self._lr = tc.get("learning_rate", 5e-5)
        self._min_lr = tc.get("min_lr", 1e-6)
        self._total_epochs = tc.get("epochs", 300)
        self._steps_per_epoch = tc.get("steps_per_epoch", 2200)

    # ── EMA update ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_ema(self):
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    # ── Land mask ───────────────────────────────────────────────────────────────

    def _land_mask(self, target: torch.Tensor) -> torch.Tensor:
        # Land = PM2.5 > 0 → in normalized log-space: target > -mean/std ≈ -3.02
        threshold = -self.ghap_mean / self.ghap_std
        return (target > threshold)  # (B, 1, H, W) bool

    # ── Training ────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        target = batch["target"]                            # (B, 1, H, W)
        t, x_0, x_t, target_v = self.flow.get_train_tuple(target)

        v_pred = self.model(
            era5=batch["era5"],
            elevation_coarse=batch["elevation_coarse"],
            proxy_input=batch["local_input"],               # (B, 18, 512, 512)
            x_t=x_t,
            elevation_hires=batch["elevation_hires"],
            t=t,
            lead_time=batch["lead_time"],
            patch_center=batch.get("patch_center"),
            wind_at_patch=batch.get("wind_at_patch"),
        )

        if torch.isnan(v_pred).any():
            # Keep graph connected — avoids DDP allreduce deadlock on empty grad
            return sum(p.sum() * 0 for p in self.model.parameters())

        land_mask = self._land_mask(target)
        loss = self.flow.compute_loss(v_pred, target_v, land_mask)

        if torch.isnan(loss):
            return sum(p.sum() * 0 for p in self.model.parameters())

        self._update_ema()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ── Validation ──────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        target = batch["target"]

        # FM loss with training model (fast, no ODE needed)
        t, x_0, x_t, target_v = self.flow.get_train_tuple(target)
        v_pred = self.model(
            era5=batch["era5"],
            elevation_coarse=batch["elevation_coarse"],
            proxy_input=batch["local_input"],
            x_t=x_t,
            elevation_hires=batch["elevation_hires"],
            t=t,
            lead_time=batch["lead_time"],
        )
        land_mask = self._land_mask(target)
        loss = self.flow.compute_loss(v_pred, target_v, land_mask)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Full ODE with EMA model every 10 val batches (expensive: N forward passes)
        if batch_idx % 10 == 0:
            x_pred = self.flow.sample(
                self.ema_model,
                era5=batch["era5"],
                elevation_coarse=batch["elevation_coarse"],
                proxy_input=batch["local_input"],
                elevation_hires=batch["elevation_hires"],
                lead_time=batch["lead_time"],
                n_steps=10,
            )
            rmse = self._compute_rmse(x_pred, target)
            self.log("val/rmse", rmse, prog_bar=True, sync_dist=True)

        return loss

    # ── RMSE in µg/m³ ───────────────────────────────────────────────────────────

    def _compute_rmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f   = torch.nan_to_num(pred.float(),   nan=0.0, posinf=0.0, neginf=0.0)
        target_f = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pred_ug   = torch.expm1((pred_f   * self.ghap_std + self.ghap_mean).clamp(0, 8))
        target_ug = torch.expm1((target_f * self.ghap_std + self.ghap_mean).clamp(0, 8))
        return torch.sqrt(((pred_ug - target_ug) ** 2).mean())

    # ── Optimizer ───────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            weight_decay=self.config["train"].get("weight_decay", 0.05),
            betas=(0.9, 0.999),
        )

        total_steps = self._total_epochs * self._steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < self._warmup_steps:
                return step / max(1, self._warmup_steps)
            progress = (step - self._warmup_steps) / max(1, total_steps - self._warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            min_ratio = self._min_lr / self._lr
            return max(min_ratio, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
