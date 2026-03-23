import sys
import torch
import torch.nn as nn
from pathlib import Path

CRANPM_PATH = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe/cran_pm_site")
sys.path.insert(0, str(CRANPM_PATH))
from cranpm.models.global_branch import GlobalBranch
from cranpm.models.cross_attention import CrossAttentionBridge
from cranpm.models.decoder import CNNDecoder

from .time_embed import TimeEmbedder
from .local_branch_flow import LocalBranchFlow


class ARIAFlow(nn.Module):
    """
    ARIA-Flow: Conditional Flow Matching model for global PM2.5 downscaling.

    Mathematical framework (Conditional Flow Matching):
        x_t = (1-t)·x_0 + t·x_1        linear interpolation path
        v_θ(x_t, t, C) ≈ x_1 - x_0     predicted velocity field
        Loss: E‖v_θ(x_t, t, C) − (x_1−x_0)‖² (land pixels only)

    Inputs:
        - Condition C = ERA5+CAMS global (72ch, 168×280, 0.25°)
        - Proxy local (18ch, 512×512, 1km): CAMS + elevation + emission proxies
        - x_t (1ch, 512×512): noisy PM2.5 at flow time t
        - t ∈ [0,1]: flow time
        - lead_time ∈ {0,1}: same-day (0) or 24h forecast (1)

    Output:
        - v_t (1ch, 512×512): velocity field at time t
    """

    def __init__(
        self,
        era5_channels: int = 72,
        global_img_size: tuple = (168, 280),
        global_patch_size: int = 8,
        global_embed_dim: int = 768,
        global_depth: int = 8,
        global_num_heads: int = 12,
        local_proxy_channels: int = 18,
        local_img_size: tuple = (512, 512),
        local_patch_size: int = 16,
        local_embed_dim: int = 512,
        local_depth: int = 6,
        local_num_heads: int = 8,
        cross_num_heads: int = 8,
        cross_layers: int = 2,
        out_channels: int = 1,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path: float = 0.1,
        global_region_h: int = 7,
        global_region_w: int = 7,
        time_freq_dim: int = 256,
    ):
        super().__init__()

        # ── Time embedding ──────────────────────────────────────────────────────────
        # t ∈ [0,1] + lead_time → (B, local_embed_dim) for AdaLN conditioning
        self.time_embedder = TimeEmbedder(embed_dim=local_embed_dim, freq_dim=time_freq_dim)

        # ── Global branch (unchanged from ARIA) ─────────────────────────────────────
        # ERA5+CAMS → 735 tokens (21×35 patches) of dim=768
        self.global_branch = GlobalBranch(
            in_channels=era5_channels,
            img_size=global_img_size,
            patch_size=global_patch_size,
            embed_dim=global_embed_dim,
            depth=global_depth,
            num_heads=global_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=drop_path,
            region_h=global_region_h,
            region_w=global_region_w,
        )

        # ── Local branch (modified: 19ch, all blocks AdaLN) ─────────────────────────
        # Proxies(18) + x_t(1) → 1024 tokens (32×32 patches) of dim=512
        self.local_branch = LocalBranchFlow(
            in_channels=local_proxy_channels + 1,  # +1 for x_t
            img_size=local_img_size,
            patch_size=local_patch_size,
            embed_dim=local_embed_dim,
            depth=local_depth,
            num_heads=local_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=drop_path,
        )

        # ── Cross-attention bridge (unchanged) ──────────────────────────────────────
        # Local tokens (Q) attend to global tokens (K, V), wind-guided
        global_grid_h = global_img_size[0] // global_patch_size
        global_grid_w = global_img_size[1] // global_patch_size
        self.cross_attention = CrossAttentionBridge(
            local_dim=local_embed_dim,
            global_dim=global_embed_dim,
            num_heads=cross_num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=drop_rate,
            drop_path=drop_path,
            num_layers=cross_layers,
            global_grid_h=global_grid_h,
            global_grid_w=global_grid_w,
        )

        # ── CNN decoder (unchanged) ──────────────────────────────────────────────────
        # PixelShuffle ×4: 32→64→128→256→512 → v_t (1ch, 512×512)
        local_grid_h = local_img_size[0] // local_patch_size
        local_grid_w = local_img_size[1] // local_patch_size
        self.decoder = CNNDecoder(
            embed_dim=local_embed_dim,
            grid_h=local_grid_h,
            grid_w=local_grid_w,
            out_channels=out_channels,
            skip_channels=local_embed_dim,
        )

    def forward(self, era5, elevation_coarse, proxy_input, x_t, elevation_hires,
                t, lead_time, patch_center=None, wind_at_patch=None):
        """
        Args:
            era5:             (B, 72, 168, 280)  ERA5+CAMS global
            elevation_coarse: (B, H_era5, W_era5) coarse DEM
            proxy_input:      (B, 18, 512, 512)  CAMS + proxies
            x_t:              (B, 1,  512, 512)  noisy PM2.5 at flow time t
            elevation_hires:  (B, 512, 512)      1km DEM
            t:                (B,)               flow time in [0, 1]
            lead_time:        (B,)               0=J+0, 1=J+1
        Returns:
            v_t: (B, 1, 512, 512) predicted velocity field
        """
        # Time embedding: [t, lead_time] → (B, 512)
        time_emb = self.time_embedder(t, lead_time)

        # Global branch: ERA5+CAMS conditioning (no time embedding — pure condition)
        global_feats = self.global_branch(era5, elevation_coarse, lead_time)  # (B, 735, 768)

        # Local branch: proxies + x_t, conditioned on time_emb via AdaLN
        local_input = torch.cat([proxy_input, x_t], dim=1)                   # (B, 19, 512, 512)
        local_feats, skip = self.local_branch(local_input, elevation_hires, time_emb)

        # Cross-attention: local attends to global, wind-guided
        fused = self.cross_attention(
            local_feats, global_feats,
            patch_center=patch_center,
            wind_at_patch=wind_at_patch,
        )

        # CNN decoder → velocity field v_t
        v_t = self.decoder(fused, skip=skip)                                  # (B, 1, 512, 512)
        return v_t
