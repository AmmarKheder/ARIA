import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from timm.layers import trunc_normal_

CRANPM_PATH = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe/cran_pm_site")
sys.path.insert(0, str(CRANPM_PATH))
from cranpm.models.topoflow_block import compute_patch_coords, compute_patch_elevations
from cranpm.utils.pos_embed import get_2d_sincos_pos_embed

from .adaln_block import AdaLNBlock, AdaLNTopoFlowBlock


class LocalBranchFlow(nn.Module):
    """
    Local ViT branch for ARIA-Flow.

    Input: 19 channels at 512×512 resolution:
        - 18 proxy channels: CAMS_t(6) + CAMS_prev(6) + elev(1) + roads(1) + lights(1) + pop(1) + lat(1) + lon(1)
        - 1 channel: x_t (noisy PM2.5 at flow time t)

    All transformer blocks are conditioned on time_emb via AdaLN:
        - Block 0: AdaLNTopoFlowBlock (topology-aware attention + AdaLN)
        - Blocks 1..5: AdaLNBlock (standard attention + AdaLN)

    Returns:
        tokens: (B, 1024, 512)   — for cross-attention + decoder
        skip:   (B, 512, 32, 32) — for CNN decoder skip connection
    """

    def __init__(
        self,
        in_channels: int = 19,
        img_size: tuple = (512, 512),
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        H, W = img_size
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # F.unfold + Linear — avoids MIOpen Conv2d numerical bug on MI250X
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.norm_embed = nn.LayerNorm(embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        # Block 0: topology-aware + AdaLN
        self.topo_block = AdaLNTopoFlowBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop_rate, attn_drop=drop_rate, drop_path=dpr[0],
            elevation_scale=500.0,
        )
        # Blocks 1..depth-1: standard AdaLN
        self.blocks = nn.ModuleList([
            AdaLNBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=drop_rate, drop_path=dpr[i],
            )
            for i in range(1, depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        pos = get_2d_sincos_pos_embed(self.embed_dim, self.grid_h, self.grid_w)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, local_input: torch.Tensor, elevation_hires: torch.Tensor,
                time_emb: torch.Tensor):
        """
        Args:
            local_input:     (B, 19, 512, 512)
            elevation_hires: (B, 512, 512)
            time_emb:        (B, embed_dim)  from TimeEmbedder
        Returns:
            tokens: (B, 1024, 512)
            skip:   (B, 512, 32, 32)
        """
        B = local_input.shape[0]

        patches = F.unfold(local_input, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)                                    # (B, N, C*p*p)
        x = self.patch_embed(patches)                                        # (B, N, D)
        skip = x.transpose(1, 2).reshape(B, self.embed_dim, self.grid_h, self.grid_w)

        x = self.norm_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        coords_2d = compute_patch_coords(self.img_size, self.patch_size, x.device).expand(B, -1, -1)
        elev_patches = compute_patch_elevations(elevation_hires, self.patch_size)

        # Block 0: topology-aware + time conditioning
        x = self.topo_block(x, coords_2d, elev_patches, time_emb)

        # Blocks 1..5: time conditioning only
        for blk in self.blocks:
            x = blk(x, time_emb)

        return self.norm(x), skip
