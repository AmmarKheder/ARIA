import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from timm.models.vision_transformer import Mlp
from timm.layers import DropPath

CRANPM_PATH = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe/cran_pm_site")
sys.path.insert(0, str(CRANPM_PATH))
from cranpm.models.topoflow_block import TopoFlowAttention


class AdaLNBlock(nn.Module):
    """
    Standard ViT block with Adaptive Layer Norm (AdaLN-Zero) conditioning.

    Given time_emb (B, dim), modulates the layer norms:
        x_norm = LayerNorm(x) * (1 + scale) + shift
        x = x + gate * Attention(x_norm)
        x = x + gate * FFN(LayerNorm(x))

    Zero-init of modulation weights → identity at init → stable start.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Attention using linear projections (avoids MIOpen Conv2d bug on MI250X)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # AdaLN-Zero: 6 modulation params per block
        # (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # Zero-init → zero modulation at start → model behaves like unmodulated ViT
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        d = D // H
        q = self.q_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        # Pre-scale q to avoid bf16 overflow before matmul
        attn = F.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1).to(x.dtype)
        attn = self.attn_drop(attn)
        return self.out_proj((attn @ v).transpose(1, 2).reshape(B, N, D))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # time_emb: (B, dim)
        mod = self.adaLN_mod(time_emb)                          # (B, 6*dim)
        s_a, sc_a, g_a, s_m, sc_m, g_m = mod.chunk(6, dim=-1)  # each (B, dim)

        def modulate(norm, z, shift, scale):
            return norm(z) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = x + g_a.unsqueeze(1) * self.drop_path(self._attn(modulate(self.norm1, x, s_a, sc_a)))
        x = x + g_m.unsqueeze(1) * self.drop_path(self.mlp(modulate(self.norm2, x, s_m, sc_m)))
        return x


class AdaLNTopoFlowBlock(nn.Module):
    """
    TopoFlowBlock with AdaLN conditioning.

    Keeps topology-aware attention (elevation bias + relative position bias from
    TopoFlowAttention) but replaces standard LayerNorm with AdaLN modulated by
    the flow time embedding.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 elevation_scale: float = 500.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.attn = TopoFlowAttention(
            dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=drop, elevation_scale=elevation_scale,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x: torch.Tensor, coords_2d: torch.Tensor,
                elevation_patches: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_mod(time_emb)
        s_a, sc_a, g_a, s_m, sc_m, g_m = mod.chunk(6, dim=-1)

        def modulate(norm, z, shift, scale):
            return norm(z) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x_norm = modulate(self.norm1, x, s_a, sc_a)
        x = x + g_a.unsqueeze(1) * self.drop_path(self.attn(x_norm, coords_2d, elevation_patches))
        x = x + g_m.unsqueeze(1) * self.drop_path(self.mlp(modulate(self.norm2, x, s_m, sc_m)))
        return x
