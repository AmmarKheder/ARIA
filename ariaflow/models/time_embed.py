import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for flow time t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / (half - 1)
        )
        # Scale t to [0, 1000] for meaningful frequency range
        args = t[:, None].float() * freqs[None] * 1000.0
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class TimeEmbedder(nn.Module):
    """
    Embeds flow time t ∈ [0,1] and lead_time ∈ {0,1} into a conditioning vector.

    Architecture:
        [sin/cos(t*1000), lead_time] → Linear → SiLU → Linear → embed_dim

    Output: (B, embed_dim) — used as AdaLN conditioning in LocalBranchFlow.
    """

    def __init__(self, embed_dim: int, freq_dim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(freq_dim)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim + 1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, lead_time: torch.Tensor) -> torch.Tensor:
        # t: (B,), lead_time: (B,)
        sin_emb = self.sinusoidal(t)                    # (B, freq_dim)
        lt = lead_time.float().unsqueeze(-1)            # (B, 1)
        x = torch.cat([sin_emb, lt], dim=-1)           # (B, freq_dim+1)
        return self.mlp(x)                              # (B, embed_dim)
