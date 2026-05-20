from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from models.common import TimestepEmbedder, LabelEmbedder
from util.model_util import get_1d_sincos_pos_embed_from_grid


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(1)
    while scale.dim() < x.dim():
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class RawViTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, attn_drop: float, proj_drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(proj_drop),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_attn.unsqueeze(1) * attn_out
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class RawFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.proj = nn.Linear(hidden_size, patch_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.proj(x)


@dataclass(frozen=True)
class _RawViTSpec:
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float


RAW_VIT_SPECS: Dict[str, _RawViTSpec] = {
    'JiT-B/16': _RawViTSpec(hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0),
    'JiT-B/32': _RawViTSpec(hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0),
    'JiT-L/16': _RawViTSpec(hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4.0),
    'JiT-L/32': _RawViTSpec(hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4.0),
    'JiT-H/16': _RawViTSpec(hidden_size=1280, depth=32, num_heads=16, mlp_ratio=4.0),
    'JiT-H/32': _RawViTSpec(hidden_size=1280, depth=32, num_heads=16, mlp_ratio=4.0),
}


class RawViTDiffusion(nn.Module):
    """ViT-style diffusion backbone operating on EEG patches."""

    def __init__(
        self,
        model_name: str,
        num_channels: int,
        patch_size: int,
        target_length: int,
        num_classes: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if target_length % patch_size != 0:
            raise ValueError("target_length must be divisible by patch_size for raw ViT mode")
        if model_name not in RAW_VIT_SPECS:
            raise ValueError(f"Unsupported model '{model_name}' for raw ViT mode. Available: {sorted(RAW_VIT_SPECS)}")
        spec = RAW_VIT_SPECS[model_name]

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.target_length = target_length
        self.patch_num = target_length // patch_size
        self.hidden_size = spec.hidden_size
        self.output_shape = (num_channels, self.patch_num, patch_size)

        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.hidden_size)

        self.patch_embed = nn.Linear(patch_size, self.hidden_size)
        positions = np.arange(self.patch_num, dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, positions)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)

        self.blocks = nn.ModuleList([
            RawViTBlock(
                hidden_size=self.hidden_size,
                num_heads=spec.num_heads,
                mlp_ratio=spec.mlp_ratio,
                attn_drop=attn_dropout,
                proj_drop=proj_dropout,
            )
            for _ in range(spec.depth)
        ])
        self.final_layer = RawFinalLayer(self.hidden_size, patch_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("RawViTDiffusion expects tensors of shape (B, C, patches, patch_size)")
        bsz, channels, patch_num, patch_size = x.shape
        if channels != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels but received {channels}")
        if patch_num != self.patch_num or patch_size != self.patch_size:
            raise ValueError("Input patch dimensions do not match the configured target length/patch size")

        cond = self.t_embedder(t) + self.y_embedder(y)
        cond = cond.repeat_interleave(channels, dim=0)

        feats = self.patch_embed(x)
        feats = feats.view(bsz * channels, patch_num, self.hidden_size)
        feats = feats + self.pos_embed

        for block in self.blocks:
            feats = block(feats, cond)

        feats = self.final_layer(feats, cond)
        feats = feats.view(bsz, channels, patch_num, patch_size)
        return feats


__all__ = ['RawViTDiffusion', 'RAW_VIT_SPECS']
