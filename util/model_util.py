# Adapted from: https://github.com/hustvl/LightningDiT

import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sin-cos position embeddings; pos shape (M,) -> out shape (M, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
