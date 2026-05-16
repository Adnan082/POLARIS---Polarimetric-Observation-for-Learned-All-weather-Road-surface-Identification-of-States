"""
Stokes parameter computation from 4-angle polarization images.
Inputs must be float32 (convert from uint16 before calling).
"""

import numpy as np


def compute_stokes(
    i0: np.ndarray,
    i45: np.ndarray,
    i90: np.ndarray,
    i135: np.ndarray,
    eps: float = 1.0,
) -> dict:
    """
    Returns S0, S1, S2, DoLP, AoLP as float32 arrays with the same spatial shape.

    eps guards against division by zero on dark pixels (S0 ≈ 0).
    Keep eps=1.0 when working in raw uint16 scale; use 1e-6 for normalised float.
    """
    S0 = i0 + i90
    S1 = i0 - i90
    S2 = i45 - i135

    DoLP = np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + eps)
    DoLP = np.clip(DoLP, 0.0, 1.0)          # physically bounded [0, 1]
    AoLP = 0.5 * np.arctan2(S2, S1)         # radians in [-pi/2, pi/2]

    return {
        "S0": S0.astype(np.float32),
        "S1": S1.astype(np.float32),
        "S2": S2.astype(np.float32),
        "DoLP": DoLP.astype(np.float32),
        "AoLP": AoLP.astype(np.float32),
    }


def pack_polar_channels(stokes: dict) -> np.ndarray:
    """Stack S0/S1/S2/DoLP/sin(2·AoLP)/cos(2·AoLP) into (6, H, W) float32.

    AoLP is circular in [-pi/2, pi/2] so raw values are not fed to the model.
    sin/cos decomposition avoids the discontinuity and removes redundancy with S2.
    """
    sin_aolp = np.sin(2 * stokes["AoLP"]).astype(np.float32)
    cos_aolp = np.cos(2 * stokes["AoLP"]).astype(np.float32)
    return np.stack(
        [stokes["S0"], stokes["S1"], stokes["S2"], stokes["DoLP"], sin_aolp, cos_aolp],
        axis=0,
    )
