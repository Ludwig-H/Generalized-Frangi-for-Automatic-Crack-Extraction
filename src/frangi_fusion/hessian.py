# src/frangi_fusion/hessian.py

import numpy as np
from typing import Dict, List, Tuple
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert to float32 in [0,1]. Accepts HxW or HxWxC (any C>=1).
    For C>=3 uses luminance on first 3 channels; for C==2 averages channels.
    """
    if img.ndim == 2:
        g = img.astype(np.float32)
    elif img.ndim == 3:
        c = img.shape[2]
        arr = img.astype(np.float32)
        if c >= 3:
            w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            g = arr[..., :3].dot(w)
        elif c == 2:
            g = arr.mean(axis=2)
        else:
            g = arr[..., 0]
    else:
        raise ValueError("Unsupported image shape")
    g -= g.min()
    if g.max() > 0:
        g /= g.max()
    return g

def _order_by_abs(e1: np.ndarray, e2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure |e1| <= |e2| pixelwise."""
    swap = np.abs(e1) > np.abs(e2)
    if np.any(swap):
        e1c, e2c = e1.copy(), e2.copy()
        e1c[swap], e2c[swap] = e2[swap], e1[swap]
        return e1c, e2c
    return e1, e2

def _hessian_at_sigma(gray: np.ndarray, sigma: float) -> Dict[str, np.ndarray]:
    """
    Hessian with Gaussian derivatives; reflective boundaries to avoid edge artifacts.
    Eigenvalues ordered so that |e1| <= |e2|.
    """
    Hxx, Hxy, Hyy = hessian_matrix(
        gray,
        sigma=float(sigma),
        order='rc',
        use_gaussian_derivatives=True,
        mode='reflect',
        cval=0.0
    )
    try:
        e1, e2 = hessian_matrix_eigvals((Hxx, Hxy, Hyy))
    except TypeError:
        e1, e2 = hessian_matrix_eigvals(Hxx, Hxy, Hyy)

    e1, e2 = _order_by_abs(e1, e2)
    theta = 0.5 * np.arctan2(2 * Hxy, (Hxx - Hyy) + 1e-12)
    return {"Hxx": Hxx, "Hxy": Hxy, "Hyy": Hyy, "e1": e1, "e2": e2, "theta": theta}

def _spectral_norm(Hxx: np.ndarray, Hxy: np.ndarray, Hyy: np.ndarray) -> np.ndarray:
    """Spectral norm of the 2x2 Hessian per pixel."""
    a, b, c = Hxx, Hxy, Hyy
    tr = (a + c) / 2.0
    disc = np.sqrt(((a - c) / 2.0) ** 2 + b ** 2)
    l1 = tr - disc
    l2 = tr + disc
    spec = np.maximum(np.abs(l1), np.abs(l2))
    return np.maximum(spec, 1e-12)

def normalize_hessian(Hd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize H and its eigenvalues by the spectral norm.
    Keeps sign information intact and |e2| in [0,1].
    """
    spec = _spectral_norm(Hd["Hxx"], Hd["Hxy"], Hd["Hyy"])
    e1 = Hd["e1"] / spec
    e2 = Hd["e2"] / spec
    e1, e2 = _order_by_abs(e1, e2)
    return {
        "Hxx": Hd["Hxx"] / spec,
        "Hxy": Hd["Hxy"] / spec,
        "Hyy": Hd["Hyy"] / spec,
        "e1":  e1,
        "e2":  e2,
        "theta": Hd["theta"]
    }

def compute_hessians_per_scale(modality_gray: np.ndarray, sigmas: List[float]) -> List[Dict[str, np.ndarray]]:
    out = []
    for s in sigmas:
        Hd = _hessian_at_sigma(modality_gray, s)
        Hd = normalize_hessian(Hd)
        Hd["sigma"] = s
        out.append(Hd)
    return out

def fuse_hessians_per_scale(
    hessians_by_modality: Dict[str, List[Dict[str, np.ndarray]]],
    weights_by_modality: Dict[str, float]
) -> List[Dict[str, np.ndarray]]:
    """
    Per-scale fusion: H_total_sigma = sum_m w_m * H_m_sigma (then re-normalize).
    Eigenvalues are recomputed from the fused H and ordered by |.| (|e1|<=|e2|).
    """
    sigmas = [Hd["sigma"] for Hd in list(hessians_by_modality.values())[0]]
    fused = []
    for sidx, sigma in enumerate(sigmas):
        Hxx = None; Hxy = None; Hyy = None
        for mod, lst in hessians_by_modality.items():
            w = float(weights_by_modality.get(mod, 1.0))
            Hd = lst[sidx]
            if Hxx is None:
                Hxx = w * Hd["Hxx"]; Hxy = w * Hd["Hxy"]; Hyy = w * Hd["Hyy"]
            else:
                Hxx += w * Hd["Hxx"]; Hxy += w * Hd["Hxy"]; Hyy += w * Hd["Hyy"]

        a, b, c = Hxx, Hxy, Hyy
        tr = (a + c) / 2.0
        disc = np.sqrt(((a - c) / 2.0) ** 2 + b ** 2)
        l1 = tr - disc
        l2 = tr + disc
        l1, l2 = _order_by_abs(l1, l2)
        theta = 0.5 * np.arctan2(2 * b, (a - c) + 1e-12)
        spec = np.maximum(np.maximum(np.abs(l1), np.abs(l2)), 1e-12)
        fused.append({
            "Hxx": Hxx / spec, "Hxy": Hxy / spec, "Hyy": Hyy / spec,
            "e1": l1 / spec,   "e2": l2 / spec,   "theta": theta, "sigma": sigma
        })
    return fused
