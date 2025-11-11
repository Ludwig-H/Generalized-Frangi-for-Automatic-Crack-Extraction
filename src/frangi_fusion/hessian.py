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

def _eigvals_from_hessian(Hxx: np.ndarray, Hxy: np.ndarray, Hyy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form eigenvalues of a 2x2 symmetric matrix."""
    tr = (Hxx + Hyy) / 2.0
    disc = np.sqrt(((Hxx - Hyy) / 2.0) ** 2 + Hxy ** 2)
    l1 = tr - disc
    l2 = tr + disc
    return l1, l2

def _hessian_raw(gray: np.ndarray, sigma: float) -> Dict[str, np.ndarray]:
    """
    Raw Hessian (Gaussian derivatives, reflective boundaries).
    Returns raw H and raw eigenvalues (ordered by |.|).
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
        e1_raw, e2_raw = hessian_matrix_eigvals((Hxx, Hxy, Hyy))
    except TypeError:
        e1_raw, e2_raw = hessian_matrix_eigvals(Hxx, Hxy, Hyy)
    e1_raw, e2_raw = _order_by_abs(e1_raw, e2_raw)
    theta = 0.5 * np.arctan2(2 * Hxy, (Hxx - Hyy) + 1e-12)
    return {"Hxx_raw": Hxx, "Hxy_raw": Hxy, "Hyy_raw": Hyy,
            "e1": e1_raw, "e2": e2_raw, "theta": theta}

def _spectral_norm(Hxx: np.ndarray, Hxy: np.ndarray, Hyy: np.ndarray) -> np.ndarray:
    """Spectral norm = max |eigenvalue| per pixel."""
    l1, l2 = _eigvals_from_hessian(Hxx, Hxy, Hyy)
    spec = np.maximum(np.abs(l1), np.abs(l2))
    return np.maximum(spec, 1e-12)

def _normalize_hessian_spectral(Hd: Dict[str, np.ndarray]) -> None:
    """
    Add spectral-normalized Hessian (Hxx,Hxy,Hyy) and its eigenvalues (e1s,e2s).
    Raw e1,e2 are untouched.
    """
    spec = _spectral_norm(Hd["Hxx_raw"], Hd["Hxy_raw"], Hd["Hyy_raw"])
    Hxx = Hd["Hxx_raw"] / spec
    Hxy = Hd["Hxy_raw"] / spec
    Hyy = Hd["Hyy_raw"] / spec
    l1s, l2s = _eigvals_from_hessian(Hxx, Hxy, Hyy)
    l1s, l2s = _order_by_abs(l1s, l2s)
    Hd["Hxx"] = Hxx; Hd["Hxy"] = Hxy; Hd["Hyy"] = Hyy
    Hd["e1s"] = l1s; Hd["e2s"] = l2s  # spectral-normalized eigs (souvent utiles pour l'orientation)

def _normalize_eigs_global_absmax(Hd: Dict[str, np.ndarray]) -> None:
    """
    Normalize raw eigenvalues by max(|e2|) over the whole matrix so e1n,e2n in [-1,1].
    """
    denom = float(np.max(np.abs(Hd["e2"])))
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    e1n = Hd["e1"] / denom
    e2n = Hd["e2"] / denom
    e1n, e2n = _order_by_abs(e1n, e2n)
    Hd["e1n"] = e1n
    Hd["e2n"] = e2n
    Hd["eig_norm_denom"] = denom  # pour info

def compute_hessians_per_scale(modality_gray: np.ndarray, sigmas: List[float]) -> List[Dict[str, np.ndarray]]:
    """
    Pour chaque sigma:
      - calcule H_raw et e1/e2 (bruts),
      - ajoute H normalisé spectralement (Hxx,Hxy,Hyy) + e1s/e2s,
      - ajoute e1n/e2n = e1/e2 / max(|e2|) (dans [-1,1]).
    """
    out = []
    for s in sigmas:
        Hd = _hessian_raw(modality_gray, s)
        _normalize_hessian_spectral(Hd)
        _normalize_eigs_global_absmax(Hd)
        Hd["sigma"] = s
        out.append(Hd)
    return out

def fuse_hessians_per_scale(
    hessians_by_modality: Dict[str, List[Dict[str, np.ndarray]]],
    weights_by_modality: Dict[str, float]
) -> List[Dict[str, np.ndarray]]:
    """
    Fusion par échelle (poids w_m):
      1) Fusion RAW: H_raw_total = Σ_m w_m * H_raw_m
      2) e1/e2 RAW du H_raw_total
      3) e1n/e2n = e1/e2 / max(|e2|) de la matrice ([-1,1])
      4) H total normalisé spectralement (Hxx,Hxy,Hyy) + e1s/e2s (optionnels)
    """
    first_key = next(iter(hessians_by_modality))
    sigmas = [Hd["sigma"] for Hd in hessians_by_modality[first_key]]

    fused = []
    for sidx, sigma in enumerate(sigmas):
        Hxx_raw = None; Hxy_raw = None; Hyy_raw = None
        for mod, lst in hessians_by_modality.items():
            w = float(weights_by_modality.get(mod, 1.0))
            Hd = lst[sidx]
            if Hxx_raw is None:
                Hxx_raw = w * Hd["Hxx_raw"]; Hxy_raw = w * Hd["Hxy_raw"]; Hyy_raw = w * Hd["Hyy_raw"]
            else:
                Hxx_raw += w * Hd["Hxx_raw"]; Hxy_raw += w * Hd["Hxy_raw"]; Hyy_raw += w * Hd["Hyy_raw"]

        e1_raw, e2_raw = _eigvals_from_hessian(Hxx_raw, Hxy_raw, Hyy_raw)
        e1_raw, e2_raw = _order_by_abs(e1_raw, e2_raw)

        Hd_fused = {"Hxx_raw": Hxx_raw, "Hxy_raw": Hxy_raw, "Hyy_raw": Hyy_raw,
                    "e1": e1_raw, "e2": e2_raw,
                    "theta": 0.5 * np.arctan2(2 * Hxy_raw, (Hxx_raw - Hyy_raw) + 1e-12),
                    "sigma": sigma}

        _normalize_eigs_global_absmax(Hd_fused)      # -> e1n, e2n in [-1,1]
        _normalize_hessian_spectral(Hd_fused)        # -> Hxx,Hxy,Hyy + e1s,e2s
        fused.append(Hd_fused)

    return fused
