import numpy as np
from typing import Dict, List
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

def _order_by_abs(e1: np.ndarray, e2: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Ensure |e1| <= |e2| pixelwise by swapping when needed.
    """
    swap = np.abs(e1) > np.abs(e2)
    if np.any(swap):
        e1_new = e1.copy()
        e2_new = e2.copy()
        e1_new[swap], e2_new[swap] = e2[swap], e1[swap]
        return e1_new, e2_new
    return e1, e2

def _hessian_at_sigma(gray: np.ndarray, sigma: float) -> Dict[str, np.ndarray]:
    """
    Compute Hessian with Gaussian derivatives and reflective boundaries
    to avoid border artifacts. Eigenvalues are ordered by |.| so that |e1| <= |e2|.
    """
    Hxx, Hxy, Hyy = hessian_matrix(
        gray,
        sigma=float(sigma),
        order='rc',
        use_gaussian_derivatives=True,  # explicit, stable
        mode='reflect',                 # reduce border effects
        cval=0.0
    )

    # Newer scikit-image expects one argument; older versions accept three.
    try:
        e1, e2 = hessian_matrix_eigvals((Hxx, Hxy, Hyy))
    except TypeError:
        e1, e2 = hessian_matrix_eigvals(Hxx, Hxy, Hyy)

    # Order by absolute value: |e1| <= |e2|
    e1, e2 = _order_by_abs(e1, e2)

    # Orientation (principal curvature)
    theta = 0.5 * np.arctan2(2 * Hxy, (Hxx - Hyy) + 1e-12)
    return {"Hxx": Hxx, "Hxy": Hxy, "Hyy": Hyy, "e1": e1, "e2": e2, "theta": theta}

def _spectral_norm(Hxx: np.ndarray, Hxy: np.ndarray, Hyy: np.ndarray) -> np.ndarray:
    """
    Spectral norm of a symmetric 2x2 matrix per pixel.
    """
    a, b, c = Hxx, Hxy, Hyy
    tr = (a + c) / 2.0
    disc = np.sqrt(((a - c) / 2.0) ** 2 + b ** 2)
    lam1 = tr - disc
    lam2 = tr + disc
    spec = np.maximum(np.abs(lam1), np.abs(lam2))
    return np.maximum(spec, 1e-12)

def normalize_hessian(Hd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize H by its spectral norm. Keep eigenvalues ordered by |.| (|e1| <= |e2|).
    """
    spec = _spectral_norm(Hd["Hxx"], Hd["Hxy"], Hd["Hyy"])
    e1 = Hd["e1"] / np.maximum(np.abs(Hd["e2"]), 1e-12)
    e2 = Hd["e2"] / np.maximum(np.abs(Hd["e2"]), 1e-12)
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
    Eigenvalues are re-computed from the fused Hessian and ordered by |.| (|e1| <= |e2|).
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

        # Recompute fused eigenvalues, order by |.|, normalize
        a, b, c = Hxx, Hxy, Hyy
        tr = (a + c) / 2.0
        disc = np.sqrt(((a - c) / 2.0) ** 2 + b ** 2)
        lam1 = tr - disc
        lam2 = tr + disc
        lam1, lam2 = _order_by_abs(lam1, lam2)
        theta = 0.5 * np.arctan2(2 * b, (a - c) + 1e-12)
        spec = np.maximum(np.maximum(np.abs(lam1), np.abs(lam2)), 1e-12)
        fused.append({
            "Hxx": Hxx / spec, "Hxy": Hxy / spec, "Hyy": Hyy / spec,
            "e1": lam1 / spec, "e2": lam2 / spec, "theta": theta, "sigma": sigma
        })
    return fused
