
import numpy as np
from typing import Dict, List, Tuple
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)/255.0
    elif img.ndim == 3:
        g = img[..., :3].dot(np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
        g = g.astype(np.float32)
        g -= g.min()
        if g.max() > 0:
            g /= g.max()
        return g
    else:
        raise ValueError("Unsupported image shape")

def _hessian_at_sigma(gray: np.ndarray, sigma: float) -> Dict[str,np.ndarray]:
    Hxx, Hxy, Hyy = hessian_matrix(gray, sigma=sigma, order='rc')
    e1, e2 = hessian_matrix_eigvals(Hxx, Hxy, Hyy)
    theta = 0.5 * np.arctan2(2*Hxy, (Hxx - Hyy) + 1e-12)
    return {"Hxx":Hxx, "Hxy":Hxy, "Hyy":Hyy, "e1":e1, "e2":e2, "theta":theta}

def _spectral_norm(Hxx, Hxy, Hyy) -> np.ndarray:
    a, b, c = Hxx, Hxy, Hyy
    tr = (a + c)/2.0
    disc = np.sqrt(((a - c)/2.0)**2 + b**2)
    lam1 = tr - disc
    lam2 = tr + disc
    spec = np.maximum(np.abs(lam1), np.abs(lam2))
    return np.maximum(spec, 1e-12)

def normalize_hessian(Hd: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    spec = _spectral_norm(Hd["Hxx"], Hd["Hxy"], Hd["Hyy"])
    return {
        "Hxx": Hd["Hxx"]/spec,
        "Hxy": Hd["Hxy"]/spec,
        "Hyy": Hd["Hyy"]/spec,
        "e1": Hd["e1"]/np.maximum(np.abs(Hd["e2"]),1e-12),
        "e2": Hd["e2"]/np.maximum(np.abs(Hd["e2"]),1e-12),
        "theta": Hd["theta"]
    }

def compute_hessians_per_scale(modality_gray: np.ndarray, sigmas: List[float]) -> List[Dict[str,np.ndarray]]:
    out = []
    for s in sigmas:
        Hd = _hessian_at_sigma(modality_gray, s)
        Hd = normalize_hessian(Hd)
        Hd["sigma"] = s
        out.append(Hd)
    return out

def fuse_hessians_per_scale(hessians_by_modality: Dict[str, List[Dict[str,np.ndarray]]],
                            weights_by_modality: Dict[str, float]) -> List[Dict[str,np.ndarray]]:
    '''
    Fuse per scale: H_total_sigma = sum_m w_m * H_m_sigma.
    Thetas are averaged by circular mean; e1,e2 recomputed from fused H to keep consistency.
    '''
    sigmas = [Hd["sigma"] for Hd in list(hessians_by_modality.values())[0]]
    fused = []
    for sidx, sigma in enumerate(sigmas):
        Hxx = None; Hxy = None; Hyy = None
        for mod, lst in hessians_by_modality.items():
            w = float(weights_by_modality.get(mod, 1.0))
            Hd = lst[sidx]
            if Hxx is None:
                Hxx = w*Hd["Hxx"]; Hxy = w*Hd["Hxy"]; Hyy = w*Hd["Hyy"]
            else:
                Hxx += w*Hd["Hxx"]; Hxy += w*Hd["Hxy"]; Hyy += w*Hd["Hyy"]
        a,b,c = Hxx, Hxy, Hyy
        tr = (a + c)/2.0
        disc = np.sqrt(((a - c)/2.0)**2 + b**2)
        lam1 = tr - disc
        lam2 = tr + disc
        theta = 0.5 * np.arctan2(2*b, (a - c) + 1e-12)
        spec = np.maximum(np.maximum(np.abs(lam1),np.abs(lam2)),1e-12)
        fused.append({"Hxx":Hxx/spec, "Hxy":Hxy/spec, "Hyy":Hyy/spec, "e1":lam1/spec, "e2":lam2/spec, "theta":theta, "sigma":sigma})
    return fused
