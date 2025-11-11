
import os, re, random, numpy as np
from glob import glob
from typing import Dict, List, Tuple, Optional
from PIL import Image

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def _is_image_file(p: str) -> bool:
    return p.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff","bmp"))

# def to_gray_uint8(img: np.ndarray) -> np.ndarray:
#     if img.ndim == 2:
#         g = img
#     elif img.ndim == 3:
#         # simple luminance
#         g = img[..., :3].dot(np.array([0.2989, 0.5870, 0.1140]))
#     else:
#         raise ValueError("Unsupported image shape")
#     g = g.astype(np.float32)
#     g = g - g.min()
#     if g.max() > 0:
#         g = g / g.max()
#     g = (g * 255.0).clip(0,255).astype(np.uint8)
#     return g

def to_gray_uint8(img):
    # Accept HxW or HxWxC (any C>=1). For C>=3 use luminance on first 3, C==2 average.
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
        else:  # C==1 or weird
            g = arr[..., 0]
    else:
        raise ValueError("Unsupported image shape")
    g -= g.min()
    if g.max() > 0:
        g /= g.max()
    return (g * 255).clip(0,255).astype(np.uint8)

# def _read_image(p: str) -> np.ndarray:
#     im = Image.open(p)
#     arr = np.array(im)
#     return arr
# ---------- robust readers / grayscale ----------
def _read_image(path):
    # 1) PIL
    try:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]
            return arr
    except Exception:
        pass
    # 2) imageio
    try:
        arr = iio.imread(path)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        return arr
    except Exception:
        pass
    # 3) skimage (tifffile backend)
    try:
        arr = ski_imread(path)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        return arr
    except Exception as e:
        raise e
        
def auto_discover_find_structure(root: str) -> Dict[str, List[str]]:
    '''
    Try to discover FIND dataset structure.
    Return mapping with keys: intensity, range, fused, label (each is a sorted list of file paths).
    '''
    all_imgs = [p for p in glob(os.path.join(root, '**', '*.*'), recursive=True) if _is_image_file(p)]
    buckets = {'intensity': [], 'range': [], 'fused': [], 'label': []}
    for p in all_imgs:
        low = p.lower().replace('\\','/')
        if any(k in low for k in ['label','labels','gt','groundtruth','ground_truth','mask']):
            buckets['label'].append(p)
        elif any(k in low for k in ['fused','fusion']):
            buckets['fused'].append(p)
        elif any(k in low for k in ['range','depth']):
            buckets['range'].append(p)
        elif any(k in low for k in ['intensity','gray','grayscale']):
            buckets['intensity'].append(p)
        else:
            buckets['intensity'].append(p)
    for k in buckets:
        buckets[k] = sorted(buckets[k])
    return buckets

def _extract_key(p: str) -> str:
    base = os.path.basename(p)
    m = re.findall(r'\d+', base)
    return m[-1] if m else base

def load_modalities_and_gt_by_index(struct: Dict[str,List[str]], index: int):
    '''
    Given discovered FIND structure, load a triplet (intensity, range, fused) if possible and a label.
    Return a dict with arrays and the matched paths.
    '''
    base_list = struct['label'] if struct['label'] else struct['intensity']
    if not base_list:
        raise RuntimeError('No images found in FIND root.')
    index = index % len(base_list)
    base_p = base_list[index]
    key = _extract_key(base_p)
    out = {'paths':{}, 'arrays':{}}
    for k in ['intensity','range','fused','label']:
        cand = [p for p in struct.get(k,[]) if _extract_key(p)==key]
        if cand:
            out['paths'][k] = cand[0]
            arr = _read_image(cand[0])
            out['arrays'][k] = to_gray_uint8(arr) if k!='range' else to_gray_uint8(arr)
    return out
