import numpy as np, re, random
import imageio.v2 as iio
from skimage.io import imread as ski_imread
from importlib import reload
import sys, os

# ---------- robust readers / grayscale ----------
def _read_image_any(path):
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

# # ---------- monkey-patch repo utilities if present ----------
# try:
import .utils as U
import .hessian as H
# patch readers / grayscale
U._read_image = _read_image_any
U.to_gray_uint8 = to_gray_uint8

def _to_gray_float01(img):
    g = to_gray_uint8(img).astype(np.float32) / 255.0
    return g
H.to_gray = _to_gray_float01
reload(U); reload(H)
# except Exception as _e:
#     print("Patch local only (no package found). Proceeding with helper fns in notebook.")

# ---------- optional: safer loader that skips broken modalities ----------
import os, glob

def _is_image_file(p):
    return p.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff","bmp"))

def _extract_key(p):
    m = re.findall(r"\d+", os.path.basename(p))
    return m[-1] if m else os.path.basename(p)

def auto_discover_find_structure(root):
    all_imgs = [p for p in glob.glob(os.path.join(root, '**', '*.*'), recursive=True) if _is_image_file(p)]
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
    for k in buckets: buckets[k] = sorted(buckets[k])
    return buckets

def load_modalities_and_gt_by_index(struct, index):
    base_list = struct['label'] if struct['label'] else struct['intensity']
    assert base_list, "No images found in FIND root."
    index = index % len(base_list)
    key = _extract_key(base_list[index])
    out = {'paths':{}, 'arrays':{}}
    for k in ['intensity','range','fused','label']:
        cand = [p for p in struct.get(k,[]) if _extract_key(p)==key]
        if not cand:
            continue
        pth = cand[0]
        try:
            arr = _read_image_any(pth)
            out['paths'][k] = pth
            if k == 'label':
                g = to_gray_uint8(arr)
                out['arrays'][k] = (g > 127).astype(np.uint8) * 255
            else:
                out['arrays'][k] = to_gray_uint8(arr)
        except Exception as e:
            # skip unreadable modality
            print(f"[WARN] Skipping unreadable {k}: {pth} ({e})")
            continue
    return out
