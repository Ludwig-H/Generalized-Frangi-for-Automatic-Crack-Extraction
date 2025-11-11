# Generalized Frangi with Multi-modal Fusion on FIND

This repository implements a **generalized Frangi** pipeline with **multi-modal fusion** (intensity + range) and evaluation on the **FIND** dataset.
It follows the ideas introduced at Gretsi and adapts them to a simple fused-Hessian setting, then builds a **Frangi similarity graph** at pixel level,
performs **HDBSCAN** clustering on the sparse distance graph, extracts an **MST** inside each cluster, and summarizes it with **k-centers** to obtain a compact **fault network**.

The repo contains:
- a clear and illustrated **Colab notebook**: `notebooks/FIND_Frangi_Fusion_Colab.ipynb`
- a small, commented **Python library** under `src/frangi_fusion/`
- a **batch script** to process many images and compare with **CrackSegDiff** outputs: `scripts/run_batch_find.py`

> Notes
> - The notebook downloads the FIND `data.zip` with `gdown` and unzips it, as requested.
> - The code supports **K=1** (plain Frangi distances) or **K=2** ("triangle-connectivity" based on a Rips filtration over triangles).
> - HDBSCAN parameters are fixed (min_cluster_size=50, min_samples=5, allow_single_cluster=True) and the distance transform `d -> d^{expZ}` uses `expZ=2`.
> - The comparison metrics are **Jaccard**, **Tversky** (α=1, β=0.5), and an **approximate 1-Wasserstein** distance using uniform masses on skeleton pixels.

## Installation (local)

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

Open the notebook:

```
notebooks/FIND_Frangi_Fusion_Colab.ipynb
```

The notebook will:
1. Download `data.zip` from Google Drive using `gdown`.
2. Unzip it.
3. Randomly pick one image (with a fixed seed), load **all modalities** (intensity, range, fused if available) and **ground truth**.
4. Convert each modality to **grayscale** for Hessian computation.
5. Compute the **Hessian** at multiple scales `Σ` per modality, **normalize** by spectral norm, then **fuse** H as a weighted sum (weights start uniform).
6. **Visualize** the Hessian maps with transparent color overlays.
7. Build the **Frangi similarity graph** on the fused Hessian, store it as **lists of lists** and as a **sparse CSR** distance graph with `distance = 1 - similarity`.
8. Optionally, for **K=2**, build a **triangle-connectivity** graph (Rips filtration with triangle value = max edge) and connect pixels via triangle edges.
9. Keep the **largest connected component** only.
10. Apply **HDBSCAN** with the required parameters to the sparse distance graph after the transform `d -> d^2`.
11. **Display** clusters over the base image.
12. For each cluster: compute **MST**, then compute **k-centers** with `k = max(3, N/100)` where `N` is the number of nodes in the cluster. Build an **arborescent fault graph** between k-centers using MST paths (edge weight = mean or median along the path).
13. Render a small **animation**: original image with the recovered fault network progressively appearing.
14. Threshold the final network at **τ=0.3** and display the result.
15. Compare against the **Lee skeletonization** of the ground truth mask and compute **Jaccard**, **Tversky**, and **Wasserstein** distances (on slightly thickened skeletons).

## Batch evaluation and CrackSegDiff comparison

Use the script below to randomly select **500** additional images and compare the generalized Frangi results with **CrackSegDiff** outputs.
You must provide a directory with CrackSegDiff predictions for the same FIND patches.

```bash
python -m src.frangi_fusion.ensure_paths   # optional: creates output folders
python scripts/run_batch_find.py   --find-root /path/to/FIND   --cracksegdiff-dir /path/to/CrackSegDiff/outputs   --num-images 500   --radius 5   --K 1   --expz 2   --out-csv results/find_batch_metrics.csv
```

If `--cracksegdiff-dir` is missing, the script will still process the **Frangi** side and skip CrackSegDiff comparison.

Internally we parallelize the heavy loops with **joblib** and show progress using **tqdm_joblib**.

## References (context)
- FIND dataset DOI: 10.5281/zenodo.6383044
- CrackSegDiff paper and code information are referenced in the notebook for fair comparison.

## License

MIT
