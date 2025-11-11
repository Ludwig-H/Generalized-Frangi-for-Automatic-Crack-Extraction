
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

def _sim_elong(lam1_a, lam2_a, lam1_b, lam2_b, beta: float) -> np.ndarray:
    r = np.abs(lam1_a/np.maximum(np.abs(lam2_a),1e-12)) + np.abs(lam1_b/np.maximum(np.abs(lam2_b),1e-12))
    return np.exp(-0.5 * (r/(beta+1e-12))**2)

def _sim_intensity(lam2_a, lam2_b, Hxx, Hxy, Hyy, c: float) -> np.ndarray:
    prod = np.abs(lam2_a*lam2_b)
    return 1.0 - np.exp(-0.5 * (prod/(c+1e-12))**2)

def _sim_angle(theta_a, theta_b, ctheta: float) -> np.ndarray:
    diff = np.sin(theta_a - theta_b)
    return np.exp(-0.5 * (np.abs(diff)/(ctheta+1e-12))**2)

def build_frangi_similarity_graph(fused_hessians: List[Dict[str,np.ndarray]],
                                  beta: float,
                                  c: float,
                                  ctheta: float,
                                  R: int,
                                  candidate_mask: Optional[np.ndarray] = None,
                                  positive_ridges: bool = True
                                  ) -> Tuple[np.ndarray, List[List[int]], csr_matrix]:
    '''
    Build similarity on fused Hessian per scale (max over sigma).
    Only pairs (i,j) with Euclidean distance <= R are considered, restricted to candidate_mask if provided.
    Returns:
      coords: (N,2) array of candidate pixel coordinates (row,col)
      neighbors_list: list of lists of neighbor indices (graph as adjacency list)
      S: CSR sparse matrix with similarity values in [0,1]
    '''
    Hxxs = [Hd["Hxx"] for Hd in fused_hessians]
    Hxys = [Hd["Hxy"] for Hd in fused_hessians]
    Hyys = [Hd["Hyy"] for Hd in fused_hessians]
    e1s  = [Hd["e1"]  for Hd in fused_hessians]
    e2s  = [Hd["e2"]  for Hd in fused_hessians]
    thetas = [Hd["theta"] for Hd in fused_hessians]
    H, W = Hxxs[0].shape

    if candidate_mask is None:
        resp = np.max([np.abs(e2) for e2 in e2s], axis=0)
        thr = np.quantile(resp, 0.95)
        candidate_mask = resp >= thr
    coords = np.argwhere(candidate_mask)
    N = coords.shape[0]
    if N == 0:
        resp = np.max([np.abs(e2) for e2 in e2s], axis=0).reshape(-1)
        idx = np.argsort(resp)[-500:]
        coords = np.column_stack(np.unravel_index(idx, (H,W)))
        N = coords.shape[0]

    tree = cKDTree(coords[:, ::-1])
    pairs = tree.query_pairs(r=R, output_type='ndarray')
    if pairs.size == 0:
        return coords, [[] for _ in range(N)], csr_matrix((N,N))

    def sim_for_pairs_at_scale(sidx: int) -> np.ndarray:
        e1 = e1s[sidx]; e2 = e2s[sidx]; th = thetas[sidx]
        r0, c0 = coords[pairs[:,0],0], coords[pairs[:,0],1]
        r1, c1 = coords[pairs[:,1],0], coords[pairs[:,1],1]
        lam1_a = e1[r0, c0]; lam2_a = e2[r0, c0]
        lam1_b = e1[r1, c1]; lam2_b = e2[r1, c1]
        theta_a = th[r0, c0]; theta_b = th[r1, c1]
        if positive_ridges:
            valid = (lam2_a > 0) & (lam2_b > 0)
        else:
            valid = (lam2_a < 0) & (lam2_b < 0)
        sel = valid.astype(float)
        s1 = _sim_elong(lam1_a, lam2_a, lam1_b, lam2_b, beta)
        s2 = _sim_intensity(lam2_a, lam2_b, None, None, None, c)
        s3 = _sim_angle(theta_a, theta_b, ctheta)
        return sel * (s1 * s2 * s3)

    sims_scales = [sim_for_pairs_at_scale(i) for i in range(len(fused_hessians))]
    sims = np.max(np.vstack(sims_scales), axis=0)

    data = sims
    row = pairs[:,0]
    col = pairs[:,1]
    S = coo_matrix((data, (row, col)), shape=(N,N))
    S = S + S.T
    S = S.tocsr()

    neighbors = [[] for _ in range(N)]
    S_coo = S.tocoo()
    for i,j,v in zip(S_coo.row, S_coo.col, S_coo.data):
        if i != j and v > 0:
            neighbors[i].append(j)
    return coords, neighbors, S

def distances_from_similarity(S: csr_matrix) -> csr_matrix:
    Sd = S.copy().astype(np.float32)
    Sd.data = np.clip(1.0 - Sd.data, 0.0, 1.0)
    return Sd

def triangle_connectivity_graph(coords: np.ndarray, D: csr_matrix, max_triangles_per_node: int = 50) -> csr_matrix:
    '''
    Build a triangle-connectivity graph following a Vietoris-Rips style filtration:
    - Triangle filtration value = max of its three edge distances.
    - For each pixel (node), associate it to the smallest filtration triangle in which it appears;
      within that triangle, associate it to the smallest edge of that triangle.
    - Each triangle that appears connects the three associated edge-sets (union operation).
    Implementation notes:
    - We approximate triangle enumeration by, for each node u, picking up to 'max_triangles_per_node' neighbor pairs (v,w)
      where edges (u,v), (u,w), (v,w) exist.
    '''
    import itertools
    from collections import defaultdict
    from scipy.sparse import csr_matrix

    N = D.shape[0]
    D = D.tocsr()
    neighbors = [D.indices[D.indptr[i]:D.indptr[i+1]] for i in range(N)]
    weights   = [D.data[D.indptr[i]:D.indptr[i+1]] for i in range(N)]
    def edge_id(i,j):
        return (i,j) if i<j else (j,i)
    edge_w = {}
    for i in range(N):
        for j,w in zip(neighbors[i], weights[i]):
            if i<j:
                edge_w[(i,j)] = float(w)

    triangles = []
    for u in range(N):
        nbrs = neighbors[u]
        if len(nbrs) < 2:
            continue
        cnt = 0
        for v,w in itertools.combinations(nbrs, 2):
            if v==w or v==u or w==u: 
                continue
            a = edge_id(min(v,w), max(v,w))
            if a not in edge_w:
                continue
            e_uv = edge_id(u,v); e_uw = edge_id(u,w); e_vw = edge_id(min(v,w), max(v,w))
            if e_uv not in edge_w or e_uw not in edge_w or e_vw not in edge_w:
                continue
            d_uv = edge_w[e_uv]; d_uw = edge_w[e_uw]; d_vw = edge_w[e_vw]
            tri_val = max(d_uv, d_uw, d_vw)
            e_min = min([(d_uv, e_uv), (d_uw, e_uw), (d_vw, e_vw)], key=lambda x:x[0])[1]
            triangles.append((tri_val, (u,v,w), e_min))
            cnt += 1
            if cnt >= max_triangles_per_node:
                break

    if not triangles:
        return D

    best_tri_for_node = {}
    for tri_val, (u,v,w), e_min in triangles:
        for x in (u,v,w):
            if x not in best_tri_for_node or tri_val < best_tri_for_node[x][0]:
                best_tri_for_node[x] = (tri_val, e_min)

    edge_to_nodes = defaultdict(set)
    for n,(tri_val, e_min) in best_tri_for_node.items():
        edge_to_nodes[e_min].add(n)

    parent = list(range(N))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    def unite(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb:
            parent[rb]=ra

    for tri_val, (u,v,w), e_min in triangles:
        e_uv = (min(u,v), max(u,v)); e_uw = (min(u,w), max(u,w)); e_vw = (min(v,w), max(v,w))
        groups = [edge_to_nodes.get(e_uv,set()), edge_to_nodes.get(e_uw,set()), edge_to_nodes.get(e_vw,set())]
        all_nodes = set().union(*groups)
        if not all_nodes:
            continue
        base = next(iter(all_nodes))
        for x in all_nodes:
            unite(base, x)

    rows, cols, data = [], [], []
    for i in range(N):
        ci = find(i)
        for j,w in zip(neighbors[i], weights[i]):
            if i<j and find(j)==ci:
                rows.append(i); cols.append(j); data.append(w)
    M = csr_matrix((data, (rows, cols)), shape=(N,N))
    M = M + M.T
    return M
