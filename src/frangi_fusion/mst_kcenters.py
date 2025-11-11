
import numpy as np
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra

def mst_on_cluster(D: csr_matrix, cluster_idx: np.ndarray) -> csr_matrix:
    '''
    Compute MST on the subgraph induced by nodes in cluster_idx.
    Return the MST as a CSR matrix in the original indexing order restricted to the cluster (0..k-1).
    '''
    sub = D[cluster_idx][:, cluster_idx]
    sub_sym = sub + sub.T
    mst = minimum_spanning_tree(sub_sym)
    mst = mst + mst.T
    return mst.tocsr()

def _farthest_first_kcenters(mst: csr_matrix, k: int) -> List[int]:
    n = mst.shape[0]
    centers = [0]
    from scipy.sparse.csgraph import dijkstra
    dist = dijkstra(mst, directed=False, return_predecessors=False, indices=centers)[0]
    for _ in range(1,k):
        nxt = int(np.argmax(dist))
        centers.append(nxt)
        d_new = dijkstra(mst, directed=False, return_predecessors=False, indices=[nxt])[0]
        dist = np.minimum(dist, d_new)
    return centers

def kcenters_on_tree(mst: csr_matrix, k: int, objective: str = "max") -> List[int]:
    k = max(1, int(k))
    if k >= mst.shape[0]:
        return list(range(mst.shape[0]))
    return _farthest_first_kcenters(mst, k)

def _edges_set_from_path(predecessors: np.ndarray, src: int, dst: int):
    path = []
    u = dst
    while u != src and u != -9999:
        v = predecessors[u]
        if v < 0: break
        a,b = sorted((u,v))
        path.append((a,b))
        u = v
    return path[::-1]

def fault_graph_from_mst_and_kcenters(mst: csr_matrix, centers: List[int], weight_agg: str = "mean") -> csr_matrix:
    '''
    Build a small tree connecting the k-centers by reusing MST paths.
    Weight of an edge between centers is the mean/median of MST edge weights along the path.
    '''
    from scipy.sparse import csr_matrix
    n = mst.shape[0]
    if len(centers) <= 1:
        return csr_matrix((n,n))
    rows, cols, data = [], [], []
    from scipy.sparse.csgraph import dijkstra
    for i, src in enumerate(centers):
        dist, pred = dijkstra(mst, directed=False, return_predecessors=True, indices=src)
        for dst in centers[i+1:]:
            path_edges = _edges_set_from_path(pred, src, dst)
            if not path_edges:
                continue
            w = []
            for a,b in path_edges:
                wa = mst[a,b]
                if wa == 0: wa = mst[b,a]
                w.append(float(wa))
            val = np.median(w) if weight_agg=="median" else np.mean(w)
            rows.append(src); cols.append(dst); data.append(val)
    G = csr_matrix((data, (rows, cols)), shape=(n,n))
    G = G + G.T
    return G
