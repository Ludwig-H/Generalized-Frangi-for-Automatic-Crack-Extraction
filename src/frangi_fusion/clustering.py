# -*- coding: utf-8 -*-
"""
Sparse HDBSCAN on a CSR precomputed distance graph.

This module runs a HDBSCAN-like pipeline directly on a sparse distance graph:
  1) compute k-core distances per node from its existing sparse neighbors
  2) mutual reachability on existing edges only: mr(i,j)=max(core[i], core[j], d_ij)
  3) MST on the sparse MR graph (SciPy csgraph)
  4) build single-linkage merge tree; accumulate cluster stability
  5) simplified EOM selection: pick children if their total stability exceeds the parent's,
     subject to min_cluster_size. Otherwise pick the parent. Unselected points -> noise (-1).

This avoids ever materializing an NxN dense matrix.

Notes:
- Input matrix must be symmetric distances (we symmetrize defensively).
- Dtype is forced to float64 to match the C backends used in MST routines.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# -------------------- utilities --------------------

def _symmetrize_min_csr(D: csr_matrix) -> csr_matrix:
    """Return a symmetric CSR by taking the elementwise min between D and D.T."""
    DT = D.T.tocsr()
    # Keep entries that exist in either; for duplicates take min
    A = D.minimum(DT)  # elementwise min on overlap
    # For one-sided edges, fill with the other side value
    B = D.maximum(DT)  # has all edges present on at least one side
    # Where A has zeros but B has nonzeros, use B
    A = A + (B - A).maximum(0)
    A.eliminate_zeros()
    return A.tocsr()

def _kth_smallest_positive(values: np.ndarray, k: int) -> float:
    """k-th smallest strictly positive value in 1D array; fallback to max if not enough."""
    vals = values[values > 0]
    if vals.size == 0:
        return np.inf
    if vals.size < k:
        return float(np.max(vals))
    # partition is O(n)
    kth = np.partition(vals, k-1)[k-1]
    return float(kth)

def _core_distances_from_csr(D: csr_matrix, k: int) -> np.ndarray:
    """Core distance of each node = distance to its k-th nearest neighbor among existing edges."""
    n = D.shape[0]
    core = np.empty(n, dtype=np.float64)
    indptr, data = D.indptr, D.data
    for i in range(n):
        row = data[indptr[i]:indptr[i+1]]
        core[i] = _kth_smallest_positive(row, k) if k > 0 else 0.0
    return core

def _mutual_reachability_csr(D: csr_matrix, core: np.ndarray) -> csr_matrix:
    """Mutual reachability on the same sparsity pattern as D."""
    D = D.tocsr()
    indptr, indices, data = D.indptr, D.indices, D.data
    out_data = np.empty_like(data, dtype=np.float64)
    for i in range(D.shape[0]):
        s, e = indptr[i], indptr[i+1]
        js = indices[s:e]
        dij = data[s:e]
        # mr(i,j) = max(core[i], core[j], d_ij)
        out_data[s:e] = np.maximum(np.maximum(core[i], core[js]), dij)
    MR = csr_matrix((out_data, indices.copy(), indptr.copy()), shape=D.shape)
    MR.eliminate_zeros()
    return MR

# -------------------- HDBSCAN via MST --------------------

@dataclass
class ClusterNode:
    id: int
    left: int  # child cluster id or -1
    right: int
    parent: int  # parent cluster id or -1
    size: int
    birth_lambda: float   # lambda at which this cluster exists (1/d)
    last_lambda: float    # last processed lambda for stability accumulation
    stability: float
    members: set          # set of point indices in this cluster

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.sz = np.ones(n, dtype=np.int64)

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> Tuple[int, bool]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra, False
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.sz[ra] += self.sz[rb]
        return ra, True

def _mst_edges_from_sparse(MR: csr_matrix) -> List[Tuple[int,int,float]]:
    """Compute MST on MR and return list of undirected edges (i,j,w) sorted by w ascending."""
    # SciPy returns a directed tree; take upper/lower, then unify
    MST = minimum_spanning_tree(MR)  # works on CSR, yields CSR with float64
    # unify both directions
    coo = MST.tocoo()
    edges = {(int(i), int(j)): float(w) for i, j, w in zip(coo.row, coo.col, coo.data)}
    # ensure symmetry (MST is effectively undirected)
    undirected = []
    for (i, j), w in edges.items():
        if (j, i) in edges:
            w = min(w, edges[(j, i)])
        undirected.append((i, j, w))
    undirected.sort(key=lambda t: t[2])  # by weight ascending
    return undirected

def _build_cluster_tree_from_mst(n: int, mst_edges: List[Tuple[int,int,float]],
                                 min_cluster_size: int) -> Tuple[List[ClusterNode], int]:
    """
    Kruskal over MST edges to build single-linkage tree with stability accumulation.
    lambdas are 1/w (infinite at start).
    Returns (nodes, root_id).
    """
    # initialize singletons
    nodes: List[ClusterNode] = []
    point2cid = np.arange(n, dtype=np.int64)  # initial cluster id == point id
    for i in range(n):
        nodes.append(ClusterNode(
            id=i, left=-1, right=-1, parent=-1, size=1,
            birth_lambda=np.inf, last_lambda=np.inf, stability=0.0, members={i}
        ))

    uf = UnionFind(n)
    next_cid = n

    for i, j, w in mst_edges:
        # skip pathological zeros
        w = float(w) if w > 0 else 1e-12
        lam = 1.0 / w

        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue

        ci, cj = point2cid[ri], point2cid[rj]
        ni, nj = nodes[ci], nodes[cj]

        # accumulate stability for children up to current lambda
        ni.stability += ni.size * (ni.last_lambda - lam)
        nj.stability += nj.size * (nj.last_lambda - lam)
        ni.last_lambda = lam
        nj.last_lambda = lam

        # create parent cluster
        parent_cid = next_cid; next_cid += 1
        members = ni.members | nj.members
        parent = ClusterNode(
            id=parent_cid, left=ci, right=cj, parent=-1,
            size=ni.size + nj.size,
            birth_lambda=lam, last_lambda=lam, stability=0.0,
            members=members
        )
        nodes.append(parent)

        # link children -> parent
        nodes[ci].parent = parent_cid
        nodes[cj].parent = parent_cid

        # union-find structure
        new_root, _ = uf.union(ri, rj)
        point2cid[new_root] = parent_cid

    # final stability accumulation to lambda=0 for the root component(s)
    # find cluster ids with no parent
    root_ids = [nd.id for nd in nodes if nd.parent == -1]
    for rid in root_ids:
        nd = nodes[rid]
        nd.stability += nd.size * (nd.last_lambda - 0.0)

    # If the MST produced multiple components (shouldn't, if MR is connected),
    # we merge them virtually with lambda=0 to define a single root.
    if len(root_ids) == 1:
        root_cid = root_ids[0]
    else:
        root_cid = next_cid; next_cid += 1
        members = set().union(*[nodes[r].members for r in root_ids])
        parent = ClusterNode(
            id=root_cid, left=-1, right=-1, parent=-1,
            size=len(members), birth_lambda=0.0, last_lambda=0.0,
            stability=0.0, members=members
        )
        nodes.append(parent)
        for r in root_ids:
            nodes[r].parent = root_cid

    return nodes, root_cid

def _select_clusters_eom(nodes: List[ClusterNode], root_id: int,
                         min_cluster_size: int, allow_single_cluster: bool) -> List[int]:
    """
    Simplified EOM: post-order, pick children if they have higher total stability than the parent,
    otherwise pick the parent. Enforce min_cluster_size. Disallow selecting the artificial root
    unless allow_single_cluster and it meets size.
    """
    # Build children list
    children: Dict[int, List[int]] = {}
    for nd in nodes:
        if nd.left != -1:
            children.setdefault(nd.id, [])
            children[nd.id] += [nd.left, nd.right]
        if nd.parent != -1:
            children.setdefault(nd.parent, [])

    # Post-order traversal to compute best stability sum
    sys_stack = [root_id]
    post = []
    seen = set()
    while sys_stack:
        u = sys_stack.pop()
        if u in seen:
            post.append(u)
            continue
        seen.add(u)
        sys_stack.append(u)
        for v in children.get(u, []):
            sys_stack.append(v)

    best_sum: Dict[int, float] = {}
    pick_children: Dict[int, bool] = {}
    for u in post:
        nd = nodes[u]
        child_ids = children.get(u, [])
        if not child_ids:
            best_sum[u] = nd.stability if nd.size >= min_cluster_size else 0.0
            pick_children[u] = False
            continue
        # sum of children's bests
        s_children = 0.0
        for v in child_ids:
            s_children += best_sum.get(v, 0.0)
        self_val = nd.stability if nd.size >= min_cluster_size else 0.0
        if s_children > self_val:
            best_sum[u] = s_children
            pick_children[u] = True
        else:
            best_sum[u] = self_val
            pick_children[u] = False

    # Recover selected cluster ids
    selected: List[int] = []
    def collect(u: int):
        nd = nodes[u]
        child_ids = children.get(u, [])
        if not child_ids:
            if nd.size >= min_cluster_size and nd.stability > 0:
                selected.append(u)
            return
        if pick_children[u]:
            for v in child_ids:
                collect(v)
        else:
            if nd.size >= min_cluster_size and nd.stability > 0:
                selected.append(u)

    collect(root_id)

    # Filter root if not allowed
    if not allow_single_cluster and root_id in selected:
        selected.remove(root_id)
    return selected

def _assign_labels_from_selection(n: int, nodes: List[ClusterNode], selected: List[int]) -> np.ndarray:
    """Assign each point to the smallest selected cluster that contains it; others are noise (-1)."""
    labels = -np.ones(n, dtype=np.int32)
    selected_set = set(selected)
    # sort selected by cluster size ascending to assign the most specific first
    selected_sorted = sorted(selected, key=lambda cid: nodes[cid].size)
    cur_label = 0
    for cid in selected_sorted:
        for p in nodes[cid].members:
            if labels[p] == -1:
                labels[p] = cur_label
        cur_label += 1
    return labels

# -------------------- public API --------------------

def hdbscan_from_sparse(
    D: csr_matrix,
    min_cluster_size: int = 50,
    min_samples: int = 5,
    allow_single_cluster: bool = True,
    expZ: float = 2.0
) -> np.ndarray:
    """
    Run a HDBSCAN-like clustering directly on a sparse CSR distance matrix.

    Parameters
    ----------
    D : csr_matrix
        Sparse, symmetric distance graph (only neighbors present as edges).
        Will be converted to float64 and symmetrized.
    min_cluster_size : int
    min_samples : int
        k for core-distances (k-th neighbor in the *sparse* neighborhood).
    allow_single_cluster : bool
    expZ : float
        Apply d -> d**expZ before clustering (as in your pipeline).

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels; -1 for noise.
    """
    if not isinstance(D, csr_matrix):
        raise TypeError("D must be a scipy.sparse.csr_matrix")
    if D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")

    # keep float64 to satisfy low-level routines
    D = D.astype(np.float64, copy=False)
    # optional power transform
    if expZ is not None and expZ != 1.0:
        D.data **= float(expZ)

    # symmetrize defensively
    D = _symmetrize_min_csr(D)

    # restrict to largest connected component (you le fais déjà en amont normalement)
    # SciPy's MST handles components independently; we proceed globally.

    # core distances from sparse neighbors
    k = int(max(1, min_samples))
    core = _core_distances_from_csr(D, k)

    # mutual reachability on sparse pattern
    MR = _mutual_reachability_csr(D, core)

    # MST edges
    edges = _mst_edges_from_sparse(MR)
    if len(edges) == 0:
        return -np.ones(D.shape[0], dtype=np.int32)

    # build merge tree and select clusters
    nodes, root_id = _build_cluster_tree_from_mst(D.shape[0], edges, min_cluster_size)
    selected = _select_clusters_eom(nodes, root_id, min_cluster_size, allow_single_cluster)
    labels = _assign_labels_from_selection(D.shape[0], nodes, selected)
    return labels
