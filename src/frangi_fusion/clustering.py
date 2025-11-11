import numpy as np
from scipy.sparse import csr_matrix
import hdbscan

def hdbscan_from_sparse(D: csr_matrix,
                        min_cluster_size: int = 50,
                        min_samples: int = 5,
                        allow_single_cluster: bool = True,
                        expZ: float = 2.0):
    """
    Run HDBSCAN with metric='precomputed' on a dense square distance matrix.
    Keep dtype float64 to satisfy the C backend.
    """
    n = D.shape[0]
    if n == 0:
        return np.array([])
    # 1) keep float64
    Dc = D.copy().astype(np.float64, copy=False)
    # 2) d -> d^{expZ}
    Dc.data = Dc.data ** expZ
    # 3) densify in float64
    dense = Dc.toarray()  # stays float64
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        allow_single_cluster=allow_single_cluster,
        metric='precomputed'
    )
    labels = clusterer.fit_predict(dense)
    return labels
