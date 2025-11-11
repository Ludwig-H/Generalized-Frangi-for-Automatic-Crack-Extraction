
import numpy as np
from scipy.sparse import csr_matrix
import hdbscan

def hdbscan_from_sparse(D: csr_matrix,
                        min_cluster_size: int = 50,
                        min_samples: int = 5,
                        allow_single_cluster: bool = True,
                        expZ: float = 2.0):
    '''
    Run HDBSCAN with metric='precomputed' on a dense square distance matrix.
    For tractability, we densify only the subgraph nodes and cap size.
    '''
    n = D.shape[0]
    if n == 0:
        return np.array([])
    Dc = D.copy().astype(np.float32)
    Dc.data = Dc.data ** expZ
    dense = Dc.toarray()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                allow_single_cluster=allow_single_cluster,
                                metric='precomputed')
    labels = clusterer.fit_predict(dense)
    return labels
