import numpy as np
from scipy import sparse
from scipy import spatial
from sklearn import neighbors

def h(x):
    r = 1
    cr = 1/r
    return cr*np.exp(-x)*(x <= r)

def k_eps(x1, x2, eps):
    return h(np.norm(x1-x2)**2/eps)

def Q_eps(pts, r, eps):
    m = pts.shape[0]
    T = pts.shape[2]
    Q = sparse.coo_matrix((m,m), dtype=float)
    rang = np.sqrt(r*eps)
    for t in range(T):
        data = pts[:,t,:]
        # kdTree1 = spatial.KDTree(data)
        # kdTree2 = spatial.KDTree(data)
        # res = kdTree1.sparse_distance_matrix(kdTree2, max_distance=rang)
        res = sparse.csr_matrix(neighbors.radius_neighbors_graph(data, radius=rang, mode='distance'))
        idx_list = np.split(res.indices, res.indptr)[1:-1]

        return idx_list, res

def assemble_sim_matrix(idx, D, m, eps):
    lv = len(idx)
    x = np.zeros(lv)
    y = np.zeros(lv)
    v = np.zeros(lv)
    icurr = 0
    x = np.concatenate([idx*np.ones(val.size) for idx, val in enumerate(idx)])
    y = np.concatenate(idx)
    v = D.data

    return lv

