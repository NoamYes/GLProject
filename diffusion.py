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
    T = pts.shape[1]
    Q = sparse.coo_matrix((m,m), dtype=float)
    rang = np.sqrt(r*eps)
    for t in range(T):
        data = pts[:,t,:]
        res = sparse.csr_matrix(neighbors.radius_neighbors_graph(data, radius=rang, mode='distance'))
        idx_list = np.split(res.indices, res.indptr)[1:-1]
        K = assemble_sim_matrix(idx_list, res, m, eps)
        q = 1/K.sum(axis=1)
        q = np.squeeze(np.asarray(q))
        Pepsi = sparse.diags(q) @ K
        depsi = 1/Pepsi.sum(axis=0)
        depsi = np.squeeze(np.asarray(depsi))
        B = sparse.diags(depsi) @ Pepsi.T @ Pepsi
        Q = Q + B
    Q = Q/T   
    return Q

def assemble_sim_matrix(idx, D, m, eps):
    icurr = 0
    x = np.concatenate([idx*np.ones(val.size) for idx, val in enumerate(idx)])
    y = np.concatenate(idx)
    v = D.data
    K = sparse.csr_matrix((np.exp(-v**2/eps), (x, y)), shape=(m,m)) 
    K = K - sparse.diags(K.diagonal()) + sparse.eye(m,m)  
    return K

