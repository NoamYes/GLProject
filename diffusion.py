import numpy as np
from scipy import sparse
from scipy import spatial
from sklearn import neighbors
from scipy.sparse.linalg import eigs
import os.path
from tqdm import tqdm
from sklearn.cluster import KMeans

def h(x):
    r = 1
    cr = 1/r
    return cr*np.exp(-x)*(x <= r)

def k_eps(x1, x2, eps):
    return h(np.norm(x1-x2)**2/eps)

## Returns a list of Q_eps for given frequency (every "freq" samples)
def Q_eps(pts, r, eps, load_cached=True, sample_freq=50):
    my_file = 'data/Q_' + str(r) + '_' + str(eps) + '.npy'
    Q_list = []
    if os.path.isfile(my_file):
        Q_list = np.load(my_file, allow_pickle=True)[()]
    else:
        m = pts.shape[0]
        T = pts.shape[1]
        Q = sparse.coo_matrix((m,m), dtype=float)
        rang = np.sqrt(r*eps)
        for t in tqdm(range(T)):
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
            if not (t % sample_freq) : Q_list.append(Q/(t+1))
        np.save(my_file, Q_list) 
        
    return Q_list

def assemble_sim_matrix(idx, D, m, eps):
    x = np.concatenate([idx*np.ones(val.size) for idx, val in enumerate(idx)])
    y = np.concatenate(idx)
    v = D.data
    K = sparse.csr_matrix((np.exp(-v**2/eps), (x, y)), shape=(m,m)) 
    K = K - sparse.diags(K.diagonal()) + sparse.eye(m,m)  
    return K

def computeQ_eigVals(Qeps, r, eps, k=15, load_cached=True):
    my_file = 'data/eigs_' + str(r) + '_' + str(eps) + '_k_' +str(k) + '.npy'
    if os.path.isfile(my_file):
        Q_eigenVals, Q_eigenVecs = np.load(my_file, allow_pickle=True)
    else:
        eigs_res = eigs(Qeps, k=k, which='LM')
        np.save(my_file, eigs_res)
        Q_eigenVals, Q_eigenVecs = eigs_res
    idx_pn = Q_eigenVals.argsort()[::-1]
    Q_eigenVals = np.real(Q_eigenVals[idx_pn])
    Q_eigenVecs = np.real(Q_eigenVecs[:, idx_pn])
    return Q_eigenVals, Q_eigenVecs.T
        

def cluster_eigVectors(eig_vecs, space_pts, n_clusters=2):
    eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=1, keepdims=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(eig_vecs)
    label_space = kmeans.predict(space_pts.T)
