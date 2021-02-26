import numpy as np
from scipy import sparse
from scipy import spatial
from sklearn import neighbors
from scipy.sparse.linalg import eigs, eigsh
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
def Q_eps(pts, r, eps, time_slices=4, load_cached=True,  dir_name=None):
    Q_file = 'data/' + str(dir_name) + '/Q_' + str(r) + '_' + str(eps) + '.npy'
    eigs_file = 'data/' + str(dir_name) + '/Q_eigs_' + str(r) + '_' + str(eps) + '.npy'
    if os.path.isfile(eigs_file):
        Q_eigs = np.load(eigs_file, allow_pickle=True)[()]
    else:
        if os.path.isfile(Q_file):
            Qeps = np.load(eigs_file, allow_pickle=True)[()]
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
            Qeps = Q / T
        Q_eigs = computeQ_eigVals(Qeps, k=15)
        np.save(eigs_file, Q_eigs) 

        
    return Q_eigs

def assemble_sim_matrix(idx, D, m, eps):
    x = np.concatenate([idx*np.ones(val.size) for idx, val in enumerate(idx)])
    y = np.concatenate(idx)
    v = D.data
    K = sparse.csr_matrix((np.exp(-v**2/eps), (x, y)), shape=(m,m)) 
    K = K - sparse.diags(K.diagonal()) + sparse.eye(m,m)  
    return K

def computeQ_eigVals(Qeps, k=15):
    eigs_res = eigsh(Qeps, k=k, which='LM')
    Q_eigenVals, Q_eigenVecs = eigs_res
    Q_eigenVals = np.real(Q_eigenVals)
    Q_eigenVecs = np.real(Q_eigenVecs)
    idx_pn = Q_eigenVals.argsort()[::-1]
    Q_eigenVals = Q_eigenVals[idx_pn]
    Q_eigenVecs = Q_eigenVecs[:, idx_pn]
    return [Q_eigenVals, Q_eigenVecs]
        

def cluster_eigVectors(eig_vecs, n_clusters=2):
    eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=1, keepdims=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(eig_vecs.T)
    label_space = kmeans.labels_
    return label_space
