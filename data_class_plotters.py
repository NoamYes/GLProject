import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.sparse.linalg import eigs
from scipy import sparse
import itertools

from utils import Q_eps, computeQ_eigVals, cluster_eigVectors

class DataAnalysis:
    def __init__(self, pts, dataset_name):
        self.pts = pts
        self.dataset_name = dataset_name 

    def eig_vals_plot(self, r, eps_list, fig=None, show=True):
        pts = self.pts
        markers = itertools.cycle((',', '+', '.', 'o', '*')) 
        colors = itertools.cycle(('b', 'r', 'y', 'g', 'm')) 
        if (fig == None):
            fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1, 2, 2)

        for eps in eps_list:
            Qeps_list = Q_eps(pts, r=r, eps=eps, load_cached=True, dir_name=self.dataset_name)
            Qeps = Qeps_list[-1]
            Q_eigVals, _  = computeQ_eigVals(Qeps, r=r, eps=eps, k=10, load_cached=True, dir_name=self.dataset_name)
            L_eigVals = (Q_eigVals - 1)/eps
            color = next(colors)
            marker=next(markers)
            ax1.scatter(range(Q_eigVals.size), Q_eigVals, s=20, c=color, marker=marker, label=r'$\epsilon=$'+str(eps))
            ax2.scatter(range(L_eigVals.size), L_eigVals, s=20, c=color, marker=marker, label=r'$\epsilon=$'+str(eps))      
            ax1.legend(loc='upper right')
            ax2.legend(loc='upper right')

        ax1.set_xlabel('n')
        ax1.set_ylabel(r'$\lambda_n$')
        ax1.set_title('Scatter eigenValues of ' + r'$Q_{\epsilon}$' + ' for various ' + r'$\epsilon$' + ' values')
        ax2.set_xlabel('n')
        ax2.set_ylabel(r'$\lambda_n$')
        ax2.set_title('Scatter eigenValues of ' + r'$L_{\epsilon}$' + ' for various ' + r'$\epsilon$' + ' values')
        if show == True:
            plt.show()
        return

    ## 3 - Eigenfunction at t = 0 and t = 19.5

    def plot_eigenfuncs_at_times(self, img_shape,  t_vec, r, eps_list, cmap=None, show=True):
        pts = self.pts
        m = pts.shape[0]
        pts_shape = (m,m)
        def computeLargeDiffSet(eigVals, n_largest=3):
            diffs = np.abs(eigVals[:-1] - eigVals[1:])
            # foundInds = diffs.argsort()[-n_largest:][::-1]
            mean_diff = np.mean(diffs)
            foundInds = np.where(diffs > mean_diff)
            return foundInds[0]

        eps_list = [0.0002]
        # eps_list = [0.0002, 0.0005, 0.001, 0.002]
        for eps in eps_list:
            Qeps_list = Q_eps(pts, r=r, eps=eps, load_cached=True, dir_name=self.dataset_name)
            Qeps_list = Qeps_list[::int(len(Qeps_list)/2)]
            time_slices = t_vec[-1]*range(len(Qeps_list))/len(Qeps_list)
            fig, axes = plt.subplots(len(Qeps_list), 1, figsize=(12,18), constrained_layout=True)
            for idx, Qeps in enumerate(Qeps_list):
                t = time_slices[idx]
                Q_eigVals, Q_eigVecs = computeQ_eigVals(Qeps, r=r, eps=eps, k=15, load_cached=True, dir_name=self.dataset_name)
                L_eigVals = (Q_eigVals - 1)/eps
                # deltaEigs = computeLargeDiffSet(L_eigVals, n_largest=3)
                deltaEigs = [2, 3, 8]
                eigFunc = Q_eigVecs[deltaEigs[0]].reshape(img_shape)
                eigFunc = np.flip(eigFunc, axis=0)
                im = axes[idx].imshow(np.real(eigFunc), cmap=cmap)
                axes[idx].set_title('Second Eigenfunction of ' + r'$Q_{\epsilon}$' + ' for time slice t = ' +str(t))
                axes[idx].set_xlabel('x')
                axes[idx].set_ylabel('y')
                # plt.colorbar(im)
            fig.suptitle(r'$\epsilon=$' + str(eps))
        if show == True:
            plt.show()


    ##  2 - Clustering 2 

    def cluster_2_labels(self, img_shape, t_vec, r, eps_list, cmap=None, show=True):
        pts = self.pts
        m = pts.shape[0]
        pts_shape = (m,m)
        for eps in eps_list:
            Qeps_list = Q_eps(pts, r=r, eps=eps, load_cached=True, dir_name=self.dataset_name)
            Qeps_list = Qeps_list[::int(len(Qeps_list)/2)]
            time_slices = t_vec[-1]*range(len(Qeps_list))/len(Qeps_list)
            fig, axes = plt.subplots(len(Qeps_list), 1, figsize=(12,18), constrained_layout=True)
            for idx, Qeps in enumerate(Qeps_list):
                t = time_slices[idx]
                Q_eigVals, Q_eigVecs = computeQ_eigVals(Qeps, r=r, eps=eps, k=15, load_cached=True, dir_name=self.dataset_name)
                L_eigVals = (Q_eigVals - 1)/eps
                # deltaEigs = computeLargeDiffSet(L_eigVals, n_largest=1)
                deltaEigs = [3, 4, 9]
                label_space = cluster_eigVectors(Q_eigVecs[deltaEigs], n_clusters=2)
                label_space = np.reshape(label_space, img_shape)
                label_space = np.flip(label_space, axis=0)
                im = axes[idx].imshow(np.real(label_space), cmap=cmap)
                axes[idx].set_title('2 Clustering of ' + r'$Q_{\epsilon}$' + ' for time slice t = ' +str(t))
                axes[idx].set_xlabel('x')
                axes[idx].set_ylabel('y')
                # plt.colorbar(im)
            fig.suptitle(r'$\epsilon=$' + str(eps))
        if show == True:
            plt.show()
