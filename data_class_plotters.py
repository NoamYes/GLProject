import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.sparse.linalg import eigs
from scipy import sparse
import itertools

from utils import Q_eps, computeQ_eigVals, cluster_eigVectors

class DataAnalysis:
    def __init__(self, pts, dataset_name, t_vec, img_shape, largest_gap_eigs, time_samples):
        self.pts = pts
        self.dataset_name = dataset_name 
        self.t_vec = t_vec
        self.img_shape = img_shape
        self.largest_gap_eigs = largest_gap_eigs
        self.T = t_vec.size
        self.time_slices = np.linspace(0, t_vec.size-5, time_samples).astype('int')


    def animate_data(self, fig=None, cmap='turbo', show=True, save=False):
        pts = self.pts
        t_vec = self.t_vec
        if fig == None:
            fig, ax = plt.subplots()
        color_func = pts[:,0,0]
        scat = ax.scatter(pts[:,0,0], pts[:,0,1], c=color_func, s=10, cmap=cmap)

        def connect(i):
            scat.set_offsets(pts[:,i,:].reshape(-1,2))
            return scat, 
        anim = animation.FuncAnimation(fig, connect, range(t_vec.size), interval=50)
        if save == True:
            # writer = animation.writers['ffmpeg']
            anim.save(str(self.dataset_name) + '_animation.gif', writer='pillow', fps=5)
        plt.xlim([0, 2])
        plt.ylim([0, 1])
        plt.title('Animation of ' +str(t_vec.size) + ' time stamps of ' + str(self.dataset_name) + ' flow')
        if show == True:
            plt.show()

    def eig_vals_plot(self, r, eps_list, fig=None, show=True):
        pts = self.pts
        markers = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
        colors = itertools.cycle(('b', 'r', 'y', 'g', 'm', 'c')) 
        if (fig == None):
            fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1, 2, 2)
        for eps in eps_list:
            Q_eigs = Q_eps(pts, r=r, eps=eps, time_slices=self.time_slices, load_cached=True, dir_name=self.dataset_name)
            Q_eigVals, Q_eigVecs = Q_eigs
            Q_eigVecs = Q_eigVecs.T
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

    def plot_eigenfuncs_at_times(self, r, eps_list, cmap=None, show=True):
        pts = self.pts
        m = pts.shape[0]
        img_shape = self.img_shape
        t_vec = self.t_vec
        def computeLargeDiffSet(eigVals, n_largest=3):
            diffs = np.abs(eigVals[:-1] - eigVals[1:])
            # foundInds = diffs.argsort()[-n_largest:][::-1]
            mean_diff = np.mean(diffs)
            foundInds = np.where(diffs > mean_diff)
            return foundInds[0]

        for eps in eps_list:
            Q_eigs = Q_eps(pts, r=r, eps=eps, time_slices=self.time_slices, load_cached=True, dir_name=self.dataset_name)
            fig, axes = plt.subplots(self.time_slices.size, 1, figsize=(12,18), constrained_layout=True)
            for idx, t_slice in enumerate(self.time_slices):
                t = t_vec[t_slice]
                Q_eigVals, Q_eigVecs = Q_eigs
                Q_eigVecs = Q_eigVecs.T
                L_eigVals = (Q_eigVals - 1)/eps
                # deltaEigs = computeLargeDiffSet(L_eigVals, n_largest=3)
                deltaEigs = self.largest_gap_eigs
                eigFunc = Q_eigVecs[1].reshape(img_shape)
                eigFunc = np.flip(eigFunc, axis=0)
                pts_t = pts[:, t_slice, :]
                # x_grid, y_grid = np.meshgrid(pts_t)
                scat = axes[idx].scatter(pts_t[:,0], pts_t[:,1], c=eigFunc, s=10, cmap=cmap)
                axes[idx].set_title('Second Eigenfunction of ' + r'$Q_{\epsilon}$' + ' for time slice t = ' +str(t))
                axes[idx].set_xlabel('x')
                axes[idx].set_ylabel('y')
                # plt.colorbar(im)
            fig.suptitle(r'$\epsilon=$' + str(eps))
        if show == True:
            plt.show()


    ##  2 - Clustering 2 

    def cluster_labels(self, r, eps_list, n_clusters=2, cmap=None, show=True):
        pts = self.pts
        m = pts.shape[0]
        img_shape = self.img_shape
        t_vec = self.t_vec
        for eps in eps_list:
            Q_eigs = Q_eps(pts, r=r, eps=eps, time_slices=self.time_slices, load_cached=True, dir_name=self.dataset_name)
            fig, axes = plt.subplots(self.time_slices.size, 1, figsize=(12,18), constrained_layout=True)
            for idx, t_slice in enumerate(self.time_slices):
                t = t_vec[t_slice]
                Q_eigVals, Q_eigVecs = Q_eigs
                Q_eigVecs = Q_eigVecs.T
                L_eigVals = (Q_eigVals - 1)/eps
                # deltaEigs = computeLargeDiffSet(L_eigVals, n_largest=1)
                deltaEigs = self.largest_gap_eigs
                label_space = cluster_eigVectors(Q_eigVecs[:n_clusters], n_clusters=n_clusters)
                label_space = np.reshape(label_space, img_shape)
                # label_space = np.flip(label_space, axis=0)
                pts_t = pts[:, t_slice, :]
                scat = axes[idx].scatter(pts_t[:,0], pts_t[:,1], c=label_space, s=10, cmap=cmap)
                axes[idx].set_title(str(n_clusters) + ' Clustering of ' + r'$Q_{\epsilon}$' + ' for time slice t = ' + str(t))
                axes[idx].set_xlabel('x')
                axes[idx].set_ylabel('y')
                # plt.colorbar(im)
            fig.suptitle(r'$\epsilon=$' + str(eps))
        if show == True:
            plt.show()


    def discard_data(self, remaining_pts_num=500):
        m = pts.shape[0]