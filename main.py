import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.sparse.linalg import eigs
from scipy import sparse
import itertools


## Generate the Double gyre flow
from generate_double_gyre import generate_trajectories

y_vec = np.arange(100)/100
x_vec = 2*np.arange(200)/200
T = 201
t_vec = 20*np.arange(T)/T
x_grid, y_grid = np.meshgrid(x_vec, y_vec)
trajectories, _, _ = generate_trajectories(x_vec, y_vec, t_vec, load_cached=True)
img_shape = x_grid.shape


## Flatten trajectories data to m tranjectories
trajectories = trajectories.reshape(-1, trajectories.shape[-2], trajectories.shape[-1])
m = trajectories.shape[0]
pts = trajectories
largest_gap_eigs = [2, 3, 8] # eigVecs indentified as before largest step

## Scatter the eigenValues of different epsilons

eps_list = [0.0002, 0.0005, 0.001, 0.002, 0.004]
# eps_list = [0.00001, 0.0002, 0.003]
r = 2

## Create class of data analysis for the double gyre dataset

from data_class_plotters import DataAnalysis
double_gyre_class = DataAnalysis(pts, 'doubleGyre', t_vec, img_shape, largest_gap_eigs, k_eigVals=15,time_samples=4)

## Animate the trajectories 

# double_gyre_class.animate_data(fig=None, cmap='turbo', show=False, save=True)

## Compute Qeps / Load from pre-computed

## Plot the largest eigenVals of the Q_eps and the Laplacian

# double_gyre_class.eig_vals_plot(r, eps_list, fig=None, show=True)

## Plot the second eigenfunction at various times

# double_gyre_class.plot_eigenfunc_at_times(r, eps_list, eig_idx=2, cmap=plt.cm.turbo, show=True)

## Cluster into 2 labels

# double_gyre_class.cluster_plot(r, eps_list, n_clusters=2, cmap='Dark2', show=True)

## Cluster into 3 labels

# double_gyre_class.cluster_plot(r, eps_list, n_clusters=3, cmap='Dark2', show=True)

## Cluster into 4 labels

# double_gyre_class.cluster_plot(r, eps_list, n_clusters=4, cmap='Dark2', show=True)

## Missing Data Section

# eps_list = [0.01]
# missing_data, discarded_pts = double_gyre_class.discard_data(remaining_pts_num=500, destroy_rate=0.8)
# missing_gyre_class = DataAnalysis(missing_data, 'missingGyre', t_vec, img_shape, largest_gap_eigs, time_samples=4)
# missing_gyre_class.eig_vals_plot(r, eps_list, fig=None, show=False)
# labeled_data = missing_gyre_class.cluster_labels(missing_data, r, eps=0.01, n_clusters=3)
# figs, axises = double_gyre_class.cluster_plot(r, [0.004], n_clusters=3, cmap='Dark2', show=False)
# fig, axes = figs[-1], axises[-1]
# axes[0].scatter(discarded_pts[:,0,0], discarded_pts[:,0,1], c=labeled_data, s=10, cmap='viridis')
# plt.show()


## Load Bickley Jet

bickley_x = np.genfromtxt('./BickleyJet/bickley_x.csv', delimiter=",")
bickley_y = np.genfromtxt('./BickleyJet/bickley_y.csv', delimiter=",")
bickley_wild_x = np.genfromtxt('./BickleyJet/bickley_wild_x.csv', delimiter=",")
bickley_wild_y = np.genfromtxt('./BickleyJet/bickley_wild_y.csv', delimiter=",")

m = bickley_x.shape[0]
T = bickley_x.shape[1]
t_vec = 40*np.arange(T)/T
pts = np.concatenate((bickley_x[:, :, np.newaxis], bickley_y[:, :, np.newaxis]), axis=2)
img_shape = (60, 200)
largest_gap_eigs = [1, 2, 8] # eigVecs indentified as before largest step

## Create class of data analysis for the bickleyjet dataset

from data_class_plotters import DataAnalysis

bickley_jet_class = DataAnalysis(pts, 'bickleyJet', t_vec, img_shape, largest_gap_eigs, k_eigVals=20, time_samples=6)

## Animate the data in time

# bickley_jet_class.animate_data(fig=None, cmap='turbo', show=False, save=True)

## Compute Qeps / Load from pre-computed

## Plot the largest eigenfunctions of the Q_eps and the Laplacian

eps_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# eps_list = [0.01, 0.02]
r = 2

bickley_jet_class.eig_vals_plot(r, eps_list, fig=None, show=True)

## Plot the eigenfunctions at various times

bickley_jet_class.plot_eigenfuncs_at_times_eps(r, eps=0.02, eig_inds=[2, 3, 4], cmap=plt.cm.turbo, show=True)

## Cluster into 9 labels

bickley_jet_class.cluster_plot(r, eps_list, n_clusters=9, cmap='Dark2', show=True )


print('ya')