import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.sparse.linalg import eigs
from scipy import sparse
import itertools

## Generate the Double gyre flow
from generate_double_gyre import generate_trajectories, animate_double_gyre_flow

y_vec = np.arange(100)/100
x_vec = 2*np.arange(200)/200
T = 201
t_vec = 20*np.arange(T)/T
x_grid, y_grid = np.meshgrid(x_vec, y_vec)
trajectories, _, _ = generate_trajectories(x_vec, y_vec, t_vec, load_cached=True)
img_shape = x_grid.shape

## Animate the trajectories 

# animate_double_gyre_flow(x_vec, y_vec, t_vec, fig=None, show=False, save=False)

## Flatten trajectories data to m tranjectories
trajectories = trajectories.reshape(-1, trajectories.shape[-2], trajectories.shape[-1])
m = trajectories.shape[0]
pts = trajectories

## Scatter the eigenValues of different epsilons

eps_list = [0.0002, 0.0005, 0.001, 0.002]
r = 2

## Compute Qeps / Load from pre-computed

from plotters import eig_vals_plot, plot_eigenfuncs_at_times, cluster_2_labels

## Plot the largest eigenfunctions of the Q_eps and the Laplacian

# eig_vals_plot(pts, r, eps_list, fig=None, show=True)

## Plot the eigenfunctions at various times

plot_eigenfuncs_at_times(pts, img_shape, t_vec, r, eps_list, cmap=plt.cm.PuBu, show=True)

## Cluster into 2 labels

eps_list = [0.0002]
# eps_list = [0.0002, 0.0005, 0.001, 0.002]

cluster_2_labels(pts, img_shape, t_vec, r, eps_list, cmap=plt.cm.PuBu, show=True)