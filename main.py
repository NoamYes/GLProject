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
largest_gap_eigs = [2, 3, 8] # eigVecs indentified as before largest step

## Scatter the eigenValues of different epsilons

eps_list = [0.0002, 0.0005, 0.001, 0.002]
r = 2

## Create class of data analysis for the double gyre dataset

from data_class_plotters import DataAnalysis
double_gyre_class = DataAnalysis(pts, 'doubleGyre', t_vec, img_shape, largest_gap_eigs)

## Compute Qeps / Load from pre-computed

## Plot the largest eigenfunctions of the Q_eps and the Laplacian

# double_gyre_class.eig_vals_plot(r, eps_list, fig=None, show=False)

## Plot the eigenfunctions at various times

# double_gyre_class.plot_eigenfuncs_at_times(r, eps_list, cmap=plt.cm.PuBu, show=False)

## Cluster into 2 labels

eps_list = [0.0002]
# eps_list = [0.0002, 0.0005, 0.001, 0.002]

# double_gyre_class.cluster_labels(r, eps_list, n_clusters=2, cmap=plt.cm.PuBu, show=False)

## Load Bickley Jet

bickley_x = np.genfromtxt('./BickleyJet/bickley_x.csv', delimiter=",")
bickley_y = np.genfromtxt('./BickleyJet/bickley_y.csv', delimiter=",")
bickley_wild_x = np.genfromtxt('./BickleyJet/bickley_wild_x.csv', delimiter=",")
bickley_wild_y = np.genfromtxt('./BickleyJet/bickley_wild_y.csv', delimiter=",")

m = bickley_x.shape[0]
T = bickley_x.shape[1]
t_vec = 40*np.arange(T)/T
pts = np.array([bickley_x, bickley_y]).reshape(m, T, 2)
img_shape = (60, 200)
largest_gap_eigs = [1, 2, 8] # eigVecs indentified as before largest step

## Create class of data analysis for the double gyre dataset

from data_class_plotters import DataAnalysis

bickley_jet_class = DataAnalysis(pts, 'bickleyJet', t_vec, img_shape, largest_gap_eigs)

## Compute Qeps / Load from pre-computed

## Plot the largest eigenfunctions of the Q_eps and the Laplacian

eps_list = [0.1, 0.2]
r = 2

# bickley_jet_class.eig_vals_plot(r, eps_list, fig=None, show=True)

## Plot the eigenfunctions at various times

bickley_jet_class.plot_eigenfuncs_at_times(r, eps_list, cmap=plt.cm.PuBu, show=True)

## Cluster into 2 labels

bickley_jet_class.cluster_labels(r, eps_list, n_clusters=9, cmap=plt.cm.PuBu, show=True)


print('ya')