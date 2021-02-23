import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.sparse.linalg import eigs
from scipy import sparse
import itertools
## Generate the Double gyre flow

## Define constants

alpha = 0.25
A = 0.25
omega = 2*np.pi

y_vec = np.arange(100)/100
x_vec = 2*np.arange(200)/200
T = 201
t_vec = 20*np.arange(T)/T

x_grid, y_grid = np.meshgrid(x_vec, y_vec)
grid_pts = np.vstack([x_grid.ravel(), y_grid.ravel()])

## define inner f function
f = lambda t,x : alpha*np.multiply(np.sin(omega*t),np.power(x,2))+np.multiply((1-2*alpha*np.sin(omega*t)), x)
f_x = lambda t,x : 2*alpha*np.multiply(np.sin(omega*t), x)+(1-2*alpha*np.sin(omega*t))

## define array dunction of [x y] for ode
def double_gyre_system(y, t, A, alpha, omega, f, f_x):
    y1 = -np.pi*A*np.sin(np.pi*f(t, y[0]))*np.cos(np.pi*y[1])
    y2 = np.pi*A*np.cos(np.pi*f(t, y[0]))*np.sin(np.pi*y[1])*f_x(t, y[0])
    return np.array([y1, y2])

trajectories =  np.load('data/trajectories.npy')
# trajectories = x_grid.tolist()

# for row_idx in range(x_grid.shape[0]):
#     for col_idx in range(x_grid.shape[1]):
#         y0 = np.array([x_vec[col_idx], y_vec[row_idx]])
#         sol = integrate.odeint(double_gyre_system, y0, t_vec, (A, alpha, omega, f, f_x))
#         trajectories[row_idx][col_idx] = sol


fig1, ax = plt.subplots()
lines = []
# for start_x in range(0, x_vec.size, 4):
#     for start_y in range(0, y_vec.size, 4):
#         trajectory = trajectories[:][start_y,start_x]
#         lines.append(ax.scatter(trajectory[0], trajectory[1]))
c = x_grid
scat = ax.scatter(x_grid, y_grid, c=c, s=10, cmap='turbo')

# plt.show()

def connect(i):
    # x_new = trajectories[:,:,i,0]
    # y_new = trajectories[:,:,i,1]
    scat.set_offsets(trajectories[:,:,i,:].reshape(-1,2))
    # scat.set_array(color_func-i/100)
    # scat.set_color(color_func.reshape(-1,4)-(i/500)*np.ones((1,4)))
    return scat, 

     
anim = animation.FuncAnimation(fig1, connect, range(t_vec.size), interval=50)
# anim.save('double_gyre_trajectories.mp4')
plt.xlim([0, 2])
plt.ylim([0, 1])
plt.title('Samples of ' + str(len(lines)) + ' Trajectories computed by the solving the ODE')
# plt.show()

## Compute Qeps / Load from pre-computed

from diffusion import Q_eps, computeQ_eigVals, cluster_eigVectors

## Flatten trajectories data to m tranjectories
trajectories = trajectories.reshape(-1, trajectories.shape[-2], trajectories.shape[-1])
m = trajectories.shape[0]

## Scatter the eigenValues of different epsilons

eps_list = [0.0002, 0.0005, 0.001, 0.002]
r = 2

# markers = itertools.cycle((',', '+', '.', 'o', '*')) 
# colors = itertools.cycle(('b', 'r', 'y', 'g', 'm')) 

# fig2 = plt.figure(2, figsize=(14,6))
# ax1 = fig2.add_subplot(1,2,1)
# ax2 = fig2.add_subplot(1, 2, 2)

# for eps in eps_list:
# 	Qeps_list = Q_eps(trajectories, r=r, eps=eps, load_cached=True)
# 	Qeps = Qeps_list[-1]
# 	Q_eigVals, Q_eigVecs = computeQ_eigVals(Qeps, r=r, eps=eps, k=10, load_cached=True)
# 	L_eigVals = (Q_eigVals - 1)/eps
# 	color = next(colors)
# 	marker=next(markers)
# 	ax1.scatter(range(Q_eigVals.size), Q_eigVals, s=20, c=color, marker=marker, label=r'$\epsilon=$'+str(eps))
# 	ax2.scatter(range(L_eigVals.size), L_eigVals, s=20, c=color, marker=marker, label=r'$\epsilon=$'+str(eps))      
# 	ax1.legend(loc='upper right')
# 	ax2.legend(loc='upper right')

# ax1.set_xlabel('n')
# ax1.set_ylabel(r'$\lambda_n$')
# ax1.set_title('Scatter eigenValues of ' + r'$Q_{\epsilon}$' + ' for various ' + r'$\epsilon$' + ' values')
# ax2.set_xlabel('n')
# ax2.set_ylabel(r'$\lambda_n$')
# ax2.set_title('Scatter eigenValues of ' + r'$L_{\epsilon}$' + ' for various ' + r'$\epsilon$' + ' values')
# plt.show()

## 3 - Clustering at t = 0 and t = 19.5

def computeLargeDiffSet(eigVals, n_largest=3):
	diffs = np.abs(eigVals[:-1] - eigVals[1:])
	# foundInds = diffs.argsort()[-n_largest:][::-1]
	mean_diff = np.mean(diffs)
	foundInds = np.where(diffs > mean_diff)
	return foundInds[0]

eps_list = [0.0002]
# eps_list = [0.0002, 0.0005, 0.001, 0.002]
# figs = range(eps_list)
for eps in eps_list:
	Qeps_list = Q_eps(trajectories, r=r, eps=eps, load_cached=True)
	Qeps_list = Qeps_list[::int(len(Qeps_list)/2)]
	time_slices = t_vec[-1]*range(len(Qeps_list))/len(Qeps_list)
	fig, axes = plt.subplots(len(Qeps_list), 1, figsize=(12,18), constrained_layout=True)
	for idx, Qeps in enumerate(Qeps_list):
		t = time_slices[idx]
		Q_eigVals, Q_eigVecs = computeQ_eigVals(Qeps, r=r, eps=eps, k=15, load_cached=True)
		L_eigVals = (Q_eigVals - 1)/eps
		deltaEigs = computeLargeDiffSet(L_eigVals, n_largest=3)
		cluster_eigVectors(Q_eigVecs[deltaEigs], grid_pts, n_clusters=2)
		eigFunc = Q_eigVecs[deltaEigs[0]].reshape(x_grid.shape)
		eigFunc = np.flip(eigFunc, axis=0)
		im = axes[idx].imshow(np.real(eigFunc), cmap=plt.cm.YlGn)
		axes[idx].set_title('Second Eigenfunction of ' + r'$Q_{\epsilon}$' + ' for time slice t = ' +str(t))
		axes[idx].set_xlabel('x')
		axes[idx].set_ylabel('y')
		# plt.colorbar(im)
	fig.suptitle(r'$\epsilon=$' + str(eps))
plt.show()


print('ya')