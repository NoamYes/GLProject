import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.sparse.linalg import eigs
from scipy import sparse
import itertools

from diffusion import Q_eps, computeQ_eigVals, cluster_eigVectors

## Flatten trajectories data to m tranjectories
trajectories = trajectories.reshape(-1, trajectories.shape[-2], trajectories.shape[-1])
m = trajectories.shape[0]

## Scatter the eigenValues of different epsilons

eps_list = [0.0002, 0.0005, 0.001, 0.002]
r = 2

markers = itertools.cycle((',', '+', '.', 'o', '*')) 
colors = itertools.cycle(('b', 'r', 'y', 'g', 'm')) 

fig2 = plt.figure(2, figsize=(14,6))
ax1 = fig2.add_subplot(1,2,1)
ax2 = fig2.add_subplot(1, 2, 2)

for eps in eps_list:
	Qeps_list = Q_eps(trajectories, r=r, eps=eps, load_cached=True)
	Qeps = Qeps_list[-1]
	Q_eigVals, Q_eigVecs = computeQ_eigVals(Qeps, r=r, eps=eps, k=10, load_cached=True)
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
plt.show()

## 3 - Eigenfunction at t = 0 and t = 19.5

def computeLargeDiffSet(eigVals, n_largest=3):
	diffs = np.abs(eigVals[:-1] - eigVals[1:])
	# foundInds = diffs.argsort()[-n_largest:][::-1]
	mean_diff = np.mean(diffs)
	foundInds = np.where(diffs > mean_diff)
	return foundInds[0]

eps_list = [0.0002]
# eps_list = [0.0002, 0.0005, 0.001, 0.002]
for eps in eps_list:
	Qeps_list = Q_eps(trajectories, r=r, eps=eps, load_cached=True)
	Qeps_list = Qeps_list[::int(len(Qeps_list)/2)]
	time_slices = t_vec[-1]*range(len(Qeps_list))/len(Qeps_list)
	fig, axes = plt.subplots(len(Qeps_list), 1, figsize=(12,18), constrained_layout=True)
	for idx, Qeps in enumerate(Qeps_list):
		t = time_slices[idx]
		Q_eigVals, Q_eigVecs = computeQ_eigVals(Qeps, r=r, eps=eps, k=15, load_cached=True)
		L_eigVals = (Q_eigVals - 1)/eps
		# deltaEigs = computeLargeDiffSet(L_eigVals, n_largest=3)
		deltaEigs = [2, 3, 8]
		eigFunc = Q_eigVecs[deltaEigs[0]].reshape(x_grid.shape)
		eigFunc = np.flip(eigFunc, axis=0)
		im = axes[idx].imshow(np.real(eigFunc), cmap=plt.cm.PuBu)
		axes[idx].set_title('Second Eigenfunction of ' + r'$Q_{\epsilon}$' + ' for time slice t = ' +str(t))
		axes[idx].set_xlabel('x')
		axes[idx].set_ylabel('y')
		# plt.colorbar(im)
	fig.suptitle(r'$\epsilon=$' + str(eps))
plt.show()

##  2 - Clustering 2 

eps_list = [0.0002]
# eps_list = [0.0002, 0.0005, 0.001, 0.002]
for eps in eps_list:
	Qeps_list = Q_eps(trajectories, r=r, eps=eps, load_cached=True)
	Qeps_list = Qeps_list[::int(len(Qeps_list)/2)]
	time_slices = t_vec[-1]*range(len(Qeps_list))/len(Qeps_list)
	fig, axes = plt.subplots(len(Qeps_list), 1, figsize=(12,18), constrained_layout=True)
	for idx, Qeps in enumerate(Qeps_list):
		t = time_slices[idx]
		Q_eigVals, Q_eigVecs = computeQ_eigVals(Qeps, r=r, eps=eps, k=15, load_cached=True)
		L_eigVals = (Q_eigVals - 1)/eps
		# deltaEigs = computeLargeDiffSet(L_eigVals, n_largest=1)
		deltaEigs = [3, 4, 9]
		label_space = cluster_eigVectors(Q_eigVecs[deltaEigs], n_clusters=4)
		label_space = np.reshape(label_space, x_grid.shape)
		label_space = np.flip(label_space, axis=0)
		im = axes[idx].imshow(np.real(label_space), cmap=plt.cm.YlGn)
		axes[idx].set_title('2 Clustering of ' + r'$Q_{\epsilon}$' + ' for time slice t = ' +str(t))
		axes[idx].set_xlabel('x')
		axes[idx].set_ylabel('y')
		# plt.colorbar(im)
	fig.suptitle(r'$\epsilon=$' + str(eps))
plt.show()

print('ya')