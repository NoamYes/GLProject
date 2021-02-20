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

# x, y, t = sp.symbols("x y t")
# f_exp = alpha*sp.sin(omega*t)*x**2+(1-2*alpha*sp.sin(omega*t))*x
# f_x_exp = 2*alpha*sp.sin(omega*t)*x+(1-2*alpha*sp.sin(omega*t))
# f_exp = sin(t)+x
# f = lambdify([t,x], f_exp, modules=['numpy', 'math'])
# f_x = lambdify([t,x], f_x_exp, modules=['numpy', 'math'])
# f = theano_function([t,x], [f_exp], dims={t: 3, x: 3})

y_vec = np.arange(100)/100
x_vec = 2*np.arange(200)/200
T = 201
t_vec = 20*np.arange(T)/T

x_grid, y_grid = np.meshgrid(x_vec, y_vec)

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
for start_x in range(0, x_vec.size, 10):
    for start_y in range(0, y_vec.size, 10):
        trajectory = trajectories[:][start_y,start_x]
        lines.append(ax.plot(trajectory[0], trajectory[1], "ro-"))

def connect(i):
    traj = 0
    for start_x in range(0, x_vec.size, 10):
        for start_y in range(0, y_vec.size, 10):
            trajectory = trajectories[:][start_y,start_x]
            start=max((i-5,0))
            lines[traj][0].set_data(trajectory[start:i,0],trajectory[start:i,1])
            traj = traj + 1
    return lines

     
anim = animation.FuncAnimation(fig1, connect, np.arange(1, t_vec.size), interval=5)
plt.xlim([0, 2])
plt.ylim([0, 1])
# plt.show()

## Compute Qeps / Load from pre-computed

from diffusion import Q_eps, computeQ_eigVals

## Flatten trajectories data to m tranjectories
trajectories = trajectories.reshape(-1, trajectories.shape[-2], trajectories.shape[-1])
m = trajectories.shape[0]

## Scatter the eigenValues of different epsilons

fig2 = plt.figure(2, figsize=(14,6))
ax1 = fig2.add_subplot(1,2,1)
ax2 = fig2.add_subplot(1, 2, 2)

eps_list = [0.0002, 0.0005, 0.001, 0.002]
markers = itertools.cycle((',', '+', '.', 'o', '*')) 
colors = itertools.cycle(('b', 'r', 'y', 'g', 'm')) 
r = 1

for eps in eps_list:
	Qeps = Q_eps(trajectories, r=r, eps=eps, load_cached=True)
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
print('ya')