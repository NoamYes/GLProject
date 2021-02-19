import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation, rc

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

x_vec = np.arange(100)/100
y_vec = 2*np.arange(200)/200
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


fig, ax = plt.subplots()
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

     
anim = animation.FuncAnimation(fig, connect, np.arange(1, t_vec.size), interval=5)

# plt.show()

## Compute Qeps / Load from pre-computed

from diffusion import Q_eps, assemble_sim_matrix

## Flatten trajectories data to m tranjectories
trajectories = trajectories.reshape(-1, trajectories.shape[-2], trajectories.shape[-1])
m = trajectories.shape[0]
# Qeps = Q_eps(trajectories, r=1, eps=1)
Qeps =  np.load('data/Q_eps.npy', allow_pickle=True)[()]

print('ya')