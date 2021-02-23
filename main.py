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
for start_x in range(0, x_vec.size, 4):
    for start_y in range(0, y_vec.size, 4):
        trajectory = trajectories[:][start_y,start_x]
        lines.append(ax.scatter(trajectory[0], trajectory[1]))
c = x_grid
scat = ax.scatter(x_grid, y_grid, c=c, s=10, cmap='turbo')


def connect(i):
    scat.set_offsets(trajectories[:,:,i,:].reshape(-1,2))
    return scat, 

     
anim = animation.FuncAnimation(fig1, connect, range(t_vec.size), interval=50)
anim.save('double_gyre_trajectories.mp4')
plt.xlim([0, 2])
plt.ylim([0, 1])
plt.title('Samples of ' + str(len(lines)) + ' Trajectories computed by the solving the ODE')
# plt.show()

## Compute Qeps / Load from pre-computed

