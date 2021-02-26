import numpy as np
from scipy import integrate
import os.path
import matplotlib.pyplot as plt
from matplotlib import animation, rc

## Generate the Double gyre flow

## Define constants

def generate_trajectories(x_vec, y_vec, t_vec, load_cached=True):
    alpha = 0.25
    A = 0.25
    omega = 2*np.pi

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

    if load_cached == True and os.path.isfile('./data/trajectories.npy'):
        trajectories =  np.load('data/trajectories.npy')
    else:
        trajectories = x_grid.tolist()
        for row_idx in range(x_grid.shape[0]):
            for col_idx in range(x_grid.shape[1]):
                y0 = np.array([x_vec[col_idx], y_vec[row_idx]])
                sol = integrate.odeint(double_gyre_system, y0, t_vec, (A, alpha, omega, f, f_x))
                trajectories[row_idx][col_idx] = sol
        np.save('./data/trajectories.npy', trajectories)
    return trajectories, x_grid, y_grid
