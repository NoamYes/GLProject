import numpy as np
import sympy as sp
from sympy.printing.theanocode import theano_function

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

x_grid, t_grid, y_grid = np.meshgrid(t_vec, x_vec, y_vec)

## compute velocity vectors
f = lambda t,x : alpha*np.multiply(np.sin(omega*t),np.power(x,2))+np.multiply((1-2*alpha*np.sin(omega*t)), x)
f_x = lambda t,x : 2*alpha*np.multiply(np.sin(omega*t), x)+(1-2*alpha*np.sin(omega*t))

f_res = f(t_grid, x_grid)

vx = -np.pi*A*np.sin(np.pi*f(t_grid, x_grid))*np.cos(np.pi*y_grid)
vy = np.pi*A*np.cos(np.pi*f(t_grid, x_grid))*np.sin(np.pi*y_grid)*f_x(t_grid, x_grid)

print('ya')