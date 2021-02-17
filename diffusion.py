import numpy as np

def h(x):
    r = 1
    cr = 1/r
    return cr*np.exp(-x)*(x <= r)

def k_eps(x1, x2, eps):
    return h(np.norm(x1-x2)**2/eps)