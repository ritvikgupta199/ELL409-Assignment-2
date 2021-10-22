import numpy as np

def get_linear_x(x1, x2):
        return np.matmul(x1, x2.T)
        
def get_gaussian_x(x1, x2, gamma):
    s1 = np.sum(x1**2, axis=1)
    s2 = np.sum(x2**2, axis=1)
    p = s1 + s2.T - 2*np.matmul(x1, x2.T)
    return np.exp(-gamma * p)

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.square(x1-x2).sum())

def linear_kernel(x1, x2):
    return np.matmul(x1, x2.T)