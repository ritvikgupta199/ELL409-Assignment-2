import numpy as np
import os

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

def create_test_outputs(preds, file_path):
    if os.path.isfile(file_path):
        f = open(file_path, 'w')
    else:
        f = open(file_path, 'x')
    f.write('Id,Class\n')
    for (i, pred_cls) in enumerate(preds):
        if i < 999:
            f.write(f'{i+1},{int(pred_cls)}\n')
        else:
            f.write(f'\"{i+1:,}\",{int(pred_cls)}\n')
    f.close()
    print(f'Test outputs written in {file_path}')