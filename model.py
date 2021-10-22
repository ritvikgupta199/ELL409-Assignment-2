import time
import numpy as np
import libsvm.svmutil as svm
import cvxopt as co
import utils
import random


class LIBSVMModel:
    def __init__(self, c, kernel, gamma):
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.model = None

    def train(self, train_x, train_y):
        print(f'Training SVM')
        t = time.time()
        params = f'-t {self.kernel} -c {self.c} -q'
        if self.gamma != None:
            params += f' -g {self.gamma}'
        self.model = svm.svm_train(train_y, train_x, params)
        print(f'Time Taken: {time.time()-t}s')

    def get_train_preds(self, x_data, y_data):
        pred_labels, pred_acc, pred_val = svm.svm_predict(
            y_data, x_data, self.model, options=f'-q')
        return pred_labels, pred_acc[0]

    def get_test_preds(self, x_data):
        y_data = np.ones(len(x_data))
        pred_labels, pred_acc, pred_val = svm.svm_predict(
            y_data, x_data, self.model, options=f'-q')
        pred_labels = np.array(pred_labels)
        return pred_labels


class CVXModel:
    def __init__(self, c, kernel, gamma):
        self.c = c
        self.kernel = kernel
        self.gamma = gamma

    def get_xmat(self, x1, x2):
        if self.kernel == 'linear':
            return utils.get_linear_x(x1, x2)
        else:
            return utils.get_gaussian_x(x1, x2, self.gamma)

    def get_matrices(self, train_x, train_y):
        n = train_y.shape[0]
        P = np.outer(train_y, train_y) * self.get_xmat(train_x, train_x)
        q = -1.0 * np.ones(train_y.shape)
        G = np.concatenate((-1.0*np.eye(n), np.eye(n)))
        h = np.concatenate(
            (np.zeros(train_y.shape), self.c*np.ones(train_y.shape)))
        A = train_y.reshape(1, n)
        b = 0.0
        P = co.matrix(P)
        q = co.matrix(q)
        G = co.matrix(G)
        h = co.matrix(h)
        A = co.matrix(A)
        b = co.matrix(b)
        return P, q, G, h, A, b

    def solve_opt(self, train_x, train_y):
        P, q, G, h, A, b = self.get_matrices(train_x, train_y)
        sol = co.solvers.qp(P, q, G, h, A, b, options={'show_progress': True})
        alphas = np.array(sol['x'])
        return alphas


class SMOSolver:
    def __init__(self, c, kernel, gamma):
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.alphas, self.b = None, None

    def reset(self, n):
        self.alphas, self.b = np.zeros(n), 0.0

    def f_x(self, x, train_x, train_y):
        if self.kernel == 'linear':
            return self.alphas*train_y*np.matmul(train_x, x.T).sum()+self.b
        else:
            return self.alphas*train_y*np.exp(-self.gamma*np.square(train_x-x).sum(1))

    def solve(self, train_x, train_y, max_passes, tol):
        n = train_x.shape[0]
        self.reset(n)
        passes = 0
        while passes < max_passes:
            n_ch_alphas = 0
            for i in range(n):
                ei = self.f_x(train_x[i], train_x, train_y) - train_y[i]
                if (train_y[i]*ei < -tol and self.alphas[i] < self.c) or (train_y[i]*ei > tol and self.alphas[i] > 0):
                    j = random.choice(range(0, i) + range(i+1, n))
                    ej = self.f_x(train_x[j], train_x, train_y) - train_y[j]
                    alpha_io, alpha_jo = self.alphas[i], self.alphas[j]
                    if train_y[j] == train_y[i]:
                        L = max(0, self.alphas[i]+self.alphas[j]-self.c)
                        H = min(self.c, self.alphas[i]+self.alphas[j])
                    else:
                        L = max(0, self.alphas[j]-self.alphas[i])
                        H = min(self.c, self.c+self.alphas[j]-self.alphas[i])
                    if L == H:
                        continue
                    neta = 2*self.kernel(train_x[i], train_x[j])-self.kernel(
                        train_x[i], train_x[i])-self.kernel(train_x[j], train_x[j])
                    if neta >= 0:
                        continue
                    self.alphas[j] -= (train_y[i]*(ei-ej))/neta
                    self.alphas[j] = min(H, max(self.alphas[j], L))
                    if abs(self.alphas[j]-alpha_jo) < 1e-5:
                        continue
                    self.alphas += train_y[i] * \
                        train_y[j]*(alpha_jo-self.alphas[j])
                    b1 = self.b - ei - train_y[i]*(self.alphas[i]-alpha_io)*self.kernel(
                        train_x[i], train_x[i]) - train_y[j]*(self.alphas[j]-alpha_jo)*self.kernel(train_x[i], train_x[j])
                    b2 = self.b - ej - train_y[i]*(self.alphas[i]-alpha_io)*self.kernel(
                        train_x[i], train_x[j]) - train_y[j]*(self.alphas[j]-alpha_jo)*self.kernel(train_x[j], train_x[j])
                    if self.alphas[i] > 0 and self.alphas[i] < self.c:
                        self.b = b1
                    elif self.alphas[j] > 0 and self.alphas[j] < self.c:
                        self.b = b2
                    else:
                        self.b = (b1+b2)/2
                    n_ch_alphas += 1
            if n_ch_alphas == 0:
                passes += 1
            else:
                passes = 0
