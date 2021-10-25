import time
import numpy as np
import libsvm.svmutil as svm
import cvxopt as co
import utils
import random

EPS = 1e-10
TOL = 1e-5


class LIBSVMModel:
    def __init__(self, c, kernel, gamma):
        self.c = c
        if kernel == 'linear':
            self.kernel = svm.LINEAR
        else:
            self.kernel = svm.RBF
        self.gamma = gamma
        self.model = None

    def train(self, train_x, train_y, quiet):
        if not quiet:
            print('Training SVM')
        t = time.time()
        params = f'-t {self.kernel} -c {self.c} -q'
        if self.gamma != None:
            params += f' -g {self.gamma}'
        self.model = svm.svm_train(train_y, train_x, params)
        if not quiet:
            print(f'Time Taken: {time.time()-t}s')

    def get_train_preds(self, x_data, y_data):
        pred_labels, pred_acc, pred_val = svm.svm_predict(
            y_data, x_data, self.model, options='-q')
        return pred_labels, pred_acc[0]

    def get_test_preds(self, x_data):
        y_data = np.ones(len(x_data))
        pred_labels, pred_acc, pred_val = svm.svm_predict(
            y_data, x_data, self.model, options='-q')
        pred_labels = np.array(pred_labels)
        return pred_labels


class CVXModel:
    def __init__(self, c, kernel, gamma, method):
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.method = method
        self.alphas, self.x_sv, self.y_sv = None, None, None

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

    def solve_opt(self, train_x, train_y, quiet):
        if self.method == 'cvxopt':
            P, q, G, h, A, b = self.get_matrices(train_x, train_y)
            sol = co.solvers.qp(P, q, G, h, A, b, options={
                                'show_progress': not quiet})
            alphas = np.array(sol['x'])
        else:
            solver = SMOSolver(self.c, self.kernel, self.gamma)
            alphas = solver.solve(train_x, train_y, 10)
        return alphas

    def get_sv_b(self, train_x, train_y):
        sv_zero = np.where(self.alphas > EPS)
        sv_c = np.where(self.alphas <= self.c-EPS)
        sv_indices = np.intersect1d(sv_zero, sv_c)
        x_sv, y_sv = train_x[sv_indices], train_y[sv_indices]
        self.alphas = self.alphas[sv_indices]
        vals = (self.alphas * y_sv * self.get_xmat(x_sv, x_sv)).sum(0)
        pos_indices = np.where(y_sv == 1)[0]
        neg_indices = np.where(y_sv == -1)[0]
        M = max([vals[i] for i in neg_indices])
        m = min([vals[i] for i in pos_indices])
        b = -(M+m)/2
        return x_sv, y_sv, b

    def train(self, train_x, train_y, quiet):
        if not quiet:
            print('Training SVM')
        t = time.time()
        self.alphas = self.solve_opt(train_x, train_y, quiet)
        self.x_sv, self.y_sv, self.b = self.get_sv_b(train_x, train_y)
        if not quiet:
            print(f'Time Taken: {time.time()-t}s')

    def get_train_preds(self, x_data, y_data):
        preds = (self.alphas.reshape(-1, 1) * self.y_sv.reshape(-1, 1) *
                 self.get_xmat(self.x_sv, x_data)).sum(0) + self.b
        pred_labels = np.where(preds >= 0, 1, -1)
        acc = 100 * (y_data == pred_labels).sum() / y_data.shape[0]
        return pred_labels, acc


class SMOSolver:
    def __init__(self, c, kernel, gamma):
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.alphas, self.b = None, None

    def reset(self, n):
        self.alphas, self.b = np.zeros(n), 0.0

    def f_x(self, x, train_x, train_y):
        x = x.reshape(1, -1)
        if self.kernel == 'linear':
            return ((self.alphas*train_y).reshape(-1, 1)*utils.get_linear_x(train_x, x)).sum(0)+self.b
        else:
            return ((self.alphas*train_y).reshape(-1, 1)*utils.get_gaussian_x(train_x, x, self.gamma)).sum(0)+self.b

    def get_kernel(self, x1, x2):
        if self.kernel == 'linear':
            return utils.linear_kernel(x1, x2)
        else:
            return utils.gaussian_kernel(x1, x2, self.gamma)

    def get_lh(self, yi, yj, i, j):
        if yj == yi:
            L = max(0, self.alphas[i]+self.alphas[j]-self.c)
            H = min(self.c, self.alphas[i]+self.alphas[j])
        else:
            L = max(0, self.alphas[j]-self.alphas[i])
            H = min(self.c, self.c+self.alphas[j]-self.alphas[i])
        return L, H

    def get_b(self, ei, ej, xi, xj, yi, yj, alpha_io, alpha_jo, i, j):
        b1 = self.b - ei - yi*(self.alphas[i]-alpha_io)*self.get_kernel(
            xi, xi) - yj*(self.alphas[j]-alpha_jo)*self.get_kernel(xi, xj)
        b2 = self.b - ej - yi*(self.alphas[i]-alpha_io)*self.get_kernel(
            xi, xj) - yj*(self.alphas[j]-alpha_jo)*self.get_kernel(xj, xj)
        if self.alphas[i] > 0 and self.alphas[i] < self.c:
            return b1
        elif self.alphas[j] > 0 and self.alphas[j] < self.c:
            return b2
        else:
            return (b1+b2)/2

    def get_eta(self, xi, xj):
        return 2*self.get_kernel(xi, xj)-self.get_kernel(xi, xi)-self.get_kernel(xj, xj)

    def solve(self, train_x, train_y, max_passes):
        n = train_x.shape[0]
        self.reset(n)
        passes = 0
        while passes < max_passes:
            n_ch_alphas = 0
            for i in range(n):
                xi, yi = train_x[i], train_y[i]
                ei = self.f_x(xi, train_x, train_y) - yi
                if (yi*ei < -TOL and self.alphas[i] < self.c) or (yi*ei > TOL and self.alphas[i] > 0):
                    j = random.choice(list(range(0, i)) + list(range(i+1, n)))
                    xj, yj = train_x[j], train_y[j]
                    ej = self.f_x(xj, train_x, train_y) - yj
                    alpha_io, alpha_jo = self.alphas[i], self.alphas[j]
                    L, H = self.get_lh(yi, yj, i, j)
                    if L == H:
                        continue
                    eta = self.get_eta(xi, xj)
                    if eta >= 0:
                        continue
                    self.alphas[j] -= (yj*(ei-ej)) / eta
                    self.alphas[j] = min(H, max(self.alphas[j], L))
                    if abs(self.alphas[j]-alpha_jo) < TOL:
                        continue
                    self.alphas[i] += yi*yj*(alpha_jo-self.alphas[j])
                    self.b = self.get_b(ei, ej, xi, xj, yi,
                                        yj, alpha_io, alpha_jo, i, j)
                    n_ch_alphas += 1
            if n_ch_alphas == 0:
                passes += 1
            else:
                passes = 0
        return self.alphas
