import time
import numpy as np
import libsvm.svmutil as svm
import cvxopt as co

class LIBSVMModel:
    def __init__(self, c, kernel):
        self.c = c
        self.kernel = kernel
        self.model = None

    def train(self, train_x, train_y):
        print(f'Training SVM')
        t = time.time()
        self.model = svm.svm_train(train_y, train_x, f'-c {self.c}')
        print(f'Time Taken: {time.time()-t}s')

    def get_train_preds(self, x_data, y_data):
        pred_labels, pred_acc, pred_val = svm.svm_predict(y_data, x_data, self.model, options = f'-q')
        return pred_labels, pred_acc[0]

    def get_test_preds(self, x_data):
        y_data = np.ones(len(x_data))
        pred_labels, pred_acc, pred_val = svm.svm_predict(y_data, x_data, self.model, options = f'-q')
        pred_labels = np.array(pred_labels)
        return pred_labels

class CVXModel:
    def __init__(self, c, kernel):
        self.c = c
        self.kernel = kernel
    
    def get_linear_x(self, x1, x2):
        return np.matmul(x1, x2.T)
        
    def get_gaussian_x(self, x1, x2):
        return np.matmul(x1, x2.T)

    def get_xmat(self, x1, x2):
        if self.kernel == 'linear':
            return self.get_linear_x(x1, x2)
        else:
            return self.get_gaussian_x(x1, x2)

    def get_matrices(self, train_x, train_y):
        n = train_y.shape[0]
        P = np.outer(train_y, train_y) * self.get_xmat(train_x, train_x)
        q = -1.0 * np.ones(train_y.shape)
        G = np.concatenate((-1.0*np.eye(n), np.eye(n)))
        h = np.concatenate((np.zeros(train_y.shape), self.c*np.ones(train_y.shape)))
        A = train_y.reshape(1,n)
        b = 0.0
        P = co.matrix(P)
        q = co.matrix(q)
        G = co.matrix(G)
        h = co.matrix(h)
        A = co.matrix(A)
        b = co.matrix(b)
        return P, q, G, h, A, b

    def train(self, train_x, train_y):
        P, q, G, h, A, b = self.get_matrices(train_x, train_y)
        sol = co.solvers.qp(P, q, G, h, A, b, options={'show_progress': True})
        alphas = np.array(sol['x'])
        return alphas