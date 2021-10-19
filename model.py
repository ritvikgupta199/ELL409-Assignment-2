import time
import numpy as np
import libsvm.svmutil as svm

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