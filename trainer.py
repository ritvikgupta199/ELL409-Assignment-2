import numpy as np
from dataloader import DataLoader, TestDataLoader
from model import LIBSVMModel, CVXModel

def train_svm(data_path, split, n_feat, is_multi, cls, model_type, c, kernel, gamma):
    dataloader = DataLoader(data_path)
    if is_multi:
        train_x, train_y, valid_x, valid_y = dataloader.get_data(split, n_feat)
    else:
        train_x, train_y, valid_x, valid_y = dataloader.get_binary_data(split, cls, n_feat)

    if model_type == 'libsvm':
        model = LIBSVMModel(c, kernel, gamma)
    else:
        print('Model not implemented')
    model.train(train_x, train_y)

    train_preds, train_acc = model.get_train_preds(train_x, train_y)
    print(f'Training Accuracy: {train_acc}')
    if split < 1:
        valid_preds, valid_acc = model.get_train_preds(valid_x, valid_y)
        print(f'Validation Accuracy: {valid_acc}')

def train_test_svm(train_path, test_path, n_feat, c, kernel, gamma):
    train_loader = DataLoader(train_path)
    test_loader = TestDataLoader(test_path)
    train_x, train_y, _, _ = train_loader.get_data(1, n_feat)
    test_x = test_loader.get_data(n_feat)

    model = LIBSVMModel(c, kernel, gamma)
    model.train(train_x, train_y)

    train_preds, train_acc = model.get_train_preds(train_x, train_y)
    print(f'Training Accuracy: {train_acc}')
    test_preds = model.get_test_preds(test_x)
    return test_preds

# train_svm('data/2019MT10512.csv', 0.7, 10, True, None, 'libsvm', 5, 'gaussian', None)