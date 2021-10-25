import numpy as np
from dataloader import DataLoader, TestDataLoader
from model import LIBSVMModel, CVXModel

def train_svm(data_path, split, n_feat, is_multi, cls, model_type, c, kernel, gamma, quiet):
    dataloader = DataLoader(data_path)
    if is_multi:
        train_x, train_y, valid_x, valid_y = dataloader.get_data(split, n_feat)
    else:
        train_x, train_y, valid_x, valid_y = dataloader.get_binary_data(split, cls, n_feat)

    if model_type == 'libsvm':
        model = LIBSVMModel(c, kernel, gamma)
    elif model_type == 'cvxopt' and not is_multi:
        model = CVXModel(c, kernel, gamma, 'cvxopt')
    elif model_type == 'smo' and not is_multi:
        model = CVXModel(c, kernel, gamma, 'smo')
    else:
        print('Model not implemented')
        return
    model.train(train_x, train_y, quiet)

    train_preds, train_acc = model.get_train_preds(train_x, train_y)
    if not quiet:
        print(f'Training Accuracy: {train_acc}')
    if split < 1:
        valid_preds, valid_acc = model.get_train_preds(valid_x, valid_y)
        if not quiet:
            print(f'Validation Accuracy: {valid_acc}')
        return train_acc, valid_acc
    return train_acc, 0.0

def train_test_svm(train_path, test_path, n_feat, c, kernel, gamma):
    train_loader = DataLoader(train_path)
    test_loader = TestDataLoader(test_path)
    train_x, train_y, _, _ = train_loader.get_data(1, n_feat)
    test_x = test_loader.get_data(n_feat)

    model = LIBSVMModel(c, kernel, gamma)
    model.train(train_x, train_y, False)

    train_preds, train_acc = model.get_train_preds(train_x, train_y)
    print(f'Training Accuracy: {train_acc}')
    test_preds = model.get_test_preds(test_x)
    return test_preds