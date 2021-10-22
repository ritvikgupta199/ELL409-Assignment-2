import numpy as np
from dataloader import DataLoader, TestDataLoader
from model import LIBSVMModel, CVXModel

def train_binary(data_path, c, split, cls):
    dataloader = DataLoader(data_path)
    train_x, train_y, valid_x, valid_y = dataloader.get_binary_data(split, cls)
    # model = LIBSVMModel(c, 'test')
    # model.train(train_x, train_y)
    model = CVXModel(10, 'linear')
    alphas = model.train(train_x, train_y)
    print(alphas)

train_binary('data/2019MT10512.csv',10, 0.5, (1,2))