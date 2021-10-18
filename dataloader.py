import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_x, self.target = self.load_data()

    def load_data(self):
        data_table  =  np.array(np.genfromtxt(self.file_path, delimiter=','))
        x, y =np.array(data_table[:,:-1]), np.array(data_table[:,-1])
        return x, y

    def get_data(self, n_feat=25):
        x, y = self.data_x[:, :n_feat], self.target
        return x, y
            
    def get_binary_data(self, cls, n_feat=25):
        c1, c2 = cls
        idx1 = np.where(self.target==c1)[0]
        idx2 = np.where(self.target==c2)[0]
        n1, n2 = idx1.shape[0], idx2.shape[0]
        idx = np.concatenate((idx1, idx2))
        x = self.data_x[idx]
        y = np.concatenate((np.ones(n1)*-1.0, np.ones(n2)))
        return x, y


class TestDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_x = self.load_data()

    def load_data(self):
        data_x = np.array(np.genfromtxt(self.file_path, delimiter=','))   
        return data_x
    
    def get_data(self, n_feat=25):
        x = self.data_x[:, :n_feat]
        return x



data = TestDataLoader('data/2019MT10512.csv')
data.get_binary_data(15)