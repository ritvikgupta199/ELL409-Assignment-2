import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        data_table  =  np.array(np.genfromtxt(self.file_path, delimiter=','))
        data_x, data_y =np.array(data_table[:,:-1]), np.array(data_table[:,-1])
        self.num_cls = len(np.unique(data_y))
        data = [[] for i in range(self.num_cls)]
        for (x, y) in zip(data_x, data_y):
            idx = int(y)-1
            data[idx].append(x)
        data = [np.array(d) for d in data]
        return data

    def get_data(self, split, n_feat=25):
        train_x, valid_x, train_y, valid_y = [], [], [], []
        for (i, d) in enumerate(self.data):
            n = len(d)
            s = int(np.floor(split * n))
            x_t, x_v = d[:s], d[s:n]
            y_t, y_v = (i+1.0)*np.ones(s), (i+1.0)*np.ones(n-s)
            train_x.append(x_t)
            train_y.append(y_t)
            valid_x.append(x_v)
            valid_y.append(y_v)
        train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
        valid_x, valid_y = np.concatenate(valid_x), np.concatenate(valid_y)
        train_x, valid_x = train_x[:, :n_feat], valid_x[:, :n_feat]
        return train_x, train_y, valid_x, valid_y
    
    def get_binary_data(self, split, cls, n_feat=25):
        c1, c2 = cls
        x1, x2 = self.data[c1-1], self.data[c2-1]
        n1, n2 = len(x1), len(x2)
        s1, s2 = int(np.floor(split * n1)), int(np.floor(split * n2))
        train_x = np.concatenate((x1[:s1], x2[:s2]))
        valid_x = np.concatenate((x1[s1:n1], x2[s2:n2]))
        train_y = np.concatenate((-1.0*np.ones(s1), np.ones(s2)))
        valid_y = np.concatenate((-1.0*np.ones(n1-s1), np.ones(n2-s2)))
        return train_x, train_y, valid_x, valid_y

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



data = DataLoader('data/2019MT10512.csv')
data.get_binary_data(0.5,(6,7))