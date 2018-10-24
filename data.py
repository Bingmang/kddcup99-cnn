import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class KddData(object):

    def __init__(self, batch_size):
        kddcup99 = datasets.fetch_kddcup99()
        self.kddcup_X = kddcup99.data
        self.kddcup_y = kddcup99.target
        self.encoder = {
            'protocal': LabelEncoder(),
            'service':  LabelEncoder(),
            'flag':     LabelEncoder(),
            'label':    LabelEncoder()
        }
        self.batch_size = batch_size
        self.train = (None, None)
        self.test = (None, None)


    def __encode_data(self):
        self.encoder['protocal'].fit(list(set(self.kddcup_X[:, 1])))
        self.encoder['service'].fit(list(set(self.kddcup_X[:, 2])))
        self.encoder['flag'].fit((list(set(self.kddcup_X[:, 3]))))
        self.encoder['label'].fit(list(set(self.kddcup_y)))
        self.kddcup_X[:, 1] = self.encoder['protocal'].transform(
            self.kddcup_X[:, 1])
        self.kddcup_X[:, 2] = self.encoder['service'].transform(
            self.kddcup_X[:, 2])
        self.kddcup_X[:, 3] = self.encoder['flag'].transform(
            self.kddcup_X[:, 3])
        self.kddcup_X = np.pad(self.kddcup_X, ((
            0, 0), (0, 64 - len(self.kddcup_X[0]))), 'constant').reshape(-1, 1, 8, 8)
        self.kddcup_y = self.encoder['label'].transform(self.kddcup_y)


    def __split_data(self):
        pass
