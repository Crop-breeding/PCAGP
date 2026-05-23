import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def load_data():
    X = pd.read_csv("C:\\Users\\Administrator\\Desktop\\PCAGP\\dataset\\2000gene.csv").values
    Y = pd.read_csv("C:\\Users\\Administrator\\Desktop\\PCAGP\\dataset\\2000_2_phe.csv")["testw"]


    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y.values.reshape(-1, 1))
    print(f"X shape:{X.shape},Y shape:{Y.shape}")
    return X, Y
X, Y = load_data()



def calculate_padding(feature_dim):
    """Calculate the target matrix shape so that it becomes as close to a square as possible.。"""
    side_length = int(np.ceil(np.sqrt(feature_dim)))
    return side_length


def snp_to_matrix(X):
    """The SNP data is converted into a matrix format that matches the model inputs and populated."""
    target_size = calculate_padding(X.shape[1])
    padding_size = target_size ** 2 - X.shape[1]

    if padding_size > 0:
        mean_value = torch.round(X.mean(dim=1, keepdim=True))
        X = torch.cat((X, mean_value.repeat(1, padding_size)), dim=1)

    return X.view(-1, 1, target_size, target_size)
