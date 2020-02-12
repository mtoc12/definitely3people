# Import standard library modules
import zipfile
from warnings import warn
# Import third-party modules
import numpy as np
import torch
# Import custom modules


def load_train_data(datafolder='./data/', skiprows=1, shuffle=False):
    ''' Try running this command
            Xtrain,ytrain = utils.load_train_data()
    '''
    if shuffle:
        warn('Data shuffling not implemented yet; returning unshuffled data')
    traindata = np.genfromtxt(datafolder + 'train.csv', skip_header=skiprows, delimiter=',')
    Xtrain = traindata[:, :-1]
    ytrain = traindata[:, -1]
    return Xtrain, ytrain

def load_test_data(datafolder='./data/', skiprows=1, shuffle=False):
    ''' Try running this command
            Xtest = utils.load_test_data()
    '''
    return np.genfromtxt(datafolder + 'test.csv', skip_header=skiprows, delimiter=',')

def load_train_Dataset(*args, **kwargs):
    ''' Functions similar to load_train_data(), but instead returns a torch Dataset object.
        Try running
            train_dataset = utils.load_train_Dataset()
    '''
    dataset = torch.utils.data.TensorDataset(load_test_data(**kwargs))
    return dataset

# def load_test_Dataset(*args, **kwargs):
#     ''' Functions similar to load_train_data(), but instead returns a torch Dataset object.
#         Try running
#             train_dataset = utils.load_train_Dataset()
#     '''
#     dataset = torch.utils.data.TensorDataset(load_test_data(**kwargs))
#     return dataset
    

def unzip_data(datafolder='./data'):
    with zipfile.ZipFile('caltech-cs155-2020.zip', 'r') as zip_ref:
        zip_ref.extractall(datafolder)

def setup():
    unzip_data()

