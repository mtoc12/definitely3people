# Import standard library modules
import zipfile
from warnings import warn
import os.path
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

    if not(os.path.exists(datafolder)):
        unzip_data(datafolder=datafolder)
    traindata = np.genfromtxt(datafolder + 'train.csv', skip_header=skiprows, delimiter=',')
    idtrain = traindata[:,0]
    Xtrain = traindata[:, 1:-1]
    ytrain = traindata[:, -1]
    return idtrain, Xtrain, ytrain

def load_test_data(datafolder='./data/', skiprows=1, shuffle=False):
    ''' Try running this command
            Xtest = utils.load_test_data()
    '''
    if shuffle:
        warn('Data shuffling not implemented yet; returning unshuffled data')

    if not(os.path.exists(datafolder)):
        unzip_data(datafolder=datafolder)
    testdata = np.genfromtxt(datafolder + 'test.csv', skip_header=skiprows, delimiter=',')
    idtest = testdata[:,0]
    Xtest = testdata[:,1:-1]
    return idtest, Xtest

def load_train_Dataset(*args, **kwargs):
    ''' Functions similar to load_train_data(), but instead returns a torch Dataset object.
        Try running
            train_dataset = utils.load_train_Dataset()
    '''
    idtrain, Xtrain, ytrain = load_test_data(**kwargs)
    dataset = torch.utils.data.TensorDataset(Xtrain, ytrain)
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
        print('Unzipping data to folder''' + datafolder + "'")
        zip_ref.extractall(datafolder)

def setup():
    unzip_data()