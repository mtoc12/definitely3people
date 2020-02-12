import zipfile
import numpy as np
from warnings import warn

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

def unzip_data(datafolder='./data'):
    with zipfile.ZipFile('caltech-cs155-2020.zip', 'r') as zip_ref:
        zip_ref.extractall(datafolder)

def setup():
    unzip_data()

