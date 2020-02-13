import pandas as pd
import numpy as np
from zipfile import ZipFile

# import data
z = ZipFile("./caltech-cs155-2020.zip")

dfs = {text_file.filename: pd.read_csv(z.open(text_file.filename))
        for text_file in z.infolist()
        if text_file.filename.endswith('.csv')}
df_train = dfs['train.csv']
df_test = dfs['test.csv']
df_combined = pd.concat((dfs['train.csv'],dfs['test.csv']))

import pdb; pdb.set_trace()

# Remove all rows with NaN to have it cleaner
array_train = df_train.to_numpy()
array_no_NaN = array_train[~np.isnan(array_train).any(axis = 1)]

# Normalize feature vectors
norm_train = np.empty_like(array_train)
for i in range(28):
    col_max = max(array_no_NaN[:, i])
    col_min = min(array_no_NaN[:, i])
    norm_train[:, i] = (array_train[:, i] - col_min) / (col_max - col_min)


