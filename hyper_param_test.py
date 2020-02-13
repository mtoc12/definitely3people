# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import math
import numbers
import torch
from torch import nn
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
import sklearn.metrics

import utils


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# %%
import importlib
importlib.reload(utils)


# %%
idtrain, Xtrain, ytrain = utils.load_train_data()
idtest, Xtest = utils.load_test_data()


# # %%
# print(Xtrain.shape)
# plt.imshow(np.corrcoef(Xtrain.transpose()), cmap='hot')
# plt.show()


# # %%
# for i in range(Xtrain.shape[1]):
#     print('i=' + repr(i) + ' has '+
#          repr(np.count_nonzero(np.isnan(Xtrain[:,i]))) +
#          ' nan values out of ' + repr(Xtrain.shape[0]))

# %%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.features = list(range(X.shape[1]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx,self.features], self.y[idx]
        else:
            return self.X[idx, self.features], 


# %%
training_samples = math.floor(len(Xtrain))

Xtrain_tensor = torch.FloatTensor(Xtrain[:training_samples, ~np.isnan(sum(Xtrain))])
ytrain_tensor = torch.FloatTensor(ytrain[:training_samples])
Xtest_tensor = torch.FloatTensor(Xtest)

# Normalize data
#for i in range(Xtrain_tensor.shape[1]):
#    col_max = max(Xtrain_tensor[:, i])
#    col_min = min(Xtrain_tensor[:,i])
#    Xtrain_tensor[:, i] = (Xtrain_tensor[:, i] - col_min) / (col_max - col_min)

test_features = range(24)
print('Testing features ' + str(test_features))

num_datapoints = len(Xtrain_tensor)/100
slice_idx = math.floor(num_datapoints)

train_dataset = MyDataset(Xtrain_tensor[slice_idx:,test_features], ytrain_tensor[slice_idx:])
test_dataset = MyDataset(Xtrain_tensor[:slice_idx,test_features], ytrain_tensor[:slice_idx])
actualtest_dataset = MyDataset(Xtest_tensor)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

learning_rates = [1e-2]# , 1e-3, 1e-4]
# layer_sizes = [[1000], [500, 500], [750,250],[250,750],[300, 600, 300],[150,600,450],[450,600,150],[150,900,150]]
# layer_sizes = [[1000], [500, 500], [150, 900, 150]]
layer_sizes = [[150, 900, 150]]
eta = 1e-2

results = []

print('AUC,         eta,         layers')

for eta in learning_rates:
    # print('-------- eta = ' + str(eta) + ' --------')
    for layers in layer_sizes:
        # print('---- hiden layer sizes = ' + str(layers) + ' ----')
        torch.manual_seed(155)
        # Get a clean model
        modules = []
        modules.append(nn.Flatten())
        size_last = 24
        for i, layer in enumerate(layers):
            if isinstance(layers[i], numbers.Number):
                size_next = layers[i]
                modules.append(nn.Linear(size_last, size_next))
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm1d(size_next))
                modules.append(nn.Dropout(0.1))
                size_last = size_next
            else:
                pass
        modules.append(nn.Linear(size_last, 1))

        model = nn.Sequential(*modules).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        loss_fn = nn.MSELoss()

        # Some layers, such as Dropout, behave differently during training
        model.train()

        for epoch in range(10):
            i = 0
            for batch_idx, (data, target) in enumerate(train_dataloader):
                if i % (math.floor(100)) == 0:
                    # print('batch number = %i' % i)
                    pass
                i += 1
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()# Erase accumulated gradients
                output = model(data)# Forward pass
                loss = loss_fn(output, target)# Calculate loss
                loss.backward()# Backward pass
                optimizer.step()# Weight update

            # Track loss each epoch
            print('Train Epoch: %d  Loss: %.4f' % (epoch + 1,  loss.item()))
            
        # Putting layers like Dropout into evaluation mode
        model.eval()

        test_loss = 0
        test_auc = 0
        correct = 0

        # Turning off automatic differentiation
        with torch.no_grad():
            for data, target in test_dataloader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()  # Sum up batch loss
                test_auc += sklearn.metrics.roc_auc_score(target.data.cpu().numpy(), output.data.cpu().numpy())
                pred = output.round()  # Get the index of the max class score
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dataloader.dataset)
        test_auc /= len(test_dataloader)

        accuracy = 100. * correct / len(test_dataloader.dataset)

        #print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f), AUC: %.4f' %
                # (test_loss, correct, len(test_dataloader.dataset),
                # accuracy, test_auc))

        print('%.4f,      %.4f,      %s' % (test_auc, eta, layers))

        results.append({'auc': test_auc, 'eta': eta, 'layers': layers})

# print(results)

for data, target in actualtest_dataloader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)


with open('results.csv', mode='w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i, y in zip(idxtest, output):
    	csvwriter.writerow(i, y)

# %%






































# %%


