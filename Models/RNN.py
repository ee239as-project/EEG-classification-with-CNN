
# coding: utf-8

# In[14]:


# Fix for issue loading Utils.preprocess_util
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pprint
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Utils.preprocess_util import *
from Utils.visualize import *

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[6]:


X_train,X_valid,X_test,Y_train,Y_valid,Y_test = load_preprocess_eeg_data()


# In[8]:


# create feature and targets tensor for training set
features_train = torch.from_numpy(X_train)
targets_train = torch.from_numpy(Y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set
features_test = torch.from_numpy(X_test)
targets_test = torch.from_numpy(Y_test).type(torch.LongTensor) # data type is long

features_valid = torch.from_numpy(X_valid)
targets_valid = torch.from_numpy(Y_valid).type(torch.LongTensor)


# In[36]:


batch_size = 1000

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(features_train, targets_train)
valid = torch.utils.data.TensorDataset(features_valid, targets_valid)
test = torch.utils.data.TensorDataset(features_test, targets_test)

# data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size) # torch.Size([1417, 22, 1000])
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size) # torch.Size([698, 22, 1000])
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size) # torch.Size([443, 22, 1000])

# pprint.pprint(test_loader.dataset.tensors[0].size())


# In[38]:


class RNNModel(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(RNNModel, self).__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # RNN
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)

        # readout layer
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self,):
            # (num_layers, batch_size, n_neurons)
            return (torch.zeros(1, self.batch_size, self.n_neurons))

    def forward(self, X):
            # transforms X to (n_steps, batch_size, n_inputs)
            #X = X.permute(1, 0, 2)
            X = X.permute(2, 0, 1)

            self.batch_size = X.size(1)
            self.hidden = self.init_hidden()

            lstm_out, self.hidden = self.basic_rnn(X, self.hidden)
            out = self.FC(self.hidden)

            return out.view(-1, self.n_outputs) # (batch_size, n_output)


# In[39]:


'''
dataiter = iter(train_loader)
images, labels = dataiter.next()

# (batch_size, n_steps, n_inputs)
logits = model(images.view(-1, 1000,22).float())
print(logits[0:10])
'''

N_STEPS = 1000
N_INPUTS = 22
N_NEURONS = 10
N_OUTPUTS = 10

model = RNNModel(batch_size, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)

# Cross Entropy Loss
dtype = torch.FloatTensor
loss_fn = nn.CrossEntropyLoss().type(dtype)

# batch GD
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, amsgrad=False)

train_loss = []
iterations = []
train_acc = []
count = 0
num_epochs = 25
for epoch in range(num_epochs):
    for i, (signals, labels) in enumerate(train_loader):
        # train = Variable(signals.view(-1, N_STEPS, N_INPUTS))
        train = Variable(signals.view(-1, N_INPUTS, N_STEPS))

        labels = Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # reset hidden states
        model.hidden = model.init_hidden()

        # Forward propagation
        outputs = model(train.float())

        # Calculate softmax and cross entropy loss
        loss = loss_fn(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        #print("parameters===",list(model.parameters())[0].data)
        count += 1
        if count % 10 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for signals, labels in valid_loader:
                # signals = Variable(signals.view(-1, N_STEPS, N_INPUTS))
                signals = Variable(signals.view(-1, N_INPUTS, N_STEPS))
                # print(signals.shape)
                # Forward propagation
                outputs_valid = model(signals.float())

                # Get predictions from the maximum value
                predicted = torch.max(outputs_valid.data, 1)[1]

                # Total number of labels
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / float(total)

            # store loss and iteration
            train_loss.append(loss.data)
            iterations.append(count)
            train_acc.append(accuracy)

        if count % 10 == 0:
                print('Iteration: {}  Loss: {}  Valid Accuracy: {} %'.format(count, loss.data, accuracy))


# In[ ]:


X_valid_tensor = Variable(torch.tensor(X_valid))
# print(X_valid.shape, X_valid_tensor.shape)
y_pred_valid = model(X_valid_tensor.float())
val_acc = get_accuracy(y_pred_valid, Y_valid,
    batch_size = X_valid.shape[0])
print('validation accuracy:', val_acc)


# In[ ]:


X_test_tensor = Variable(torch.tensor(X_test))
y_pred_test = model( X_test_tensor.float())
test_acc = get_accuracy(y_pred_test, Y_test,
    batch_size=X_test.shape[0])
print(test_acc)
