
# coding: utf-8

# In[2]:


# Fix for issue loading Utils.preprocess_util
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from Utils.preprocess_util import *
from Utils.visualize import *

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[3]:


X_train,X_valid,X_test,Y_train,Y_valid,Y_test = load_preprocess_eeg_data()


# In[ ]:


# create feature and targets tensor for train set
features_train = torch.from_numpy(X_train)
targets_train = torch.from_numpy(Y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set
features_test = torch.from_numpy(X_test)
targets_test = torch.from_numpy(Y_test).type(torch.LongTensor)

features_valid = torch.from_numpy(X_valid)
targets_valid = torch.from_numpy(Y_valid).type(torch.LongTensor)


# In[ ]:


class LSTMModel(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(LSTMModel, self).__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # 2 is the number of lstms stacked vertically(num_layers)
        self.lstm = nn.LSTM(self.n_inputs, self.n_neurons,2)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self,):
            # (num_layers, batch_size, n_neurons)
            # 2 is the number of lstms stacked vertically (num_layers)
            return (torch.zeros(2, self.batch_size, self.n_neurons))

    def forward(self, X):
            # transforms X to (n_steps, batch_size, n_inputs)
            X = X.permute(1, 0, 2)
            self.batch_size = X.size(1)
            self.hidden = self.init_hidden()
            self.cellstate = self.init_hidden()
            lstm_out, (self.hidden, self.cellstate)= self.lstm(X, (self.hidden,self.cellstate))
            out = self.FC(self.hidden[1])

            return out.view(-1, self.n_outputs) # (batch_size, n_output)


# In[ ]:


batch_size = 500
# Pytorch train and test sets
train = torch.utils.data.TensorDataset(features_train, targets_train)
valid = torch.utils.data.TensorDataset(features_valid, targets_valid)
test = torch.utils.data.TensorDataset(features_test, targets_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

# pprint.pprint(test_loader.dataset.tensors[0].size())


# In[ ]:


N_STEPS = 1000
N_INPUTS = 22
N_NEURONS = 10
N_OUTPUTS = 10
N_EPOCHS = 10

dataiter = iter(train_loader)
images, labels = dataiter.next()
model = LSTMModel(batch_size, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)

# (batch_size, n_steps, n_inputs)
images_modified = images.view(-1, 1000, 22)
logits = model(images_modified.float())
# print(logits[0:10])


# In[ ]:


dtype = torch.FloatTensor
n_iters = 10000
num_epochs = int(n_iters / (len(X_train)/batch_size))

model = LSTMModel(batch_size, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)

# Cross Entropy Loss
loss_fn = nn.CrossEntropyLoss().type(dtype)

# batch GD
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, amsgrad=False)

# Create RNN
input_dim = 22
seq_dim = 1000

train_loss = []
iterations = []
train_acc = []

count = 0
for epoch in range(num_epochs):
    for i, (signals, labels) in enumerate(train_loader):
        train  = Variable(signals.view(-1, seq_dim, input_dim))
        labels = Variable(labels )

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
        if count % 1 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for signals, labels in valid_loader:
                signals = Variable(signals.view(-1, seq_dim, input_dim))
                #print(signals.shape)
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
            if count % 1 == 0:
                print('Iteration: {}  Loss: {}  Valid Accuracy: {} %'.format(count, loss.data, accuracy))


# In[ ]:


X_valid_tensor = torch.from_numpy(X_valid.reshape(-1, seq_dim, input_dim))
print(X_valid_tensor.shape)
y_pred_valid = model( X_valid_tensor.float())
val_acc = get_accuracy(y_pred_valid, Y_valid,
    batch_size=X_valid.shape[0])
print('validation accuracy:', val_acc)


# In[ ]:


X_test_tensor = torch.from_numpy(X_test.reshape(-1, seq_dim, input_dim))
y_pred_test = model( X_test_tensor.float())
test_acc = get_accuracy(y_pred_test, Y_test,
    batch_size=X_test.shape[0])
print('test accuracy:', test_acc)
