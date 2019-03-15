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
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Utils.preprocess_util import *
from Utils.visualize import *


# ----------------------------------- PREPROCESSING -----------------------------------
# Load EEG data
X_train,X_valid,X_test, Y_train,Y_valid,Y_test = load_preprocess_eeg_data()

# create feature and targets tensor for train set
features_train = torch.from_numpy(X_train)
targets_train = torch.from_numpy(Y_train).type(torch.LongTensor) # data type is long
features_valid = torch.from_numpy(X_valid)
targets_valid = torch.from_numpy(Y_valid).type(torch.LongTensor)
features_test = torch.from_numpy(X_test)
targets_test = torch.from_numpy(Y_test).type(torch.LongTensor)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(features_train, targets_train)
valid = torch.utils.data.TensorDataset(features_valid, targets_valid)
test = torch.utils.data.TensorDataset(features_test, targets_test)

# data loaders
batch_size = 500
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
# pprint.pprint(test_loader.dataset.tensors[0].size())

input_dim = 22
seq_dim = 500
X_train_tensor = torch.from_numpy(X_train.reshape(-1, seq_dim, input_dim))
X_valid_tensor = torch.from_numpy(X_valid.reshape(-1, seq_dim, input_dim))
X_test_tensor = torch.from_numpy(X_test.reshape(-1, seq_dim, input_dim))

# ------------------------------------- LSTM MODEL -------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs,n_layers,dropout):
        super(LSTMModel, self).__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.num_layers = n_layers
        self.lstm = nn.LSTM(self.n_inputs, self.n_neurons, self.num_layers)
        # self.lstm.weight_hh_l0.data.fill_(0)
        # torch.nn.init.xavier_uniform_(self.lstm.weight_ih_l0.data)
        # torch.nn.init.orthogonal_(self.lstm.weight_hh_l0.data)

        # initialising w(rec) to I and b(rec) to 0
        ih_size = list(self.lstm.weight_ih_l0.data.shape)
        hh_size = list(self.lstm.weight_hh_l0.data.shape)
        self.lstm.weight_ih_l0.data.copy_(torch.eye(ih_size[0], ih_size[1]))
        self.lstm.weight_hh_l0.data.copy_(torch.eye(hh_size[0], hh_size[1]))
        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)

        self.dropout = nn.Dropout(p=dropout)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self,):
        # (num_layers, batch_size, n_neurons)
        return torch.zeros(self.num_layers, self.batch_size, self.n_neurons)
        # return torch.nn.init.xavier_uniform_((self.num_layers, self.batch_size, self.n_neurons), gain=1)

    def forward(self, X):
        # transforms X to (n_steps, batch_size, n_inputs)
        X = X.permute(1, 0, 2)
        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()
        self.cellstate = self.init_hidden()
        lstm_out, (self.hidden, self.cellstate)= self.lstm(X, (self.hidden, self.cellstate))
        hidden_out = self.hidden[self.num_layers - 1]
        dropout_out = self.dropout(hidden_out)
        out = self.FC(dropout_out)

        return out.view(-1, self.n_outputs) # (batch_size, n_output)

N_STEPS = 500
N_INPUTS = 22
N_NEURONS = 75
N_OUTPUTS = 4
N_EPOCHS = 10
N_LAYERS = 1 # actually how many LSTMs are stacked on each other
dropout = 0

dtype = torch.FloatTensor
model = LSTMModel(batch_size, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS, N_LAYERS, dropout)
loss_fn = nn.CrossEntropyLoss().type(dtype)

# ----------------------------------- HYPERPARAMETERS ----------------------------------
lr = 1e-6
betas = (0.9, 0.999)
eps = 1e-8
wt_dcy = 0
amsgrad = False

# Optimizer
params = model.parameters()
optimizer = optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wt_dcy, amsgrad=amsgrad)
# optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay =0, betas=(0.9, 0.999),
# amsgrad=False)
# optimizer = optim.RMSprop(model.parameters(),lr=0.001, alpha=0.99, eps=1e-08,
# weight_decay=0, momentum=0.1)

# ------------------------------------- TRAINING -------------------------------------
n_epochs = 1 # int(n_iter / (len(X_train)/batch_size))

# Book-keeping
best_val_acc = 0
loss_history = []
train_acc_history = []
val_acc_history = []

count = 0
for epoch in range(n_epochs):
    for i, (signals, labels) in enumerate(train_loader):
        train  = Variable(signals.view(-1, seq_dim, input_dim))
        labels = Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # reset hidden states
        model.hidden = model.init_hidden()

        # Forward propagation
        outputs = model(train.float())

        # Calculate softmax and cross entropy loss
        loss = loss_fn(outputs, labels)
        loss_history.append(loss.data)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()
        #print("parameters===",list(model.parameters())[0].data)

        count += 1
        if count % 1 == 0:
            y_pred_train = model(train.float())
            train_acc = get_accuracy(y_pred_train, labels, batch_size=labels.shape[0])
            train_acc_history.append(train_acc)

            indices = np.random.choice(X_valid.shape[0], 50, replace=False)
            X_valid_tensor = torch.from_numpy(X_valid[indices].reshape(-1, seq_dim, input_dim))
            y_pred_valid = model(X_valid_tensor.float())
            val_acc = get_accuracy(y_pred_valid, Y_valid[indices], batch_size=50)
            val_acc_history.append(val_acc)

            # print('Iteration: {}  Loss: {}' .format(count, loss.data))
            print('Iteration: {}  Loss: {}  Train Accuracy: {} Valid Accuracy: {} %'.format(count,
             loss.data, train_acc, val_acc))

            '''
            # if(train_acc_history> 35 and val_acc>35):
                # return

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
            loss_history.append(loss.data)
            iterations.append(count)
            train_acc_history.append(accuracy)
            print('Iteration: {}  Loss: {}  Valid Accuracy: {} %'.format(count, loss.data,
            accuracy))
            '''

# ------------------------------------- PLOTTING -------------------------------------
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Training loss over iterations
plt.subplot(2, 1, 1)
plt.plot(loss_history, 'o')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Training and validation accuracy over epochs
plt.subplot(2, 1, 2)
plt.plot(train_acc_history, '-o')
plt.plot(val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
