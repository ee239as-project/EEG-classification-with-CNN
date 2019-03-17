# Fix for issue loading Utils.preprocess_util
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Variable: wrapper for PyTorch tensor, stores gradients with requires_grad=True
# x.data, x.grad.data: values of tensor x, gradient values of x
from torch.autograd import Variable
from Utils.preprocess_util import *
from Utils.visualize import *
from Utils.cnn_helpers import *

# ----------------------------------- PREPROCESSING -----------------------------------
X_train,X_valid,X_test, Y_train,Y_valid,Y_test = load_preprocess_eeg_data(subsample_data=False)
'''
Training data: (177125, 22, 500)
Training target: (177125,)
Validation data: (87250, 22, 500)
Validation target: (87250,)
Test data: (55375, 22, 500)
Test target: (55375,)
'''

# ------------------------------------- CNN MODEL -------------------------------------
model = nn.Sequential()
# Recommended kernel size of 25
# After the two convolutions of the shallow ConvNet, a squaring nonlinearity, a mean pooling
# layer and a logarithmic activation function followed
# Input: (125,1,22,500)
model.add_module('conv_across_time', nn.Conv2d(1, 40, kernel_size=(1,51), stride=1))
model.add_module('conv_across_electrodes', nn.Conv2d(40, 40, kernel_size=(22,1), stride=1))
# apply batch norm to the output of conv layers before the nonlinearity
model.add_module('BatchNorm2d', nn.BatchNorm2d(40, momentum=0.1))
model.add_module('Nonlinearity', nn.ELU(inplace=True))
model.add_module('correct_dimensions', threed_to_twod())
model.add_module('AvgPool2d', nn.AvgPool2d(kernel_size=(1,76), stride=(1,15)))
model.add_module('drop', nn.Dropout(p=0.5))
model.add_module('Flatten', Flatten())
model.add_module('Fc_layer', nn.Linear(2360,4))
# model.add_module('correct_dimensions_2', twod_to_threed())
# model.add_module('conv_classifier', nn.Conv2d(40, 4, kernel_size=(1,25), stride=1))
# model.add_module('correct_dimensions_3', threed_to_oned())
torch.nn.init.xavier_uniform_(model.conv_across_time.weight, gain=1)
torch.nn.init.xavier_uniform_(model.conv_across_electrodes.weight, gain=1)
# torch.nn.init.xavier_uniform_(model.conv_classifier.weight, gain=1)

# CPU datatype: change to torch.cuda.FloatTensor to use GPU
dtype = torch.FloatTensor
model.type(dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)

# ---------------------------------- OPTIMIZATION ----------------------------------
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
wt_dcy = 0.01
amsgrad = False
'''
wt_scale = 0.01
reg = 0.001
lr_decay = 0.9 - Adam does implicitly
'''
params = model.parameters()
optimizer = optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wt_dcy, amsgrad=amsgrad)

# ------------------------------------- TRAINING -------------------------------------
'''
x = Variable(torch.tensor(X_train))
y = Variable(torch.tensor(Y_train), requires_grad=False)
x.type(dtype)
y.type(dtype)
for t in range(3):
    # calculate loss
    y_pred = model(x.float())
    loss = loss_fn(y_pred, y.type(torch.LongTensor))

    # backprop
    model.zero_grad()
    loss.backward()

    # update the parameters
    optimizer.step()
y_pred = model(x)
loss = loss_fn(y_pred, y.type(torch.LongTensor))
print(loss)
print(model)
'''
def check_early_stopping(val_acc, val_acc_history):
    n_epochs = 5
    prev_val_accs = val_acc_history[-(n_epochs+1):]
    if len(np.where(val_acc < prev_val_accs)) is 0:
        print('Stopping early')
        perform_plotting()
        raise SystemExit

n_train = X_train.shape[0] # 177125
n_validation = X_valid.shape[0] # 87250
batch_size = 100
iter_per_epoch = max(n_train // batch_size, 1) # 1417

n_epochs = 100
n_iter = n_epochs * iter_per_epoch # 708500

# Book-keeping
best_val_acc = 0
loss_history = []
train_acc_history = []
val_acc_history = []

epoch = 0
for t in range(n_iter):
    # Make a minibatch of training data
    batch_mask = np.random.choice(n_train, batch_size)
    # Input is a 2d-array with time steps as width and electrodes as height
    X_batch_tensor = threeD_to_fourDTensor(X_train[batch_mask]).float() # (125, 1, 22, 500)
    y_batch_tensor = Variable(torch.tensor(Y_train[batch_mask])).type(torch.LongTensor) # (125)

    # Compute loss and gradient
    y_pred = model(X_batch_tensor)
    loss = loss_fn(y_pred, y_batch_tensor)
    loss_history.append(loss.data)

    # Perform a parameter update
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training loss
    if (t%5 == 0):
        print('(Iteration %d / %d) loss: %f' % (t + 1, n_iter, loss_history[-1]))

    # At end of every epoch, increment epoch counters and consider decaying learning rate
    epoch_end = (t+1) % iter_per_epoch == 0
    if epoch_end:
        epoch += 1

    # Check train and val accuracy on first iteration, last iteration, and at end of each epoch
    first_iter = (t == 0)
    last_iter = (t == n_iter - 1)

    if first_iter or last_iter or epoch_end:
        train_acc = check_accuracy(model, X_train, Y_train)
        val_acc = check_accuracy(model, X_valid, Y_valid)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        best_val_acc = max(best_val_acc, val_acc)
        save_checkpoint(epoch, loss_history, train_acc_history, val_acc_history, best_val_acc)

        print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (epoch, n_epochs, train_acc, val_acc))
        check_early_stopping(val_acc, val_acc_history)

# ------------------------------------- PLOTTING -------------------------------------
# Save model
save_model(model)

# Perform plotting
perform_plotting(loss_history, train_acc_history, val_acc_history)

# Testing performance
test_acc = check_accuracy(model, X_test, Y_test)
print('Testing accuracy:', test_acc)
