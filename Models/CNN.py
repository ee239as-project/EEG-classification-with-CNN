# Fix for issue loading Utils.preprocess_util
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim  as optim
from torch.autograd import Variable
from Utils.preprocess_util import *
from Utils.visualize import *


# ----------------------------------- PREPROCESSING -----------------------------------
X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_preprocess_eeg_data()

class Flatten(nn.Module):
    def forward(self, x):
        a = x.view(x.size(0), -1)
        return a

class threed_to_twod(nn.Module):
    def forward(self, x):
        # print(x.shape)
        a = x.reshape(x.shape[0],x.shape[3],x.shape[1])
        # print (a)
        return a

# ------------------------------------- CNN MODEL -------------------------------------
model = nn.Sequential()
model.add_module('conv_across_time',nn.Conv2d(1,40,kernel_size=(1,51),stride = 1))
model.add_module('conv_across_electrodes',nn.Conv2d(40,40,kernel_size=(22,1),stride = 1))
model.add_module('BatchNorm2d',nn.BatchNorm2d(40,momentum=0.1))
model.add_module('Nonlinearity', nn.ReLU())
model.add_module('correct_dimensions',threed_to_twod())
model.add_module('AvgPool2d',nn.AvgPool2d(kernel_size=(135,1),stride = (5,1)))
model.add_module('drop', nn.Dropout(p=0.5))
model.add_module('Flatten',Flatten())
model.add_module('Fc_layer',nn.Linear(2560,10))
torch.nn.init.xavier_uniform_(model.conv_across_time.weight, gain=1)
torch.nn.init.xavier_uniform_(model.conv_across_electrodes.weight, gain=1)

dtype = torch.FloatTensor
model.type(dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)

# N,C,H,W = 18,1,25,1000
# x = Variable(torch.tensor(X_train.reshape((18,1, 25, 1000))))
x = Variable(torch.tensor(X_train))
y = Variable(torch.tensor(Y_train), requires_grad=False)
dtype = torch.FloatTensor
x.type(dtype)
y.type(dtype)

'''
for t in range(3):
    y_pred = model( x.float())
    loss = loss_fn(y_pred,y.type(torch.LongTensor))
    print(loss.data)
    model.zero_grad()
    loss.backward()
loss = loss_fn(y_pred,y.type(torch.LongTensor))
'''

# ------------------------------------- TRAINING -------------------------------------
num_train = X_train.shape[0]
num_valid = X_valid.shape[0]
batch_size = 100
num_epochs = 50
iterations_per_epoch = max(num_train // batch_size, 1)
num_iterations = num_epochs * iterations_per_epoch
epoch = 1
train_loss = []
iterations = []

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0, amsgrad=False)

for t in range(num_iterations):
    batch_mask = np.random.choice(num_train, batch_size)
    X_batch = X_train[batch_mask]
    y_batch = Y_train[batch_mask]
    X_batch_tensor = threeD_to_fourDTensor(X_batch)
    y_batch_tensor = Variable(torch.tensor(y_batch))

    y_pred = model(X_batch_tensor.float())

    loss = loss_fn(y_pred, y_batch_tensor.type(torch.LongTensor))

    train_loss.append(loss.data)
    iterations.append(t)

    model.zero_grad()
    loss.backward()
    optimizer.step()
    if (t%10 == 0):
        print('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations,
                                                loss.detach().numpy()))

    epoch_end = (t + 1) % iterations_per_epoch == 0

    if epoch_end:
        epoch += 1

    first_it = (t == 0)
    last_it = (t == num_iterations - 1)

    if first_it or last_it or epoch_end:
        X_train_tensor = threeD_to_fourDTensor(X_train[0:100,:,:])
        y_pred_train = model( X_train_tensor.float())
        train_acc = get_accuracy(y_pred_train, Y_train[0:100], batch_size=50)

        X_valid_tensor = threeD_to_fourDTensor(X_valid[0:50,:,:])
        y_pred_valid = model( X_valid_tensor.float())
        val_acc = get_accuracy(y_pred_valid, Y_valid[0:50], batch_size=50)
        print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (epoch, num_epochs,
                                                              train_acc, val_acc))


# ------------------------------------- ACCURACY -------------------------------------
# Validation accuracy
get_accuracy_in_batches(model, X_valid, Y_valid, seq_dim, input_dim, 'validation',
                        iterations=500, cnn=True)

# Testing accuracy
get_accuracy_in_batches(model, X_test, Y_test, seq_dim, input_dim, 'testing',
                        iterations=500, cnn=True)

# Training accuracy
get_accuracy_in_batches(model, X_train, Y_train, seq_dim, input_dim, 'training',
                        iterations=500, cnn=True)

plt.plot(iterations, train_loss)
plt.show()
