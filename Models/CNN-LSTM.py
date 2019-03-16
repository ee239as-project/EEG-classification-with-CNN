
# coding: utf-8

# In[3]:


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
import torch.utils.data
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


X_train,X_valid,X_test,Y_train,Y_valid,Y_test = load_preprocess_eeg_data()


# In[5]:


indices = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
X_train = X_train[indices]
Y_train = Y_train[indices]


# In[6]:


indices = np.random.choice(X_valid.shape[0], X_valid.shape[0], replace=False)
X_valid = X_valid[indices]
Y_valid = Y_valid[indices]

indices = np.random.choice(X_test.shape[0], X_test.shape[0], replace=False)
X_test = X_test[indices]
Y_test = Y_test[indices]


# In[7]:


# create feature and targets tensor for train set
features_train = torch.from_numpy(X_train)
targets_train = torch.from_numpy(Y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set
features_test = torch.from_numpy(X_test)
targets_test = torch.from_numpy(Y_test).type(torch.LongTensor)

features_valid = torch.from_numpy(X_valid)
targets_valid = torch.from_numpy(Y_valid).type(torch.LongTensor)


# In[8]:


class Flatten(nn.Module):
    def forward(self, x):
        a= x.view(x.size(0), -1)
        return a
    
class threed_to_twod(nn.Module):
    def forward(self, x):
        # print(x.shape)
        a = x.reshape(x.shape[0],x.shape[3],x.shape[1])
        print ("a=",a.shape)
        return a

class permute(nn.Module):
    def forward(self, x):
        # print(x.shape)
        a = x.permute(1, 0, 2) 
        #print ("a=",a.shape)
        return a
        #,((torch.zeros(num_layers, batch_size, n_neurons)),(torch.zeros(num_layers, batch_size, n_neurons)))

class get_hidden(nn.Module):
    def forward(self, x):
        #print(x.shape)
        lstm_out, (hidden, cellstate) = x  
        print ("hidden=",hidden.shape)
        return hidden


# In[23]:


class LSTMModel(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs,n_layers,droput):
        super(LSTMModel, self).__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.num_layers = n_layers
        self.cnn1 = nn.Conv2d(1,40,kernel_size=(1,51),stride = 1)
        self.cnn2= nn.Conv2d(40,40,kernel_size=(22,1),stride = 1)
        self.batchnorm= nn.BatchNorm2d(40,momentum=0.1)
        self.relu = nn.ReLU()
        self.threed_to_twod = threed_to_twod()
        self.avgpool2d =  nn.AvgPool2d(kernel_size=(1,135),stride = (1,5))
        self.dropout =  nn.Dropout(p=droput)
        self.lstm = nn.LSTM(self.n_inputs, self.n_neurons,self.num_layers) 
        #self.lstm.weight_hh_l0.data.fill_(0)
        #torch.nn.init.xavier_uniform_(self.lstm.weight_ih_l0.data )
        #torch.nn.init.orthogonal_(self.lstm.weight_hh_l0.data)
        
        torch.nn.init.xavier_uniform_(self.cnn1.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.cnn2.weight, gain=1)

        
        #initialising w(rec) to I and b(rec) to 0 
        ih_size = list(self.lstm.weight_ih_l0.data.shape)
        hh_size =list(self.lstm.weight_hh_l0.data.shape)
        self.lstm.weight_ih_l0.data.copy_(torch.eye(ih_size[0],ih_size[1]))
        self.lstm.weight_hh_l0.data.copy_(torch.eye(hh_size[0],hh_size[1]))
        
        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        
        self.droput = nn.Dropout(p=droput)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)
        
    def init_hidden(self,):
            # (num_layers, batch_size, n_neurons)
            return (torch.zeros(self.num_layers, self.batch_size, self.n_neurons))
            #return torch.nn.init.xavier_uniform_((self.num_layers, self.batch_size, self.n_neurons), gain=1)

    def forward(self, X):
            # transforms X to (n_steps, batch_size, n_inputs0
            conv_out1 = self.cnn1(X)
            conv_out2 = self.cnn2(conv_out1)
            batchnorm_out = self.batchnorm(conv_out2)
            relu_out = self.relu(batchnorm_out)
            AvgPool2d_out = self.avgpool2d(relu_out)
            drop_out = self.dropout(AvgPool2d_out)
            # transforms X to (n_steps, batch_size, n_inputs)
            X_new = drop_out.view(drop_out.shape[0],drop_out.shape[1],drop_out.shape[3])  
            X_new = X_new.permute(2, 0, 1) 
            self.batch_size = X_new.size(1)
            self.hidden = self.init_hidden()
            self.cellstate = self.init_hidden()
            #torch.nn.init.xavier_uniform_(self.hidden)
            #torch.nn.init.xavier_uniform_(self.cellstate)
            lstm_out, (self.hidden, self.cellstate)= self.lstm(X_new, (self.hidden,self.cellstate))
            hidden_out =self.hidden[self.num_layers-1]
            hidden_out = hidden_out.view(1,hidden_out.shape[0],hidden_out.shape[1])
            cellstate_out =self.hidden[self.num_layers-1]
            cellstate_out  = cellstate_out.view(1,cellstate_out.shape[0],cellstate_out.shape[1])
            lstm_out, (self.hidden, self.cellstate)= self.lstm(X_new, (hidden_out,cellstate_out))
            hidden_out =self.hidden[self.num_layers-1]
            dropout_out = self.droput(hidden_out)
            out = self.FC(dropout_out)

            return out.view(-1, self.n_outputs) # (batch_size, n_output)


# In[29]:


batch_size = 500
# Pytorch train and test sets
train = torch.utils.data.TensorDataset(features_train, targets_train)
valid = torch.utils.data.TensorDataset(features_valid, targets_valid)
test = torch.utils.data.TensorDataset(features_test, targets_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

# pprint.pprint(test_loader.dataset.tensors[0].size())


# In[30]:


N_STEPS = 500
N_INPUTS = 40
N_NEURONS = 75
N_OUTPUTS = 10
N_EPOCHS = 10
N_LAYERS = 1# This actually corresponds to how many lsts are stacked one above the other
droput = 0
dataiter = iter(train_loader)
signals, labels = dataiter.next()
model = LSTMModel(batch_size, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS,N_LAYERS,droput)

signals_modified= threeD_to_fourDTensor(signals)
print(signals_modified.shape)
logits = model(signals_modified.float())
print(logits[0:10])


# In[ ]:


dtype = torch.FloatTensor
n_iters = 10000
num_epochs = int(n_iters / (len(X_train)/batch_size))
model = LSTMModel(batch_size, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS,N_LAYERS,droput)

# Cross Entropy Loss 
loss_fn = nn.CrossEntropyLoss().type(dtype)

# batch GD
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.1)
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)

train_loss = []
iterations = []
train_acc = []

#X_valid_tensor = threeD_to_fourDTensor(X_valid)
#X_train_tensor = threeD_to_fourDTensor(X_train)

#print("X=",X_train_tensor.shape)
print("num_epochs = ", num_epochs)
print("n_iters = ", n_iters)
print("starting training..")

num_train = X_train.shape[0]
num_valid = X_valid.shape[0]
batch_size = 100

count = 0
num_epochs = 2
print("starting training..")

for epoch in range(num_epochs):
    print("epoch=",epoch)
    # reset hidden states
    
    for i, (signals, labels) in enumerate(train_loader):
        #train  = Variable(signals.view(-1, seq_dim, input_dim))
        X_batch_tensor = threeD_to_fourDTensor(signals)
        labels = Variable(labels )
        '''
        for t in range(n_iters):
        batch_mask = np.random.choice(num_train, batch_size,replace = False)
        X_batch = X_train[batch_mask]
        y_batch = Y_train[batch_mask]
        X_batch_tensor = threeD_to_fourDTensor(X_batch)
        y_batch_tensor = Variable(torch.tensor(y_batch))

        y_pred = model( X_batch_tensor.float())                
        '''
        
        y_pred = model( X_batch_tensor.float())
        # Calculate softmax and cross entropy loss
        loss = loss_fn(y_pred,labels.type(torch.LongTensor))
              
        model.zero_grad()
        loss.backward()
        optimizer.step()
                    
        #print("parameters===",list(model.parameters())[0].data)

        count += 1
        train_loss.append(loss.data)
        iterations.append(count)
        if count % 1 == 0:
                X_train_tensor =threeD_to_fourDTensor(X_train[0:100,:,:])
                y_pred_train = model( X_train_tensor.float())
                train_acc = get_accuracy(y_pred_train, Y_train[0:100],
                    batch_size=100)

                X_valid_tensor = threeD_to_fourDTensor(X_valid[0:100,:,:])
                y_pred_valid = model( X_valid_tensor.float())
                val_acc = get_accuracy(y_pred_valid, Y_valid[0:100],
                    batch_size=100)
                
                print('Iteration: {}  Loss: {}  Train Accuracy: {} Valid Accuracy: {} %'.format(count, loss.data,train_acc,
                                                                                            val_acc))


# In[28]:


import matplotlib.pyplot as plt
plt.plot(iterations,train_loss)
plt.show();

