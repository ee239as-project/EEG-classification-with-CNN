import torch.nn as nn

# ---------------------------------- CNN MODEL HELPERS ----------------------------------
class Flatten(nn.Module):
    def forward(self, x):
        # example x.size: ([125, 64, 40])
        a = x.view(x.size(0), -1)
        return a

class twod_to_threed(nn.Module):
    def forward(self, x):
        # print('twod_to_threed x:', x.shape)
        a = x.reshape(x.shape[0], x.shape[1], -1, x.shape[2])
        # print('twod_to_threed a:', a.shape)
        return a

class threed_to_twod(nn.Module):
    def forward(self, x):
        # example x.shape: ([125, 40, 1, 450])
        # print('x:', x.shape)
        a = x.reshape(x.shape[0], x.shape[1], x.shape[3])
        # example a.shape: ([125, 450, 40])
        # print('a:', a.shape)
        return a

class threed_to_oned(nn.Module):
    def forward(self, x):
        # print('threed_to_oned x:', x.shape)
        a = x.reshape(x.shape[0], -1)
        # print('threed_to_oned a:', a.shape)
        return a
