import numpy as np
import torch

def get_accuracy(ouput, target, batch_size):
    ''' Obtain accuracy for training round '''
    classes_predicted = torch.max(ouput, 1)[1]
    corrects = (np.equal(classes_predicted.tolist(),target.tolist()).astype(int)).sum()
    # corrects = (max_values[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
