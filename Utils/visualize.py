import numpy as np
import torch


def get_accuracy(ouput, target, batch_size):
    # get accuracy for training batch
    classes_predicted = torch.max(ouput, 1)[1]
    correct = (np.equal(classes_predicted.tolist(), target.tolist()).astype(int)).sum()
    # correct = (max_values[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * correct / batch_size
    return accuracy.item()
