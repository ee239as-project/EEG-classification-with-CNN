# Fix for issue loading Utils.preprocess_util
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
from Utils.preprocess_util import *

def get_accuracy(ouput, target, batch_size):
    # get accuracy for training batch
    classes_predicted = torch.max(ouput, 1)[1]
    correct = (np.equal(classes_predicted.tolist(), target.tolist()).astype(int)).sum()
    # correct = (max_values[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * correct / batch_size
    return accuracy.item()

def get_accuracy_in_batches(model, X, Y, seq_dim, input_dim, dataset='', iterations=20, cnn=False):
    dataset_size = X.shape[0]
    batch_size = int(dataset_size / iterations)
    print('Dataset:', dataset_size)
    print('Batch size:', batch_size)

    start = 0
    valid_accuracies = []
    for i in range(iterations):
        end = start + batch_size

        if cnn:
            y_pred = model(threeD_to_fourDTensor(X[start:end]).float())
        else:
            X_tensor = torch.from_numpy(X[start:end].reshape(-1, seq_dim, input_dim))
            print(X_tensor.shape)
            y_pred = model(X_tensor.float())

        y_true = Y[start:end]
        val_acc = get_accuracy(y_pred, y_true, batch_size=len(y_true))

        start = end
        valid_accuracies.append(val_acc)
    print(dataset, ' accuracy:', np.mean(valid_accuracies))
