# Fix for issue loading Utils.preprocess_util
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import pickle
from Utils.preprocess_util import *


def get_accuracy(ouput, target, batch_size, testing=False):
    # get accuracy for training batch
    pred_classes = torch.max(ouput, 1)[1].tolist()

    # during testing, the mean of the 125 cropped trial predictions is used as the
    # final prediction value for each cropped trial
    if testing:
        pred_classes = np.repeat(np.mean(pred_classes), len(pred_classes))

    correct = (np.equal(pred_classes, target.tolist()).astype(int)).sum()
    accuracy = 100.0 * correct / batch_size
    return accuracy.item()

def check_accuracy(model, X, y, num_samples=None, batch_size=125, testing=False):
    # Subsample the data
    N = X.shape[0]
    N_subsample = N // 500 # subsample of data
    if num_samples > N_subsample:
        mask = np.random.choice(N, N_subsample)
        N = N_subsample
        X = X[mask]
        y = y[mask]

    # Compute predictions in batches
    n_batches = N // batch_size
    accuracies = []
    for i in range(n_batches):
        start = i * batch_size
        end = (i+1) * batch_size

        X_tensor = threeD_to_fourDTensor(X[start:end])
        y_pred = model(X_tensor.float())
        y_true = y[start:end]

        acc = get_accuracy(y_pred, y_true, len(y_true), testing)
        accuracies.append(acc)

    return np.mean(accuracies)

def save_checkpoint(epoch, loss_history, train_acc_history, val_acc_history,
                    best_val_acc, model='cnn'):
    checkpoint = {
      'epoch': epoch,
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
      'best_val_acc': best_val_acc,
    }
    filename = '../Data/%s_epoch_%d.pkl' % (model, epoch)
    print('Saving checkpoint to %s' % filename)
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
