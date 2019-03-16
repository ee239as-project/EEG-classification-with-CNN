# Fix for issue loading Utils.preprocess_util
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import pickle
from Utils.preprocess_util import *
from Utils.cnn_helpers import *

# ------------------------------------ ACCURACY ------------------------------------
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

def check_accuracy(model, X, y, subsample=True, testing=False, batch_size=125):
    N = X.shape[0]
    if subsample:
        N_subsample = N // 500 # 0.2% subsample of data
        mask = np.random.choice(N, N_subsample)
        N = N_subsample
        X = X[mask]
        y = y[mask]

    # Compute predictions in batches
    n_batches = N // batch_size
    wtd_accuracies = []
    for i in range(n_batches):
        start = i * batch_size
        end = (i+1) * batch_size

        X_tensor = threeD_to_fourDTensor(X[start:end])
        y_pred = model(X_tensor.float())
        y_true = y[start:end]

        n_samples = len(y_true)
        acc = get_accuracy(y_pred, y_true, n_samples, testing)
        wtd_accuracies.append(acc * (n_samples / N))

    return np.sum(wtd_accuracies)

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

# ---------------------------------- MODEL EVALUATION ----------------------------------
def evaluate_model():
    # Load model
    f = '../Data/cnn_model_16_epochs.pkl' # change to evaluate other model
    file = open(f,'rb')
    model = pickle.load(file)
    file.close()

    # Load data
    print('Loading data')
    X_valid_f = '../Data/X_valid_c.npy'
    y_valid_f = '../Data/y_valid_c.npy'
    X_test_f = '../Data/X_test_c.npy'
    y_test_f = '../Data/y_test_c.npy'
    X_valid = np.load(X_valid_f)
    Y_valid = np.load(y_valid_f)

    print('Evaluating model...')
    valid_acc = check_accuracy(model, X_valid, Y_valid, subsample=False)
    print('Validation accuracy:', valid_acc)
    del X_valid, Y_valid

    X_test = np.load(X_test_f)
    Y_test = np.load(y_test_f)
    test_acc = check_accuracy(model, X_test, Y_test, subsample=False, testing=True)
    print('Testing accuracy:', test_acc)
    del X_test, Y_test

if __name__ == '__main__':
    evaluate_model()
