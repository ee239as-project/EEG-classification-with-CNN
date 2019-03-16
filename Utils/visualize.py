# Fix for issue loading Utils.preprocess_util
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from Utils.preprocess_util import *
from Utils.cnn_helpers import *

# ------------------------------------ ACCURACY ------------------------------------
def get_accuracy(predictions, target, testing=True):
    target = target.tolist()

    # during testing, the mean of the 125 cropped trial predictions is used as the
    # final prediction value for each cropped trial
    if testing:
        mean_preds = torch.mean(predictions, 0)
        pred = int(torch.argmax(mean_preds))
        return pred == target[0]

    # get accuracy for training batch
    preds = torch.max(predictions, 1)[1].tolist()
    n_samples = len(preds)

    correct = np.sum(np.equal(preds, target))
    return 100 * correct / n_samples

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
        acc = get_accuracy(y_pred, y_true, testing)
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
def perform_plotting(loss_history, train_acc_history, val_acc_history):
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

    # Training loss over iterations
    plt.subplot(2, 1, 1)
    plt.plot(loss_history, ',')
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

def evaluate_model():
    # Load loss and accuracy history
    f = '../Data/cnn_epoch_19.pkl' # change to evaluate other epoch
    file = open(f,'rb')
    epoch = pickle.load(file)
    file.close()

    # Perform plotting
    perform_plotting(epoch['loss_history'], epoch['train_acc_history'], epoch['val_acc_history'])

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
    # X_valid = np.load(X_valid_f)
    # Y_valid = np.load(y_valid_f)

    # print('Evaluating model...')
    # valid_acc = check_accuracy(model, X_valid, Y_valid, subsample=False)
    # print('Validation accuracy:', valid_acc)
    # del X_valid, Y_valid

    X_test = np.load(X_test_f)
    Y_test = np.load(y_test_f)
    test_acc = check_accuracy(model, X_test, Y_test, subsample=False, testing=True)
    print('Testing accuracy:', test_acc)
    del X_test, Y_test

if __name__ == '__main__':
    evaluate_model()
