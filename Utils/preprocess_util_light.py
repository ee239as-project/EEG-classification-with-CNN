import pandas as pd
import numpy as np
import os.path
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

def exponential_running_demean(data,factor_new=0.001):
    """
    computes exponential moving mean for each channel given by the formula in
    https://arxiv.org/abs/1703.05051
    data: 2darray (time, channels)
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)

    return demeaned

def exponential_running_standardize(data, factor_new=0.001, eps=1e-4):
    '''
    Perform exponential running standardization.

    Compute the exponental running mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot
    m_{t-1}`.

    Then, compute exponential running variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew})
    \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.

    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    eps: float
        Stabilizer for division by zero variance.
    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    '''
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)

    return standardized

def threeD_to_fourDTensor(X):
    return Variable(torch.tensor(X.reshape((X.shape[0],1,X.shape[1],X.shape[2],))))

'''
Crops 4 sec time interval into 2 second chunks at intervals .016 seconds apart
(250 Hz sampling corresponds to sample intervals of .004 seconds), creating 125
data points from 1 given data point, i.e. 1 time step (lends 125x increase in
training data)

Note:
During testing, the mean of these 125 predictions is used as the final prediction
for the trial.
During training, we compute a loss for each prediction. Therefore, cropped training
increases our training set size by a factor of 125, albeit with highly correlated
training examples.
Since our crops are smaller than the trials, the ConvNet input size will be smaller
(from about 1000 to 500 input samples for this 250 Hz sampling rate), while all other
hyperparameters stay the same. (That is, the new timestep dimension is size 500
instead of 1000, or 2 as opposed to 4 seconds, so the ConvNet input size will be
half as large.)
'''
def crop_per_timestep(X, y, X_file, y_file):
    Hz = 250
    interval_sec = 2
    stride = 4
    timesteps = 1000
    n_crops = int((timesteps/interval_sec) / stride)
    if len(X.shape) is 2: # X, y: person_train_valid, person_test
        X = np.array([np.repeat(label, n_crops) for label in X]).flatten()
        y = np.array([np.repeat(label, n_crops) for label in y]).flatten()
        np.save(X_file, X)
        np.save(y_file, y)
        del X, y
        return

    trials, electrodes, _ = X.shape
    new_trials = trials * n_crops
    crop_size = Hz * interval_sec
    X_new = np.zeros((new_trials, 22, crop_size))

    X_new_trial_i = 0
    for trial_i in range(trials):
        print('Cropping trial ', trial_i, ' out of ', trials)
        X_new_trial_i_checkpoint = X_new_trial_i
        for electrode_i in range(electrodes):
            '''
            To understand X_new_trial_i:
            1) Assume we are at X_new_trial_i = 0.
            2) After iterating through an electrode and setting 125 different X_new
            trial indices (0,124) for an electrode index to a cropped_time (500,)
            (i.e. X_new[X_new_trial_i, electrode_i] = cropped_time), we have to
            iterate through those same 125 X_new trial indices again for each
            electrode to set a cropped_time (500,) for the 2nd to 22nd electrode
            indices.
            3) After iterating through an entire old trial_i in X (i.e. all 22
            electrodes), X_new_trial_i is set to 125 and the function continues.
            It ends when the last X_new_trial_i (X.shape[0] * 125) has a
            cropped_time (500,) set for its last 22nd electrode.
            '''
            X_new_trial_i = X_new_trial_i_checkpoint

            timestep_i = 0
            for j in range(n_crops):
                cropped_time = X[trial_i,electrode_i,timestep_i:timestep_i+crop_size]

                X_new[X_new_trial_i, electrode_i] = cropped_time
                X_new_trial_i += 1
                timestep_i += stride
    # save cropped X and y and delete variables to manage memory
    np.save(X_file, X_new)
    del X, X_new

    y_new = np.array([np.repeat(label, n_crops) for label in y]).flatten()
    np.save(y_file, y_new)
    del y, y_new

def crop_trials(X_train, y_train, X_valid, y_valid, X_test, y_test,
                person_tr, person_test):
    X_train_f = '../Data/X_train_c.npy'
    y_train_f = '../Data/y_train_c.npy'
    X_valid_f = '../Data/X_valid_c.npy'
    y_valid_f = '../Data/y_valid_c.npy'
    X_test_f = '../Data/X_test_c.npy'
    y_test_f = '../Data/y_test_c.npy'
    person_tr_f = '../Data/person_train_valid_c.npy'
    person_test_f = '../Data/person_test_c.npy'

    # training data
    if not os.path.exists(X_train_f) or not os.path.exists(y_train_f):
        crop_per_timestep(X_train, y_train, X_train_f, y_train_f)

    # validation data
    if not os.path.exists(X_valid_f) or not os.path.exists(y_valid_f):
        crop_per_timestep(X_valid, y_valid, X_valid_f, y_valid_f)

    # test data
    if not os.path.exists(X_test_f) or not os.path.exists(y_test_f):
        crop_per_timestep(X_test, y_test, X_test_f, y_test_f)

    # person training, validation, and test data
    if not os.path.exists(person_tr_f) or not os.path.exists(person_test_f):
        crop_per_timestep(person_tr, person_test, person_tr_f, person_test_f)

    # load all data
    X_train = np.load(X_train_f)
    y_train = np.load(y_train_f)
    X_valid = np.load(X_valid_f)
    y_valid = np.load(y_valid_f)
    X_test = np.load(X_test_f)
    y_test = np.load(y_test_f)
    person_tr = np.load(person_tr_f)
    person_test = np.load(person_test_f)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, person_tr, person_test

def load_preprocess_eeg_data():
    # only use first 22 electrodes which are EEG not EOG
    X_train_valid = np.load('../Data/X_train_valid.npy')[:,0:22,:]
    y_train_valid = np.load('../Data/y_train_valid.npy')
    X_test = np.load('../Data/X_test.npy')[:,0:22,:]
    y_test = np.load('../Data/y_test.npy')
    person_test = np.load('../Data/person_test.npy')
    person_train_valid = np.load('../Data/person_train_valid.npy')

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,
                                                          y_train_valid,
                                                          test_size=0.33,
                                                          random_state=42)

    # electrode-wise exponenential moving standardization of the continuous data
    X_train_mod = []
    X_valid_mod = []
    X_test_mod = []
    for xi in X_train:
        X_train_mod.append(exponential_running_standardize(xi.T, eps=1e-4))
    for xi in X_valid:
        X_valid_mod.append(exponential_running_standardize(xi.T, eps=1e-4))
    for xi in X_test:
        X_test_mod.append(exponential_running_standardize(xi.T, eps=1e-4))
    del X_train_valid, y_train_valid, X_test
    X_train = np.array(X_train_mod)
    X_valid = np.array(X_valid_mod)
    X_test = np.array(X_test_mod)

    # get (trials, electrodes=22, timesteps=1000) form
    X_train = np.transpose(X_train,[0,2,1])
    X_valid = np.transpose(X_valid,[0,2,1])
    X_test = np.transpose(X_test,[0,2,1])

    '''
    Currently before cropping:
    Training data: (1417, 22, 1000)
    Training target: (1417,)
    Validation data: (698, 22, 1000)
    Validation target: (698,)
    Test data: (443, 22, 1000)
    Test target: (443,)
    Person train/validation: (2115, 1)
    Person test: (443, 1)
    '''
    # implement cropped training strategy (https://arxiv.org/pdf/1703.05051.pdf)
#    X_train, y_train, X_valid, y_valid, X_test, y_test, person_train_valid,
#    person_test = crop_trials(X_train, y_train, X_valid, y_valid, X_test, y_test,
#                              person_train_valid, person_test)
    '''
    After cropping:
    Training data: (177125, 22, 500)
    Training target: (177125,)
    Validation data: (87250, 22, 500)
    Validation target: (87250,)
    Test data: (55375, 22, 500)
    Test target: (55375,)
    Person train/validation: (264375,)
    Person test: (55375,)
    '''
    print('After cropping:')
    print('Training data: {}'.format(X_train.shape))
    print('Training target: {}'.format(y_train.shape))
    print('Validation data: {}'.format(X_valid.shape))
    print('Validation target: {}'.format(y_valid.shape))
    print('Test data: {}'.format(X_test.shape))
    print('Test target: {}'.format(y_test.shape))
    print('Person train/validation: {}'.format(person_train_valid.shape))
    print('Person test: {}\n'.format(person_test.shape))

    # converts labels to range 0-3 for number of classes
    Y_train = np.abs(769 - y_train)
    Y_valid = np.abs(769 - y_valid)
    Y_test = np.abs(769 - y_test)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


if __name__ == '__main__':
    load_preprocess_eeg_data()
