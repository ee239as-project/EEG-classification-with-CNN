import pandas as pd
import numpy as np
import scipy
import scipy.signal
from sklearn.model_selection import train_test_split

def exponential_running_demean(data, factor=0.001):
    """
    computes exponential moving mean  for each channel given by the formula in https://arxiv.org/abs/1703.05051
    data: 2darray (time, channels)
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    
    return demeaned
def exponential_running_standardize(data, factor_new=0.001,
                                     eps=1e-4):
    '''
    Perform exponential running standardization.

    Compute the exponental running mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential running variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

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

def load_preprocess_eeg_data():
    
    X_test = np.load("../Data/X_test.npy")[:,0:22,:]
    y_test = np.load("../Data/y_test.npy")
    person_train_valid = np.load("../Data/person_train_valid.npy")
    X_train_valid = np.load("../Data/X_train_valid.npy")[:,0:22,:]
    y_train_valid = np.load("../Data/y_train_valid.npy")
    person_test = np.load("../Data/person_test.npy")
    
    X_train, X_valid, y_train, y_valid = train_test_split(  X_train_valid, y_train_valid, test_size=0.33,                    random_state=42)
    
    #standardize every data point
    X_train_modified =[]
    X_valid_modified =[]
    X_test_modified =[]

    for xi in X_train:
        X_train_modified.append(exponential_running_standardize(xi.T, eps=1e-4))

    for xi in X_valid:
        X_valid_modified.append(exponential_running_standardize(xi.T, eps=1e-4))

    for xi in X_test:
        X_test_modified.append(exponential_running_standardize(xi.T, eps=1e-4))


    X_train = np.array(X_train_modified)
    X_valid = np.array(X_valid_modified)
    X_test = np.array(X_test_modified)
    
    print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
    print ('Test data shape: {}'.format(X_test.shape))
    print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
    print ('Test target shape: {}'.format(y_test.shape))
    print ('Person train/valid shape: {}'.format(person_train_valid.shape))
    print ('Person test shape: {}'.format(person_test.shape))
    print()
    
    #remmoving eog sginals
    X_train = np.transpose(X_train,[0,2,1])
    X_valid = np.transpose(X_valid,[0,2,1])
    X_test = np.transpose(X_test,[0,2,1])
    print ('Training data shape: {}'.format(X_train.shape))
    print ('Valid data shape: {}'.format(X_valid.shape))
    print ('Training target shape: {}'.format(y_train.shape))
    print ('Valid target shape: {}'.format(y_valid.shape))
    
    #converting labels to range 0-No. of classes
    Y_train = []
    for y in y_train:
        value = np.abs(769-y)
        Y_train.append(value)
    Y_train = np.array(Y_train)

    Y_valid = []
    for y in y_valid:
        value = np.abs(769-y)
        Y_valid.append(value)
    Y_valid = np.array(Y_valid)

    Y_test = []
    for y in y_test:
        value = np.abs(769-y)
        Y_test.append(value)
    Y_test = np.array(Y_test)
    
    return X_train,X_valid,X_test,Y_train,Y_valid,Y_test