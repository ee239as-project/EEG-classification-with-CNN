import pandas as pd
import numpy as np
import scipy
import scipy.signal

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
    print('hi')
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)

    return standardized
