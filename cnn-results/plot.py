
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def visualize(sgd, window = 30, total = 1, zeroes = False):
    if zeroes:
        sgd = sgd[sgd != 0]

    sgd, std_sgd = movingAverage(sgd, window)

    sgdx = np.linspace(0, 1, num = sgd.shape[0])

    plt.style.use('seaborn-colorblind')
    plt.figure()
    plt.plot(sgdx, sgd) # , label = 'GPOMDP')
    plt.legend()

    # plt.fill_between(sgdx, sgd - std_sgd, sgd + std_sgd, alpha = 0.10, linestyle = ":")


    # plt.title('Lunar Lander Reward')
    #plt.xlabel('Number of Trajectories')
    #plt.ylabel('Reward')

    plt.show()
    # plt.savefig("cartpole-reward-compare-new.jpg")


def movingAverage(data, window):
    rolling_mean = pd.Series(data).rolling(window).mean()
    std = pd.Series(data).rolling(window).std()
    return rolling_mean, std




if __name__ == '__main__':
    data = np.loadtxt("cnn-loss-data-end.csv")
    data_start = np.loadtxt("cnn-loss-data-start.csv")


    visualize( data, window = 20, total = data.shape[0], zeroes = True )
    visualize( data_start , window = 20, total = data.shape[0], zeroes = True )
