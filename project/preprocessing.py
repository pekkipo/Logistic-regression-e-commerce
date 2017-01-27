import numpy as np
import pandas as pd

def get_data():
    data = pd.read_csv('data/ecommerce_data.csv')
    data = data.as_matrix()  # easier to work with

    X = data[:, :-1]   # everything aside from the last column
    # Y is the last column
    Y = data[:, -1]

    # Normalize X1
    X[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()  # std standard deviation
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    # Work with category column Time_of_day
    N, D = X.shape
    # making a new X with a new shape. 4 different categorical values (6-12, 12-6, 6-12, 12-6) divide 24 hours into 4 categories
    X2 = np.zeros((N, D+3))  # basically just adding three dimensions
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]
    # most of x is going to be the same

    # Do one-hot encoding for the other 4 columns
    for n in range(N):  # loop through every sample
        t = int(X[n, D-1])  # get the time of day for each sample
        # D-1 is a penultimate column as the last one is the output Y
        X2[n, t+D-1] = 1
        # making a certain column equal 1. While other columns of one-hot encoding will be zeros

    # print("X:",X)
    # print("Y:",Y)

    return X2, Y

    # Basically X is original data, X2 is a data with more dimensions because we implemented one-hot encoding

    # method 2 instead of a loop above
    # Z = np.zeros((N, 4))
    # Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # # assign: X2[:,-4:] = Z
    # assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)


def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]  # all the xs where y is less or equal to 1
    Y2 = Y[Y <= 1]
    # Y is also categories

    # print("X2:", X2)
    # print("Y2:", Y2)
    return X2, Y2

get_data()
get_binary_data()





