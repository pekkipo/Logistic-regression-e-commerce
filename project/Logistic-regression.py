import numpy as np

# Generate some data
N = 100
D = 2

X = np.random.randn(N, D)  # normally distributed matrix

# Need the bias term
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

# Weights
w = np.random.randn(D + 1)

z = Xb.dot(w)

# func that performs the sigmoid operation
def sigmoid(z):
    return 1/(1 + np.exp(-z))  # formula for sigmoid from the notes

print(sigmoid(z))