import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)

# This time we will have labels
X[:50, :] = X[:50, :] - 2*np.ones((50, D))  # first 50 points
# -2*... centered at x =-2, y = -2
# the same goes for the other class. But center at +2;+2
X[50:, :] = X[50:, :] - 2*np.ones((50, D))

# Create an array of targets
T = np.array([0]*50 + [1]*50)  # first 50 zeros, second 50 ones

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize weights
w = np.random.randn(D+1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))  # formula for sigmoid from the notes

Y = sigmoid(z)

# func to calculate Cross-entropy-error
# takes in targets and predicted output
def cross_entropy(T, Y):
    # based on the formula in the notes
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:  # target is zero
            E -= np.log(1-Y[i])
    return E

print(cross_entropy(T, Y))


w = np.array([0, 4, 4])
z = Xb.dot(w)  # output
Y = sigmoid(z)  #
print(cross_entropy(T, Y))
