import numpy as np
import matplotlib.pyplot as plt


# XOR - is a logic gate

N = 4
D = 2

# 4 data points that represent different combinations of true and false
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

# manually set targets of my choice
T = np.array([0, 1, 1, 0])

# add a column of ones
ones = np.array([[1]*N]).T

# Plot it. four points at 4 places. Cannot really find one line separating them
plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()

# We will add another dimension and make 2D problem a 3D problem
# That will make the data linearly separable
xy = np.matrix(X[:, 0] * X[:, 1]).T
Xb = np.array(np.concatenate((ones, xy, X), axis=1))

# randomly initialize the weights
w = np.random.randn(D + 2)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


# let's do gradient descent 100 times
learning_rate = 0.001
error = []
for i in range(10000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 100 == 0:
        print(e)

    # gradient descent weight update with regularization
    # w += learning_rate * ( np.dot((T - Y).T, Xb) - 0.01*w )
    w += learning_rate * ( Xb.T.dot(T - Y) - 0.01*w )
    # 0.01 regularization term

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy per iteration")
plt.show()

print("Final w:", w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)

