import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

print(X)

# This time we will have labels
X[:50, :] = X[:50, :] - 2*np.ones((50, D))  # first 50 points
# -2*... centered at x =-2, y = -2
# the same goes for the other class. But center at +2;+2
X[50:, :] = X[50:, :] + 2*np.ones((50, D))

# Create an array of targets
T = np.array([0]*50 + [1]*50)  # first 50 zeros, second 50 ones

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)


def sigmoid(z):
    return 1/(1 + np.exp(-z))  # formula for sigmoid from the notes

# closed form solution
w = np.array([0, 4, 4])

# Draw points
plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)  # colors, size of the dots, transparency

# Draw a line
x_axis = np.linspace(-6, -6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
