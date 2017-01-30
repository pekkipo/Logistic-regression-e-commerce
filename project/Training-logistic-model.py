import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
#http://scikit-learn.org/stable/
from preprocessing import get_binary_data
# the way to get the data

X, Y = get_binary_data()
X, Y = shuffle(X, Y)  # in case it was in order

# Create test sets
X_train = X[:-100]  # all but last 100 -> 298x8
Y_train = Y[:-100]
X_test = X[-100:]  # withhold 100 samples to be our test set -> 100x8
Y_test = Y[-100:]
# X 398x8
# Y 398x8

# print(Y)
# print(Y_test)
#
# print(X.shape)
# print(Y.shape)
# print(X_train.shape)
# print(X_test.shape)

D = X.shape[1]  # number of columns
W = np.random.randn(D)
b = 0

# More about those functions is in Logistic-prediction.py
def sigmoid(a):
    return 1 / (1+np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


# Function to determine classification rate
def classification_rate(Y, P):
    return np.mean(Y == P)  # returns ones and zeroes. Divides number of correct by the total number

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))  # formula is in the notes

train_costs = []
test_costs = []
learning_rate = 0.001

for i in range(10000):
    pY_train = forward(X_train, W, b)
    pY_test = forward(X_test, W, b)
    c_train = cross_entropy(Y_train, pY_train)
    c_test = cross_entropy(Y_test, pY_test)  # test cost

    # append to the list of costs
    train_costs.append(c_train)
    test_costs.append(c_test)

    # Gradient descent
    W -= learning_rate*X_train.T.dot(pY_train - Y_train)  # Y-train - targets
    b -= learning_rate*(pY_train - Y_train).sum()
    if i % 1000 == 0:
        print(i, c_train, c_test)

print("Final classification rate: ", classification_rate(Y_train, np.round(pY_train)))
print("Final test classification rate: ", classification_rate(Y_test, np.round(pY_test)))

# Plot the costs
legend1, = plt.plot(train_costs, label = 'train costs')
legend2, = plt.plot(test_costs, label = 'test costs')
plt.legend([legend1, legend2])
plt.show()
