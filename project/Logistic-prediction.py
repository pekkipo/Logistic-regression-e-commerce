import numpy as np
from preprocessing import get_binary_data

X, Y = get_binary_data()

D = X.shape[1]  # gives 8 - number of dimensions, i.e. features (number of columns)
print(D)
# initialize weight
W = np.random.randn(D)
# Weight for each feature (dimension)
print(W)
b = 0  # bias term. 0 means that we have equal number of samples of both classes

def sigmoid(a):
    return 1 / (1+np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)
print(P_Y_given_X)
predictions = np.round(P_Y_given_X)  # if > 0.5 makes it 1, else makes it zero
print(predictions)
# so basically we made a classification based on whether the value is larger or less than 0.5, i.e. whether closer to zero or to one

# Function to determine classification rate
def classification_rate(Y, P):
    return np.mean(Y == P)  # returns ones and zeroes. Divides number of correct by the total number

print("Score:", classification_rate(Y, predictions))
