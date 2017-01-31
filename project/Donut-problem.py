
import numpy as np
import matplotlib.pyplot as plt


N = 1000
D = 2

# 2 radiuses
R_inner = 5
R_outer = 10

# distance from origin is radius + random normal
# angle theta is uniformly distributed between (0, 2pi)
R1 = np.random.randn(N//2) + R_inner  # for half the data. spread around by 5
theta = 2*np.pi*np.random.random(N//2)  # polar coordinates uniformly distributed
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T  # convert the polar coordinates into xy coordinates

# The same for the outer radius
R2 = np.random.randn(N//2) + R_outer  # normally distributed variable surrounding outer
theta = 2*np.pi*np.random.random(N//2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

# Will be two ellipses with radiuses of 5 and 10

# Calculate entire X by concatating these two above
X = np.concatenate([ X_inner, X_outer ])
# targets are
T = np.array([0]*(N//2) + [1]*(N//2)) # labels: first 50 are 0, last 50 are 1

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()



# add a column of ones
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))

# add a column of r = sqrt(x^2 + y^2)
r = np.zeros((N,1))
for i in range(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,]))
Xb = np.concatenate((ones, r, X), axis=1)
# add another column

# randomly initialize the weights
w = np.random.randn(D + 2)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    # E = 0
    # for i in xrange(N):
    #     if T[i] == 1:
    #         E -= np.log(Y[i])
    #     else:
    #         E -= np.log(1 - Y[i])
    # return E
    return (T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


# let's do gradient descent 100 times
learning_rate = 0.0001
error = []
for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 100 == 0:
        print(e)

    # gradient descent weight update with regularization
    # w += learning_rate * ( np.dot((T - Y).T, Xb) - 0.01*w ) # old
    w += learning_rate * ( Xb.T.dot(T - Y) - 0.1*w )

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy per iteration")
plt.show()

print("Final w:", w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)
