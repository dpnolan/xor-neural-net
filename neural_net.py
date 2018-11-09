import numpy as np

np.random.seed(0)

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters():
    W1 = np.random.rand(2, 2)
    W2 = np.random.rand(1, 2)
    b1 = np.zeros((1, 1))
    b2 = np.zeros((1, 1))
    return {
        "W1": W1,
        "W2": W2,
        "b1" : b1,
        "b2" : b2
    }

def forward_prop():
    z1 = np.dot(W1, X) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    cache = {
        "a1": a1,
        "a2": a2
    }
    return a2, cache

def calculate_cost(yhat, Y):
    return -(np.dot(Y, np.log(yhat).T) + np.dot(1-Y, np.log(1-yhat).T))/4

# def backward_prop(Y, cache, parameters):


parameters = initialize_parameters()
W1 = parameters["W1"]
W2 = parameters["W2"]
b1 = parameters["b1"]
b2 = parameters["b2"]
a2, cache = forward_prop()

print(calculate_cost(a2, Y))