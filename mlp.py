import numpy as np
w1 =np.random.randn(features_number,5)
w2 =np.random.randn(5,1)
b1 =np.zeros((1,5))
b2 =np.zeros((1,1))


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float) 

def binary_cross_entropy(y_true,y_pred):
    # y * log(p) + (1 - y) * log(1 - p)
    pass

def binary_cross_entropy_derivative(y_true, y_pred_prob):
    return -(y_true / y_pred_prob) + ((1 - y_true) / (1 - y_pred_prob))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)
def forward(X):
    Z1 = np.dot(X,w1)+b1
    A1 = relu(Z1)
    Z2 = np.dot(A1,w2)+b2
    A2 = sigmoid(Z2)
    return A2

def backward(x):

    pass

def train(x):
    #epochs
    pass

m=100
dL_dA2=binary_cross_entropy_derivative(y_true,y_pred_prob)
dZ2=dL_dA2*sigmoid_derivative(Z2)

dW2 = np.dot(A1.T,dZ2)/m
db2 = np.sum(dZ2, axis=0, keepdims=True) / m 
dA1 = np.dot(dZ2, W2.T) 
dZ1 = dA1*relu_derivative(Z1)

dW1 = np.dot(X.T, dZ1) / m 
db1 = np.sum(dZ1, axis=0, keepdims=True) / m