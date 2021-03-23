# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import os

# %% [markdown]
# $z^{x, l}= w^{l}a^{x, l-1} + b^{l}$ and $a^{x, l} = \sigma(z^{x, l})$

# %%
class MLFFNN:

    # constructor to initialize instance of class
    def __init__(self, layers_size = [2, 6, 3]):
        self.layers_size = layers_size
        self.biases = []
        self.weights = []

        # initialize weights and biases with random values
        for n_neurons in self.layers_size[1:]:
            self.biases.append(np.random.randn(n_neurons, 1))
        for n1_neurons, n2_neurons in zip(self.layers_size[:-1], self.layers_size[1:]):
            self.weights.append(np.random.randn(n2_neurons, n1_neurons))

    def feedForward(self, x, return_final_only=False):
        a = x.reshape(-1, 1) # activation vector of a single layer, in this case input layer 
        A = [x.reshape(-1, 1)] # list for activation vectors for every layer

        Z = [] # list for weighted input vectors for every layer

        for b, w in zip(self.biases, self.weights):
            # print("--> ", w.shape, a.shape, b.shape)
            z = np.matmul(w, a) + b # z_l = w_l * a_l-1 + b_l
            Z.append(z)
            a = self.sigmoid_activation(z)
            # print("z = ", z.shape)
            # print("a = ", a.shape)
            A.append(a)

        if (return_final_only):
            return a
        else:
            return (A, Z)

    def backPropagation(self, A, Z, y):
        # print("*** Debug***")
        # print(A[0].shape, Z[0].shape)
        # print("*********")
        del_C_by_del_b = [np.zeros(b.shape) for b in self.biases]
        del_C_by_del_w = [np.zeros(w.shape) for w in self.weights]
        # print("debug--> ", A[-1].shape, y.shape, self.sigmoid_derivative(Z[-1]).shape)
        del_ = (A[-1] - y.reshape(-1, 1)) * self.sigmoid_derivative(Z[-1]) # del_ = del_C_by_del_z
        # print("debug - del_", del_.shape)
        del_C_by_del_b[-1] = del_
        del_C_by_del_w[-1] = np.matmul(del_, A[-2].T)

        for l in range(2, len(self.layers_size)):
            z = Z[-l]
            # print("*** Debug2***")
            # print(self.weights[-l+1].T.shape, del_.shape)
            # print("*********")
            del_ = np.matmul(self.weights[-l+1].T, del_) * self.sigmoid_derivative(z)
            del_C_by_del_b[-l] = del_
            # print("*** Debug3***")
            # print(del_.shape, A[-l-1].T.shape)
            # print("*********")
            del_C_by_del_w[-l] = np.matmul(del_, A[-l-1].T)

        return (del_C_by_del_b, del_C_by_del_w)


    def gradientDescent(self, X, Y, eta):
        n = X.shape[0]
        sum_delC_by_del_b = [np.zeros(b.shape) for b in self.biases]
        sum_delC_by_del_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(len(X)):
            # foward pass
            A, Z = self.feedForward(X[i])
            # backward pass
            del_C_by_del_b, del_C_by_del_w = self.backPropagation(A, Z, Y[i])
            # sum for calculating average 
            sum_delC_by_del_b = [s_dcb+dcb for s_dcb, dcb in zip(sum_delC_by_del_b, del_C_by_del_b)]
            sum_delC_by_del_w = [s_dcw+dcw for s_dcw, dcw in zip(sum_delC_by_del_w, del_C_by_del_w)]

        # update weights and biases
        self.weights = [w - (eta/n)*s_dcb for w, s_dcb in zip(self.weights, sum_delC_by_del_w)]
        self.biases = [b - (eta/n)*s_dcb for b, s_dcb in zip(self.biases, sum_delC_by_del_b)]

    def train(self, train_X, train_Y, test_X, test_Y, epochs=100, eta=0.01):

        for e in range(epochs):
            self.gradientDescent(train_X, train_Y, eta)
            test_accuracy = self.test(test_X, test_Y)
            print(f"=== Epoch {e+1}/{epochs} - test accuracy = {test_accuracy*100}% ===\n")

    def predictClass(self, x):
        return np.argmax(self.feedForward(x, return_final_only=True))

    def test(self, X, Y):
        n = X.shape[0]
        # Y = Y.reshape(-1)
        pred = np.apply_along_axis(self.predictClass, axis=1, arr=X)
        Y = np.matmul(Y, np.arange(Y.shape[-1]))
        # print("Debug")
        # print(pred.shape, Y.shape)
        # print(pred == Y)
        accuracy = (pred == Y).sum()/n

        return accuracy

    def sigmoid_activation(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_derivative(self, z):
        return np.exp(-z)/((1.0+np.exp(-z))**2)


        


# %%
c1 = np.loadtxt("./Group21/Classification/LS_Group21/Class1.txt")
c2 = np.loadtxt("./Group21/Classification/LS_Group21/Class2.txt")
c3 = np.loadtxt("./Group21/Classification/LS_Group21/Class3.txt")


# %%
c1 = np.append(c1, np.full((c1.shape[0], 3), [1, 0, 0]), axis=1)
c2 = np.append(c2, np.full((c2.shape[0], 3), [0, 1, 0]), axis=1)
c3 = np.append(c3, np.full((c3.shape[0], 3), [0, 0, 1]), axis=1)


# %%
net = MLFFNN()
net.train(train_data[:, :-3], train_data[:, 2:], test_data[:, :-3], test_data[:, 2:], epochs=100, eta=1)


# %%



