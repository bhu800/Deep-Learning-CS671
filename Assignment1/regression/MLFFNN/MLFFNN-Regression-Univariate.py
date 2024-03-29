# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %% [markdown]
# $z^{x, l}= w^{l}a^{x, l-1} + b^{l}$ and $a^{x, l} = \sigma(z^{x, l})$
# %%
# -- data storage --
avg_error = []
model_output=[]
model_output2=[]

# %%
class MLFFNN:

    # constructor to initialize instance of class
    def __init__(self, layers_size = [2, 6, 1]):
        self.layers_size = layers_size
        self.biases = []
        self.weights = []
        self.A = []
        global avg_error
        avg_error = []

        # initialize weights and biases with random values
        for n_neurons in self.layers_size[1:]:
            self.biases.append(np.random.randn(n_neurons, 1))
        for n1_neurons, n2_neurons in zip(self.layers_size[:-1], self.layers_size[1:]):
            self.weights.append(np.random.randn(n2_neurons, n1_neurons))

    def feedForward(self, x, return_final_only=False):
        a = x.reshape(-1, 1) # activation vector of a single layer, in this case input layer 
        self.A = [x.reshape(-1, 1)] # list for activation vectors for every layer

        Z = [] # list for weighted input vectors for every layer

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            # print("--> ", w.shape, a.shape, b.shape)
            z = np.matmul(w, a) + b # z_l = w_l * a_l-1 + b_l
            Z.append(z)
            a = self.sigmoid_activation(z)
            # print("z = ", z.shape)
            # print("a = ", a.shape)
            self.A.append(a)
            
        # output layer
        z = np.matmul(self.weights[-1], a) + self.biases[-1] # z_l = w_l * a_l-1 + b_l
        Z.append(z)
        a = self.linear_activation(z)
        self.A.append(a)

        if (return_final_only):
            return a
        else:
            return (self.A, Z)

    def backPropagation(self, A, Z, y):
        # print("*** Debug***")
        # print(A[0].shape, Z[0].shape)
        # print("*********")
        del_C_by_del_b = [np.zeros(b.shape) for b in self.biases]
        del_C_by_del_w = [np.zeros(w.shape) for w in self.weights]
        # print("debug--> ", A[-1].shape, y.shape, self.linear_derivative(Z[-1]).shape)
        del_ = (A[-1] - y.reshape(-1, 1)) * self.linear_derivative(Z[-1]) # del_ = del_C_by_del_z
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
        global avg_error
        avg_error = []
        
        for e in range(epochs):
            self.gradientDescent(train_X, train_Y, eta)
            train_err=self.train_error(train_X,train_Y)
            test_error = self.test(test_X, test_Y)
            print(f"=== Epoch {e+1}/{epochs} - train error = {train_err}, test error = {test_error} ===\n")
            
            # model output
            global model_output
            global model_output2
            model_output = np.apply_along_axis(self.predictValue, axis=1, arr=train_X)
            model_output2 = np.apply_along_axis(self.predictValue, axis=1, arr=test_X)
            if(e%1000==999):
                self.midgraphs(test_data[:, :-1])
    def predictValue(self, x):
        return self.feedForward(x, return_final_only=True)

    def train_error(self, X, Y):
        n = X.shape[0]
        pred = np.apply_along_axis(self.predictValue, axis=1, arr=X)
        
        pred=pred.reshape(len(pred))
        error=np.sqrt((((pred-Y)**2).sum())/(2*n))
        
        return error
    
    def test(self, X, Y):
        n = X.shape[0]
        # Y = Y.reshape(-1)
        pred = np.apply_along_axis(self.predictValue, axis=1, arr=X)
        '''pred = []
        for i in range(0,n):
            x=net.predictValue(X[i])
            pred.append(x)'''
#        Y = np.matmul(Y, np.arange(Y.shape[-1]))
        # print("Debug")
        # print(pred.shape, Y.shape)
        # print(pred == Y)
#        print(pred,Y)
        #pred=np.array(pred)
        
        coll = 0
        
        for i in range(len(pred)):
            coll = coll + (Y[i]-pred[i])**2
        
        error = np.sqrt(coll.sum()/(2*n))
        
#        pred=pred.reshape(len(pred))
#        error=np.sqrt((((pred-Y)**2).sum())/(2*n))
#        
#        global avg_error
#        avg_error.append(error*error)

        return error
    
    def midgraphs(self,X):
        layer1_val=[]
        for i in range(0,len(X)):
            a=self.feedForward(X[i],return_final_only=True)
            layer1_val.append(self.A[1][3])
        xxx = [X[i][0] for i in range(0,len(X))]
        plt.scatter(xxx,layer1_val)
        plt.title('Hidden Layer node 4')
        plt.show()
        layer1_val=[]
        for i in range(0,len(X)):
            a=self.feedForward(X[i],return_final_only=True)
            layer1_val.append(self.A[1][4])
        xxx = [X[i][0] for i in range(0,len(X))]
        plt.scatter(xxx,layer1_val)
        plt.title('Hidden Layer node 5')
        plt.show()
        layer1_val=[]
        for i in range(0,len(X)):
            a=self.feedForward(X[i],return_final_only=True)
            layer1_val.append(self.A[1][5])
        xxx = [X[i][0] for i in range(0,len(X))]
        plt.scatter(xxx,layer1_val)
        plt.title('Hidden Layer node 6')
        plt.show()
        o_layer=[]
        for i in range(0,len(X)):
            a=self.feedForward(X[i],return_final_only=True)
            o_layer.append(self.A[2][0])
        xxx = [X[i][0] for i in range(0,len(X))]
        plt.scatter(xxx,o_layer)
        plt.title('Output layer node ')
        plt.show()
    def sigmoid_activation(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_derivative(self, z):
        return np.exp(-z)/((1.0+np.exp(-z))**2)

    def linear_activation(self, z):
        return z

    def linear_derivative(self, z):
        return 1
    
    def relu_activation(self, z):
        z[z<0]=0
        return z

    def relu_derivative(self, z):
        return z>=0


# %%
# ------ read data from file --------------
def get_data(input_path):
    data = pd.read_csv(input_path,header = None)
    return data

# %%
#----- input specs -----
input_dimension=1
input_path=r"../../Group21/Regression/UnivariateData/21.csv"

# ----read data from file -----
data=get_data(input_path)
add_col = np.ones(len(data))
data[input_dimension+1]=data[input_dimension]
data[input_dimension]=add_col

data=np.array(data)

# %%
np.random.shuffle(data)
train_data = data[:int(.6*data.shape[0]), :]
validation_data = data[int(.6*data.shape[0]):int(.8*data.shape[0]), :]
test_data = data[int(.8*data.shape[0]):, :]
# %%
net = MLFFNN()
net.train(train_data[:, :-1], train_data[:, -1], validation_data[:, :-1], validation_data[:, -1], epochs=5000, eta=1)
#net.train(train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1], epochs=100, eta=0.1)

# average error
#plt.scatter(np.arange(1,101,1),avg_error)
#plt.title("Epoch vs AvgError (Regression-Univariate-MLFFNN)")
#plt.xlabel("Epochs")
#plt.ylabel("Average error")
#plt.plot()

# plot output
plt.scatter(train_data[:, 0], train_data[:, -1],c='blue',label='target')
plt.scatter(train_data[:, 0], model_output,c='orange',label='model')
plt.title('Train data plot')
plt.xlabel('input')
plt.ylabel('output')
plt.legend()
plt.show()

plt.scatter(validation_data[:, 0], validation_data[:, -1],c='blue',label='target')
plt.scatter(validation_data[:, 0], model_output2,c='orange',label='model')
plt.title('Test data plot')
plt.xlabel('input')
plt.ylabel('output')
plt.legend()
plt.show()

# target vs model output
plt.scatter(train_data[:, -1], model_output)
plt.title("Train - Output comparison")
plt.xlabel("Target output")
plt.ylabel("Model output")
plt.plot()
plt.show()

plt.scatter(validation_data[:, -1], model_output2)
plt.title("Test - Output comparison")
plt.xlabel("Target output")
plt.ylabel("Model output")
plt.plot()
plt.show()

# %%


