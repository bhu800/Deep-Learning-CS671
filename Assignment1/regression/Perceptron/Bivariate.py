# =============================================================================
# Libraries
# =============================================================================


from mpl_toolkits import mplot3d


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================
# Functions
# =============================================================================

# ---- predict/find class ------------------??
# low --> class_0, high --> class_1
def predict(weights,x):
    a = np.matmul(weights,x)
    
    return a


# ----------- activation function -----------
# Logistic function
def activation_function(a):
    # slope parameter
    beta = 1
    
    return 1/(1 + np.exp(-beta*a))
    

# derivative of activation function wrt weighted input to neuron
# Logistic function derivative
def activation_function_derivative(a):
    # slope parameter
    beta = 1
    
    s=activation_function(a)
    return beta*(s)*(1-s)


# ----------- error -----------------
# average error 
def error(data,W):
    

    total_error=0
    for i in range(data.shape[0]):
            x=data.iloc[i]
            a=np.matmul(W,[x[i] for i in range(0,input_dimension+1)])
            total_error+= (x[input_dimension+1]-a)*(x[input_dimension+1]-a)
    total_error/=2.0*len(data)
    return total_error


# -------- learn perceptron weights -------
def perceptron(data,weights):
    # error bw 2 successive epochs
    err_limit = 1e-3
    error_array=[]
    
    # learning rate
    eta = 0.0001
    
    # epochs
    err_last = 1e3
    
    while(1):
        
        # weight-update-rule
        for i in range(len(data)):
            x=data.iloc[i]
            a=np.matmul(weights,[x[i] for i in range(0,input_dimension+1)])
            
            weights = weights + eta*(x[input_dimension+1]-a)*(np.array([x[i] for i in range(0,input_dimension+1)]))
        
        
        
        
        err=error(data,weights)
        error_array.append(err)
        if (abs(err - err_last) < err_limit):
            break
       # print(err)
        err_last = err
    plt.plot(error_array)
    plt.show()
    return weights    


# ------Initilize and find weights of perceptrons ------
# w[i][j] --> weights,bias of perceptron for diff b/w Class-i & Class-j
# w[i][j][-1] --> bias
def perceptron_weights(train):
    # initialize
    W=np.random.randint(100, size=(input_dimension+1))
    
    # find weights of perceptrons
    # call like this --> perceptron(classi, classj)  ==> label(classi) < label(classj)
    
    W=perceptron(train,W)
    return W


# -------- Find class labels for test-data -------
def predict_test_data_class(test,W):
    mat=[test[i] for i in range(0,input_dimension+1)]
    
    return np.matmul(W,mat)


# ----- split data into train-validate-test ----
# train : validate : test :: 6:2:2
def split_data(data):
    
    
    # -----divide into train-validation-test-------
    # train[i],validate[i] --> data of class-i
    train,validate=[],[]
    # (test[i],actual_test_class[i]) --> (data, correct label)
    test,actual_test_class=[],np.array([])
    
    
    length=data.shape[0]
        
    train_classi,validate_classi,test_classi=np.split(data,[int(.6*length),int(.8*length)])
    train.append(train_classi);validate.append(validate_classi),test.append(test_classi)
        
    actual_test_class=np.append(actual_test_class,test_classi[input_dimension+1])
    
    
    return train[0],validate[0],test[0],actual_test_class


# ------ read data from file --------------
def get_data(input_path):
    data = pd.read_csv(input_path,header = None)
    return data


# =============================================================================
# main
# =============================================================================

#IMP:-
# classes --> 0-based indexing
# index in array = label of class
# 1-vs-1 classification model used

#----- input specs -----

input_dimension=2

input_path=r"../../Group21/Regression/BivariateData/21.csv"

# ----read data from file -----
# data[i] --> data of class with label=i
data=get_data(input_path)
add_col = np.ones(len(data))
data[input_dimension+1]=data[input_dimension]
data[input_dimension]=add_col



# split data --> train,validate,test
train,validate,test,actual_test_class=split_data(data)


# perceptron weights
W=perceptron_weights(train)

# predict class label
predicted=predict_test_data_class(test,W)
'''
plt.scatter(test[0],test[2])
plt.scatter(test[0],predicted)

'''
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(test[0], test[1], predicted, c=predicted,cmap='prism',label='Predicted');
ax.scatter3D(test[0], test[1], test[3], c=test[3], cmap='Greens',label='Original');

plt.title('Test Data')
plt.legend()
plt.show()
plt.scatter(predicted,test[3])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Data')
plt.legend()
plt.show()
predicted=predict_test_data_class(validate,W)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(validate[0], validate[1], predicted, c=predicted, cmap='prism',label='Predicted');
ax.scatter3D(validate[0], validate[1], validate[3], c=validate[3], cmap='Greens',label='Original');

plt.title('Validate Data')
plt.legend()
plt.show()
plt.scatter(predicted,validate[3])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Validate Data')
plt.legend()
plt.show()
predicted=predict_test_data_class(train,W)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(train[0], train[1], predicted, c=predicted,cmap='prism',label='Predicted');
ax.scatter3D(train[0], train[1], train[3], c=train[3], cmap='Greens',label='Original');

plt.title('Train Data')
plt.legend()
plt.show()
plt.scatter(predicted,train[3])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Train Data')
plt.legend()
plt.show()