# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# Functions
# =============================================================================

# ---- predict/find class ------------------??
# low --> class_0, high --> class_1
def predict(weights,x):
    a = np.matmul(weights,x)
    
    return activation_function(a)


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
def error(class_0,class_1,W):
    l0=len(class_0)
    
    W=np.array([W])
    W_repeated=np.repeat(W,l0,axis=0)
    
    total_error=0
    
    # class_0
    total_error+=((0 - activation_function(np.einsum('ij,ij->i', W_repeated, class_0)))**2).sum()

    # class_1
    total_error+=((1 - activation_function(np.einsum('ij,ij->i', W_repeated, class_1)))**2).sum()

    total_error/=2.0
    return total_error


# -------- learn perceptron weights -------
def perceptron(class_0,class_1,weights):
    # error bw 2 successive epochs
    err_limit = 1e-3
    
    # learning rate
    eta = 0.1
    
    # epochs
    err_last = 1e9
    while(1):
        
        # weight-update-rule
        for i in range(class_0.shape[0]):
            x=class_0[i]
            a=np.matmul(weights,x)
            s=activation_function(a)
            derivative=activation_function_derivative(a)
            weights = weights + eta*(0-s)*derivative*x
        
        for i in range(class_1.shape[0]):
            x=class_1[i]
            a=np.matmul(weights,x)
            s=activation_function(a)
            derivative=activation_function_derivative(a)
            weights = weights + eta*(1-s)*derivative*x
        
        
        err=error(class_0,class_1,weights)

        if (abs(err - err_last) < err_limit):
            break
        
        err_last = err
        
    return weights    


# ------Initilize and find weights of perceptrons ------
# w[i][j] --> weights,bias of perceptron for diff b/w Class-i & Class-j
# w[i][j][-1] --> bias
def perceptron_weights(train,no_of_classes):
    # initialize
    W=np.ones((no_of_classes,no_of_classes,input_dimension+1))
    
    # find weights of perceptrons
    # call like this --> perceptron(classi, classj)  ==> label(classi) < label(classj)
    for i in range(no_of_classes):
        for j in range(no_of_classes):
            if(i<j):
                W[i,j]=perceptron(train[i],train[j],W[i,j])
    return W


# -------- Find class labels for test-data -------
def predict_test_data_class(test,W, no_of_classes):
    # logistic activation --> [0,1] --> 0.5
    decision_value=0.5
    
    # 1vs1 strategy --> predicted classes count for all pairs & test inputs --> argmax --> predicted class
    test_data_info=np.zeros((len(test),no_of_classes))
    for i in range(len(test)):
        for j in range(no_of_classes):
            for k in range(no_of_classes):
                if(j>=k):
                    continue
                
                value=predict(W[j,k],test[i])
                if(value > decision_value):
                    test_data_info[i][k]+=1
                else:
                    test_data_info[i][j]+=1
    
    print(test_data_info)
    # predicted class label
    predicted_test_class=np.argmax(test_data_info,axis=1)
    return predicted_test_class


# ----- split data into train-validate-test ----
# train : validate : test :: 6:2:2
def split_data(data,no_of_classes):
    # shuffle rows per class to get random ordering
    for x in data:
        np.random.shuffle(x)
    
    # -----divide into train-validation-test-------
    # train[i],validate[i] --> data of class-i
    train,validate=[],[]
    # (test[i],actual_test_class[i]) --> (data, correct label)
    test,actual_test_class=[],np.array([])
    
    for i in range(no_of_classes):
        length=data[i].shape[0]
        
        train_classi,validate_classi,test_classi=np.split(data[i],[int(.6*length),int(.8*length)])
        train.append(train_classi);validate.append(validate_classi),test.extend(test_classi)
        
        actual_test_class=np.append(actual_test_class,np.zeros(int(0.2*length))+i)
    
    train,validate,test=np.array(train),np.array(validate),np.array(test)
    return train,validate,test,actual_test_class


# ------ read data from file --------------
def get_data(input_path,no_of_classes):
    data=[]
    for i in range(no_of_classes):
        data_path=input_path+str(i+1)+".txt"
        data.append(np.array(pd.read_csv(data_path,delimiter=" ",header=None)))
    
    data=np.array(data)
    return data


# =============================================================================
# main
# =============================================================================

#IMP:-
# classes --> 0-based indexing
# index in array = label of class
# 1-vs-1 classification model used

#----- input specs -----
no_of_classes=3
input_dimension=2
rows_per_class=500
input_path="./Group21/Classification/LS_Group21/Class"

# ----read data from file -----
# data[i] --> data of class with label=i
data=get_data(input_path,no_of_classes)
# add 1 to input (for bias in weights)
data=np.append(data,np.ones((no_of_classes,rows_per_class,1)),axis=2)

# split data --> train,validate,test
train,validate,test,actual_test_class=split_data(data,no_of_classes)

# perceptron weights
W=perceptron_weights(train,no_of_classes)

# predict class label
predict_test_class=predict_test_data_class(test,W,no_of_classes)

