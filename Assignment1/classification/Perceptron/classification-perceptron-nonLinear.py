# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

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
    error_array=[]
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
        error_array.append(err)
        if (abs(err - err_last) < err_limit):
            break
        
        err_last = err
    plt.plot(error_array)
    plt.show()      
    return weights    


# ------Initilize and find weights of perceptrons ------
# w[i][j] --> weights,bias of perceptron for diff b/w Class-i & Class-j
# w[i][j][-1] --> bias
def perceptron_weights(train,no_of_classes):
    # initialize
    W=np.zeros((no_of_classes,no_of_classes,input_dimension+1))
    
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
def get_data(input_path,no_of_classes,rows_per_class):
    raw_data=pd.read_csv(input_path,skiprows=1,skipinitialspace=True,delimiter=" ",header=None)
    raw_data=raw_data.values
    
    data=[]    
    data.append(raw_data[0:rows_per_class])
    data.append(raw_data[rows_per_class:2*rows_per_class])
    
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
no_of_classes=2
input_dimension=2
rows_per_class=2446
input_path=r"../../Group21/Classification/NLS_Group21.txt"


# ----read data from file -----
# data[i] --> data of class with label=i
data=get_data(input_path,no_of_classes,rows_per_class)
# add 1 to input (for bias in weights)
data=np.append(data,np.ones((no_of_classes,rows_per_class,1)),axis=2)

# split data --> train,validate,test
train,validate,test,actual_test_class=split_data(data,no_of_classes)

# ------- perceptron weights -------
W=perceptron_weights(train,no_of_classes)

# true class label
# assumption - all classes have equal test examples, class0 class1 ...... class[no_of_classes-1]
true_test_class=np.array([])
for i in range(no_of_classes):
    true_test_class=np.append(true_test_class,np.zeros(int(len(test)/2)) + i)
    
# ------ predict class label --------
predict_test_class=predict_test_data_class(test,W,no_of_classes)


# =============================================================================
# Results
# =============================================================================
# confusion matrix
confusion_matrix=metrics.confusion_matrix(true_test_class, predict_test_class)
print("Confusion matrix:-")
print(confusion_matrix)


# Visualising the Decision Region
from matplotlib.colors import ListedColormap
X_set=test
y_set = predict_test_class
# create a grid in 2d plane
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 100, stop = X_set[:, 0].max() + 100, step = 1),
                     np.arange(start = X_set[:, 1].min() - 100, stop = X_set[:, 1].max() + 100, step = 1))
# plot contours with predicted output/label as height of contour
plt.contourf(X1, X2, predict_test_data_class(np.array([X1.ravel(), X2.ravel(),np.ones(X1.ravel().shape[0])]).T,W,no_of_classes).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
# plot test data
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = "class "+ str(j+1))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title('Decision Region')
plt.xlabel("dimension-1 (data[0])")
plt.ylabel("dimension-2 (data[1])")
plt.legend()
plt.show()