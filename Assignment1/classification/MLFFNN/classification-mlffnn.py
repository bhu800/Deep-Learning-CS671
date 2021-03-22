# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pandas as pd


# =============================================================================
# Functions
# =============================================================================
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

# ----------- activation function -----------
# Logistic function
def activation_function(a):
    # slope parameter
    beta = 1
    
    return 1/(1 + np.exp(-beta*a))

# -------- feed forward------------
def feedForward(no_of_classes,train,Wh,Wo):
    result=[]
    for i in range(no_of_classes):
        for j in train[i]:
            # i is correct class label for data j
            ah=np.matmul(Wh.T,j)
            sh=activation_function(ah)
            
            ao=np.matmul(Wo.T,sh)
            so=activation_function(ao)
            
            predict_label=np.argmax(so)
            result.append(predict_label)
    
    return result

# =============================================================================
# main
# =============================================================================

#IMP:-
# classes --> 0-based indexing
# index in array = label of class

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


# ---- Neural Network ------
# Architecture : 1-hidden layer neural network
input_layer_nodes=input_dimension+1
hidden_layer_nodes=3
output_layer_nodes=no_of_classes

# initialize weights
Wh=np.zeros((input_layer_nodes,hidden_layer_nodes))
Wo=np.zeros((hidden_layer_nodes,output_layer_nodes))

x=feedForward(no_of_classes,train,Wh,Wo)
print(x)
