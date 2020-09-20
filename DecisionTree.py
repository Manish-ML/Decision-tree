# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:51:26 2020

@author: Nikhil
"""


# Decision tree
# Decision tree is a machine learning algorithm to predict target
# variable using a tree like structure unlike regression and 
# logistic regression where we get an equation for prediction.

# Here decision are made using tree structure which include
# root node and decision nodes. at each node we use feature
# variable and some condition applied in it. These condition
# will help us to filter data and reach to a decision for
# target variable.
# root node is first variable to be used to filter data
# decision node is variable used between leaf node and root node
# leaf node is target variable value (prediction result)

# there is two methods to select root node
# 1. CART model (classification And Regression Tree)- under this
# method we use gini index to find root node variable
# gini index = sum(p^2,q^2)
# p- probability of success
# q = 1-p
# example - if out of 10 observations (records), if we have 
# 4 yes and 6 no, then
# p= 4/10, q=6/10
# which variable has highest gini value will be used as root node

# 2. ID3 model (iterative dichomiser 3) - under this method, we use 
# entropy and information gain to find root node
# entropy = sum(-p*log2p, -q*log2q)
# information gain = entropy of target variable - entropy of feature variable

# which ever variable has highest information gain, will be used as root node

# entropy can be defined probability of homogenity



# Steps to perfrom decision tree
# 1. Extraction
# 2. Identify X, Y and split into train and test
# 3. Build model using train data
# 4. Predict for test data using model
# 5. calculate accuracy of model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def importdata():
    data=pd.read_csv("D:/Part 2-20200118T070546Z-001/Part 2/Python Part 2 Classnotes/Material/Class 12/balance_scale.csv",
                     header=None)
    print(data.shape)
    print(data.head())
    return data;

def train_test(dataset):
    X=dataset.iloc[:,1:]
    Y=dataset.iloc[:,0]
    train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,
                       random_state=10 )
    return train_x,test_x,train_y,test_y;

def decision_tree_model(train_x,train_y,method):
    model=DecisionTreeClassifier(criterion=method).fit(train_x,train_y)
    return model;

def prediction(model_name,test_x):
    pred_y=model_name.predict(test_x)
    return pred_y;

def model_accuracy(test_y,pred_y):
    print(confusion_matrix(test_y,pred_y))
    print(accuracy_score(test_y,pred_y))
    return ;

def main():
    data=importdata()
    train_x,test_x,train_y,test_y=train_test(data)
    model_gini=decision_tree_model(train_x,train_y,"gini")
    model_entropy=decision_tree_model(train_x,train_y,"entropy")
    pred_y_gini=prediction(model_gini,test_x)
    pred_y_entropy=prediction(model_entropy,test_x)

    print("Gini Result")
    model_accuracy(test_y,pred_y_gini)

    print("Entropy Result")
    model_accuracy(test_y,pred_y_entropy)
    return ;

if (__name__=="__main__"):
    main()
    
    
Q2 Assignment
  
def importdata():
    data=pd.read_excel("D:\Part 2-20200118T070546Z-001\Part 2\Python Part 2 Classnotes\Material\Class 12\Case Study\Case 2/Data.xlsx",
                     sheetname='Sheet2', header=0)
    print(data.shape)
    print(data.head())
    return data;

def train_test(dataset):
    X=dataset.iloc[:,:4]
    Y=dataset.iloc[:,4]
    train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,
                       random_state=10 )
    return train_x,test_x,train_y,test_y;

def decision_tree_model(train_x,train_y,method):
    model=DecisionTreeClassifier(criterion=method).fit(train_x,train_y)
    return model;

def prediction(model_name,test_x):
    pred_y=model_name.predict(test_x)
    return pred_y;

def model_accuracy(test_y,pred_y):
    print(confusion_matrix(test_y,pred_y))
    print(accuracy_score(test_y,pred_y))
    return ;

def main():
    data=importdata()
    train_x,test_x,train_y,test_y=train_test(data)
    model_gini=decision_tree_model(train_x,train_y,"gini")
    model_entropy=decision_tree_model(train_x,train_y,"entropy")
    pred_y_gini=prediction(model_gini,test_x)
    pred_y_entropy=prediction(model_entropy,test_x)

    print("Gini Result")
    model_accuracy(test_y,pred_y_gini)

    print("Entropy Result")
    model_accuracy(test_y,pred_y_entropy)
    return ;

if (__name__=="__main__"):
    main()
    
    
    
Q3 Assignment
  
def importdata():
    data=pd.read_excel("D:\Part 2-20200118T070546Z-001\Part 2\Python Part 2 Classnotes\Material\Class 12\Case Study\Case 2/Data.xlsx",
                     sheetname='Sheet2', header=0)
    print(data.shape)
    print(data.head())
    return data;

def train_test(dataset):
    X=dataset.iloc[:,:4]
    Y=dataset.iloc[:,4]
    train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,
                       random_state=10 )
    return train_x,test_x,train_y,test_y;

def decision_tree_model(train_x,train_y,method):
    model=DecisionTreeClassifier(criterion=method).fit(train_x,train_y)
    return model;

def prediction(model_name,test_x):
    pred_y=model_name.predict(test_x)
    return pred_y;

def model_accuracy(test_y,pred_y):
    print(confusion_matrix(test_y,pred_y))
    print(accuracy_score(test_y,pred_y))
    return ;

def main():
    data=importdata()
    train_x,test_x,train_y,test_y=train_test(data)
    model_gini=decision_tree_model(train_x,train_y,"gini")
    model_entropy=decision_tree_model(train_x,train_y,"entropy")
    pred_y_gini=prediction(model_gini,test_x)
    pred_y_entropy=prediction(model_entropy,test_x)

    print("Gini Result")
    model_accuracy(test_y,pred_y_gini)

    print("Entropy Result")
    model_accuracy(test_y,pred_y_entropy)
    return ;

if (__name__=="__main__"):
    main()    
    
    
    
    
    
