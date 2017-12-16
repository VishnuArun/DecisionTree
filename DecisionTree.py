import pandas as pd
import numpy as np


#Filenames
Training_File = '/Users/Vishthefish73/Desktop/DecisionTreePy/optdigits_train.txt'
Valid_File = '/Users/Vishthefish73/Desktop/DecisionTreePy/optdigits_valid.txt'
Test_File = '/Users/Vishthefish73/Desktop/DecisionTreePy/optdigits_test.txt'


#Node used to construct tree
class Node:
    def __init__(self):
        feature = None
        left = None
        right = None
        clas = None

#Function to read csv data line by line
def ReadData(trn_file,val_file,tst_file):
    #lat column is y_pred values
    df_trn = pd.read_csv(trn_file,header=None)
    df_val = pd.read_csv(val_file,header=None)
    df_tst = pd.read_csv(tst_file,header=None)

    return [df_trn.iloc[:,0:-1],df_trn.iloc[:,-1], df_val.iloc[:,0:-1],df_val.iloc[:,-1], df_tst.iloc[:,0:-1],df_tst.iloc[:,-1], ]


#Calculate Node entropy
def NodeEntropy(y): 
    if(len(y) == 0):
        return 0
    unique, cnt = np.unique(y, return_counts=True)
    n = len(unique)
    
    sm = 0
    for x in range(0,n):
        if(cnt[x] > 0):
            sm = sm + (-1 * ((cnt[x]/len(y))* np.log2(cnt[x]/len(y))));
    

    return sm

#Calculates entropy by feature
def SplitEntropy(y0,y1):
    l0 = len(y0)
    l1 = len(y1)
    if(l1+l0) == 0:
        return float("inf")
    N0 = NodeEntropy(y0)
    N1 = NodeEntropy(y1)

    return ( (l0/(l1+l0)) * N0) + ((l1/(l1+l0)) * N1)

#Return best feature to split on 
def SplitAttribute(X,y):
    minimum = float("inf")
    rows, dim = X.shape
    best = 0

    for i in range(0,dim):
        #print(i)
        ind0 = X.index[X.iloc[:,i] == 0]
        ind1 = X.index[X.iloc[:,i] == 1]
        
        
        y0 = y[ind0].reset_index(drop=True)
        y1 = y[ind1].reset_index(drop=True)

        e = SplitEntropy(y0,y1)
        
        
        
        if (e < minimum):
            minimum = e
            best = i
        #print(minimum)
    return best


#Generate a BiVvariate Decision Tree
def GenerateTree(X,y,theta,node):
    
    if(NodeEntropy(y) < theta):
        if(len(y) == 0):
            print("error")
            
        else:
        
            node.clas = y.mode()[0]
            return node
            

    else:
        
        i = SplitAttribute(X,y)
        node.clas = -1
        node.feature = i
        
        ind0 = X.index[X.iloc[:,i] == 0]
        
        ind1 = X.index[X.iloc[:,i] == 1]
    
      
        X0 = X.iloc[ind0,:].reset_index(drop=True)
        
        X1 = X.iloc[ind1,:].reset_index(drop=True)
        
        y0 = y[ind0].reset_index(drop=True)
        
        y1 = y[ind1].reset_index(drop=True)
       
    
    
        node.left = (GenerateTree(X0,y0,theta,Node()))
        
        node.right = (GenerateTree(X1,y1,theta,Node()))

    return node

#Using generated tree, makes predictions      
def Predict_with_Tree(root,X):
    node = root
    while(node.clas == -1):
        atrib = node.feature
        
        
        if (X.iloc[atrib] == 0):
            node = node.left
        else:
            node = node.right
    return node.clas

#main
def main():

    #Read in Data
    data = ReadData(Training_File,Valid_File,Test_File)
    X_trn = data[0]
    y_trn = data[1]
    X_val = data[2]
    y_val = data[3]
    X_tst = data[4]
    y_tst = data[5]

    #Calculate number of rows per dataset
    Trn_rows = X_trn.shape[0]
    Val_rows = X_val.shape[0]
    Tst_rows = X_tst.shape[0]

    #Theta parameter for decision tree
    thetas = [0.01,0.2,0.3,0.4,0.5,1.0,2.0]
    val_errors = []

    
    for theta in thetas:

        #Generate a tree with current predictions on Training set
        
        root = Node()
        GenerateTree(X_trn,y_trn,theta,root)
        
        
        error_rate = 0
        
        for r in range(0,Trn_rows):
            row_vector = X_trn.iloc[r,:]
            y_pred = Predict_with_Tree(root,row_vector)
            if(y_pred != y_trn.iloc[r]):
                error_rate+=1

        error_rate = error_rate / Trn_rows
        print("The error on the training set for " + str(theta) + " is " + str(round(error_rate,6)))

        #Predict labels on Validation set
        
        error_rate = 0
        
        for r in range(0,Val_rows):
            row_vector = X_val.iloc[r,:]
            y_pred = Predict_with_Tree(root,row_vector)
            if(y_pred != y_val.iloc[r]):
                error_rate+=1

        error_rate = error_rate / Val_rows
        val_errors.append(error_rate)
        print("The error on the validation set for theta = " + str(theta) + " is " + str(round(error_rate,6)))

    #Test set Prediction
    
    BestTheta = thetas[np.argmin(val_errors)]

    
    root = Node()

    GenerateTree(X_trn,y_trn,BestTheta,root)
     
    error_rate = 0
    
    for r in range(0,Tst_rows):
        row_vector = X_tst.iloc[r,:]
        y_pred = Predict_with_Tree(root,row_vector)
        if(y_pred != y_tst.iloc[r]):
            error_rate+=1

    error_rate = error_rate / Tst_rows
    print("The error on the test set for " + str(BestTheta) + " is " + str(round(error_rate,6)))
    
main()    
    

    




    
