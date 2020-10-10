#!/usr/bin/env python
# coding: utf-8

# #que_3 logistic regression

# In[1]:


#Import all the required libraries here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[3]:


#l_dash function
def l_dash_func(X, g_theta, Y):
    return -1*( X.T @ (g_theta - Y))


# In[4]:


#function to calculate hession
def hessian(X, g_theta):
    diag_val = np.array([i*(1-i) for i in g_theta])
    diag_matrix = np.diag(diag_val)
    Temp = (X.T @ diag_matrix @ X)
    H = (-1)*Temp
    return H


# In[5]:


#Function to normalize the data
def normalization (input_arr, temp):
    mean = np.mean(input_arr)
    std  = np.std(input_arr)
    med  = np.median(input_arr)
    print("mean of data :{}, standard deviation :{}, median of data :{}".format(mean,std,med))

    input_arr = (input_arr-mean)/std
    return input_arr


# In[6]:


#It draws the linear boundary in the data
def line_draw(slope, intercept):
    xx = np.linspace(-2,5,10)
    yy = -1*intercept + -1*slope * xx
    y = slope * xx + intercept
    plt.plot(xx, yy, 'r-',label = "Linear boundary by logistic regression")
    legend = plt.legend(loc="upper left")


# In[7]:


#Logistic function
def logistic_function(X,Y,theta, epsilon):
    flag = 1
    for i in range(0,1000):
        g_theta = sigmoid(X@theta)
        L_dash = l_dash_func(X,g_theta, Y)
        H = hessian(X,g_theta)
        inv_H = np.linalg.inv(H)
        theta = theta - inv_H @ L_dash

        for l in L_dash:
            if np.abs(l) > epsilon:
                flag = 0
        if(flag !=0):
            break
        flag = 1
    print('Theta :', theta)
    print('Iterations# ',i)
    slope = (theta[1]/theta[2])
    intercept = (theta[0]/theta[2])
    line_draw(slope,intercept)
    
    plt.xlabel('X -Input 1 ')
    plt.ylabel('X -Input 2 ')
    print(X.shape)
    xdata1=[]
    ydata1=[]
    xdata2=[]
    ydata2=[]
    
    for i in range(0,100):
        if Y[i] == 0:
            xdata1.append(X[i][1]) 
            ydata1.append(X[i][2]) 
        else :
            xdata2.append(X[i][1]) 
            ydata2.append(X[i][2])
            
    plt.title('Decision Boundary (Newtons Method)')
    plt.scatter(xdata1,ydata1,s=10,c='b',label='Label0')
    plt.scatter(xdata2,ydata2,s=10,c='r',label='Label1')
    plt.legend()
    plt.show()
    


# In[8]:


#Read input files
input_fileX = "ass1_data/q3/logisticX.csv"
input_fileY = "ass1_data/q3/logisticY.csv"

X_data = np.loadtxt(input_fileX,delimiter = ',')
Y_data = np.loadtxt(input_fileY)

Y = Y_data

m,n =X_data.shape
theta = [0,0,0]
no_ones = np.ones(m)


X_data0 = normalization(X_data.T[0],0)
x_data1 = normalization(X_data.T[1],0)
#ones are added to the x matrix
X = np.stack((no_ones,X_data0,x_data1), axis=-1)
#stoping condition
epsilon = 0.000000000001 #10^-1
logistic_function(X,Y, theta, epsilon ) 


# In[ ]:





# In[ ]:




