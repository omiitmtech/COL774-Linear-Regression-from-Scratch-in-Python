#!/usr/bin/env python
# coding: utf-8

# In[12]:


#all the neccessary libraries are utilized here
import numpy as np
import matplotlib.pyplot as plt 
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import matplotlib.animation as animation
import matplotlib as mpl


# In[13]:


#This function generates the sample data based on the given distribution values
def generate_sample(sample_count):
    x0 = np.ones(sample_count).reshape(sample_count,1)
    x1 = np.random.normal(3,2,sample_count).reshape(sample_count,1)
    x2 = np.random.normal(-1,2,sample_count).reshape(sample_count,1)
    theta = np.array([3,1,2]).reshape(3,1)
#     print(theta)
#     print(theta.shape)
    e =  np.random.normal(0,math.sqrt(2),sample_count).reshape(sample_count,1)
    x_matrix = np.hstack((x0, x1, x2))
#     print(x_matrix)
#     print(x_matrix.shape)
    h_theeta = np.dot(x_matrix,theta)
#     print(h_theeta)
    y = h_theeta + e
    return x_matrix,y#-- function to generate sample data
# sample_count = no. of samples to be generated
def generate_sample(sample_count):
    x0 = np.ones(sample_count).reshape(sample_count,1)
    x1 = np.random.normal(3,2,sample_count).reshape(sample_count,1)
    x2 = np.random.normal(-1,2,sample_count).reshape(sample_count,1)
    theta = np.array([3,1,2]).reshape(3,1)
    e =  np.random.normal(0,math.sqrt(2),sample_count).reshape(sample_count,1)
    x_matrix = np.hstack((x0, x1, x2))
    h_theeta = np.dot(x_matrix,theta)
    y = h_theeta + e
    return x_matrix,y


# In[3]:


def get_batch(X,Y,b,r):
    m = len(X)
    left_index = b*r
    right_index = left_index + r
    if m < right_index:
        new_index = left_index+(right_index - m)
        X_Slice = X[left_index:new_index]
        Y_Slice = Y[left_index:new_index]
    else:
        X_Slice = X[left_index:right_index]
        Y_Slice = Y[left_index:right_index]
    return (X_Slice,Y_Slice)


# In[4]:


def loss_func(X, Y, theta):
    m = len(X)
    hypothesis = np.dot(X, theta)
    loss = hypothesis - Y
    return loss


# In[5]:


def cost_function(X, Y,theta):
    m = len(X)
    loss = loss_func(X, Y, theta)
    cost = np.sum(loss **2)
    cost = cost/(2*m)
    return cost


# In[6]:


def SGD(X,Y,m,r):
    print(X.shape,Y.shape,m,r)
    theta=np.array([0,0,0]).reshape(3,1)
    epoch = 0
    not_converged = True
    prev_cost = 100
    iteration=0
    total_iterations = 0
    cost_list = []
    theta0_list = []
    theta1_list = []
    theta2_list = []
    e = 1e-6 #10^-14
    while(not_converged):

        print('epoch :', epoch)
        print('Updated theta:',theta)
        epoch +=1
        for b in range(0, math.ceil(m/r)):
            X_Slice, Y_Slice = get_batch(X,Y,b,r)
#             loss = loss_func(X_Slice, Y_Slice, theta)
#             theta = theta - 0.001*(np.dot(X_Slice.T,loss)/r)
            t1 = np.dot(X_Slice,theta)
            t2 = t1 - Y_Slice
            t3 = np.dot(X_Slice.T,t2)
            t4 = 0.001*t3
            t5 = t4/r
            theta = theta - t5
            theta0_list.append(theta[0])
            theta1_list.append(theta[1])
            theta2_list.append(theta[2])
            current_cost = cost_function(X_Slice,Y_Slice,theta)
            cost_list.append(current_cost)
            iteration +=1
            total_iterations+=1
            
            if(r == 1 and total_iterations==50000):
                not_converged = False
                break
            
            if(r == 100 and total_iterations==100000):
                not_converged = False
                break
            
            if(iteration == 1000):
                
                iteration=0
                list_sum = sum(cost_list)
                new_cost = list_sum/1000
                cost_list = []
                print('Previous Cost :',prev_cost,'Next Cost :', new_cost,'Difference:', prev_cost - new_cost)
                if abs(prev_cost - new_cost) < e:
                    not_converged = False
                prev_cost = new_cost
                
            
    print('No. of samples :',m,'Batch Size :',r,'Total Iterations:',total_iterations,'No of Epochs :',epoch)
    print('Final Theta :',theta)
    data = np.hstack((theta0_list, theta1_list,theta2_list))
    print(theta.shape)
    return data.T,theta


# In[7]:


start_time = time.time()
X1,Y1 = generate_sample(1000000)
data,t2 = SGD(X1,Y1,1000000,1)
end_time = time.time()
print('Execution Time in seconds :',round(end_time - start_time,2) ) 


# In[8]:


def cost_function(X, Y, theta):
    m = len(X)
    #hypothesis is calculated here
#     print('ert')
#     print(type(X))
#     print(type(theta))
#     theta = theta.astype('float64')
    hypothesis = np.dot(X, theta)
    loss = hypothesis - Y
    cost = np.sum(loss **2)
    cost = cost/(2*m)
    return cost


# In[9]:


def read_data():
    acidity = pd.read_csv("ass1_data/q2/q2test.csv", header=None, skiprows=1)
    T_Y = acidity[2].values
    m = len(T_Y)
    T_X = acidity[[0,1]]
    T_X = np.array(T_X)
    ones_arr = np.ones(m)
    ones_arr = ones_arr.reshape(m,1)
    T_X = np.hstack((ones_arr,T_X))
    return T_X,T_Y


# In[10]:


def test():
    X,Y = read_data()
    orig_theta = np.array([3,1,2])
    rtheta_0 = np.array([2.99968088, 0.95472704, 1.97118111])
    rtheta_1 = np.array([2.98449633 , 1.00397983, 1.99754537])
    rtheta_2 = np.array([2.99101259 , 1.0017618 , 1.99969984])
    rtheta_3 = np.array([2.99776263 ,1.00061538 ,1.9999845 ])
    theta_list=np.vstack((rtheta_0,rtheta_1,rtheta_2,rtheta_3))
#     print(theta_list.shape)
#     print(type(rtheta_0[0]))
#     print(type(orig_theta[0]))
    for i in range(0,4):
        orig_cost = cost_function(X, Y, orig_theta)
        print('original cost :',orig_cost)
        my_cost = cost_function(X, Y, theta_list[i])
        print('My Theta :',theta_list[i])
        print('My Cost :',my_cost)
        print('Error difference :',my_cost - orig_cost)
    


# In[11]:


test()


# In[ ]:




