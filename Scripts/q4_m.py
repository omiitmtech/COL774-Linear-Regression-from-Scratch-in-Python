#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


#- Read Data from files
#Alaska=0 Canada=1
def read_data():
    X_DF = pd.read_table("ass1_data/q4/q4x.dat",sep="\s+",header = None)
    Y_DF = pd.read_csv("ass1_data/q4/q4y.dat",sep="\s+",header = None)

    X = pd.DataFrame(X_DF).to_numpy()
    Y = pd.DataFrame(Y_DF).to_numpy()
    m = len(X)
    Y = Y.reshape(m,)
    return X,Y,m


# In[3]:


def normalization(input_arr):
    mean = np.mean(input_arr, axis=0)
    std  = np.std(input_arr, axis=0)
    norm_val = (input_arr-mean)/std
    return norm_val


# In[4]:


X,Y,m = read_data()
X = normalization(X)


# In[5]:


def cal_phi(Y):
    m = len(Y)
    count = sum(map(lambda x : x == 'Canada', Y))
    phi = count/m
    return phi


# In[6]:


def cal_mu(X,Y,m,val):
    sum_val = 0
    count = 0
    for i in range(0,m):
        if(Y[i] == val):
            sum_val += X[i][0]
            count +=1
    mu0 = sum_val/count
    sum_val =0
    count = 0
    for i in range(0,m):
        if(Y[i] == val):
            sum_val += X[i][1]
            count +=1
    mu1 = sum_val/count
    mu = np.array([mu0,mu1])
    return mu
    


# In[7]:


#calculate mu values
mu0 = cal_mu(X,Y,m,'Alaska')
mu1 = cal_mu(X,Y,m,'Canada')
print('mu0 :',mu0)
print('mu1 :',mu1)


# In[8]:


def cal_sigma(X,Y,m,mu0,mu1):
    sum = 0
    for i in range(0,m):
        if Y[i] == 'Alaska':
            temp = (X[i] - mu0).reshape(2,1)
            t_temp = temp.T
            t_mul = np.dot(temp,t_temp)
        else:
            temp = (X[i] - mu1).reshape(2,1)
            t_temp = temp.T
            t_mul = np.dot(temp,t_temp)
        
        sum = sum + t_mul
    return sum/m
            
    


# In[9]:


sigma = cal_sigma(X,Y,m,mu0,mu1)
print('sigma :\n',sigma)


# In[10]:



def plot_data(X,Y):
    xdata1=[]
    ydata1=[]
    xdata2=[]
    ydata2=[]
    for i in range(0,100):
        if Y[i] == 'Alaska':
            xdata1.append(X[i][0]) 
            ydata1.append(X[i][1]) 
        else :
            xdata2.append(X[i][0]) 
            ydata2.append(X[i][1])
            
#     print(xdata.shape)
    plt.title('GDA - Data Plots')
    plt.scatter(xdata1,ydata1,s=25,c='b', label="Alaska", marker="+")
    plt.scatter(xdata2,ydata2,s=25,c='r',label="Canada", marker="*" )
    plt.xlabel('X1 Data')
    plt.ylabel('X2 Data')
    plt.legend(loc="upper right")
    plt.show()


# In[11]:


plot_data(X,Y)


# In[12]:


#Que4.c
def cal_coff_val(sigma, mu0, mu1):
    mu0 = mu0.reshape(2,1)
    mu1 = mu1.reshape(2,1)
    mu_diff = mu0-mu1
    sigm_env = np.linalg.inv(sigma)
    coff1 = np.dot(mu_diff.T,sigm_env)
    
    temp1 = np.dot(mu0.T,sigm_env)
    mu0_term = np.dot(temp1, mu0)
    
    temp2 = np.dot(mu1.T,sigm_env)
    mu1_term = np.dot(temp2, mu1)
    
    coff0 = (mu1_term - mu0_term)/2
    
    return coff0, coff1
    


# In[13]:


#draw decision boundary when sigma0 = sigma1
def lin_decision_boundary(X,sigma, mu0, mu1):
    c0,c1 =  cal_coff_val(sigma, mu0, mu1)
    xdata1=[]
    ydata1=[]
    xdata2=[]
    ydata2=[]
    for i in range(0,100):
        if Y[i] == 'Alaska':
            xdata1.append(X[i][0]) 
            ydata1.append(X[i][1]) 
        else :
            xdata2.append(X[i][0]) 
            ydata2.append(X[i][1])
    
    a = c1[0][0]
    b = c1[0][1]
    x1 = X[:,0].reshape(100,1)
    x2 = (c0 + a*x1)/b
    x2 = (-1)*x2
    
    plt.plot(x1,x2,label = "Linear boundary")
#     plt.title('GDA - Covariance matrixes are not same')
    plt.scatter(xdata1,ydata1,s=25,c='b', label="Alaska", marker="+")
    plt.scatter(xdata2,ydata2,s=25,c='r',label="Canada", marker="*" )
    plt.legend()
    plt.show()
    


# In[14]:


lin_decision_boundary(X,sigma, mu0, mu1)


# In[15]:


#Que4.c
#it calculates both the covariance matrices
def cal_diff_sigma(X,Y,m,mu0,mu1):
    sigma0 = 0
    sigma1 = 0
    y0 = 0
    y1 = 0
    for i in range(0,m):
        if Y[i] == 'Alaska':
            y0 +=1
            temp = (X[i] - mu0).reshape(2,1)
            t_temp = temp.T
            t_mul = np.dot(temp,t_temp)
            sigma0 = sigma0 + t_mul
        else:
            y1 +=1
            temp = (X[i] - mu1).reshape(2,1)
            t_temp = temp.T
            t_mul = np.dot(temp,t_temp)
            sigma1 = sigma1 + t_mul
    return sigma0/y0, sigma1/y1


# In[16]:


def cal_quad_equation(X,Y,m,mu0, mu1):
    print('mu_0 :',mu0,'mu_1 :',mu1)
    phi = cal_phi(Y)
    sigma0, sigma1 = cal_diff_sigma(X,Y,m,mu0,mu1)
    print('sigma0 :',sigma0)
    sigma0_inv = np.linalg.inv(sigma0)
    print('sigma1 :',sigma1)
    sigma1_inv = np.linalg.inv(sigma1)
    detcov_0 = np.linalg.det(sigma0_inv)
    detcov_1 = np.linalg.det(sigma1_inv)
    
    a = sigma1_inv - sigma0_inv
    mu0 = mu0.reshape(1,2)
    
    b = 2*(mu0.dot(sigma0_inv)-mu1.dot(sigma1_inv))
    
    print('sigma0_inv shape',sigma0_inv.shape)
    
    y1 = a[0][0]
    y2 = a[1][1]
    print('mu0 shape',mu0.shape)
    delt = a[1][0]+a[0][1]
    mm = b[0][0]
    print('mu0 :',mu0)
    bb = b[0][1]
    print('mu1 :',mu1)
    tr = (mu1.dot(sigma1_inv)).dot(mu1.T)
    cost = (tr-(mu0.dot(sigma0_inv)).dot(mu0.T))[0,0] 

    cost = cost + np.log(detcov_0/detcov_1)

    return y1,y2,delt,mm,bb,cost
    


# In[17]:


def plot_quad_graph(X,Y,m,mu0, mu1):

    a,c,b,d,e,f = cal_quad_equation(X,Y,m,mu0, mu1)

    print(a,b)


    x1 = np.arange(-4, 4, 0.05)
    x2 = np.arange(-10, 7.5, 0.05)
    x, y = np.meshgrid(x1, x2)
    tt = (a*x**2 + b*x*y + c*y**2 + d*x + e*y )

    print(c,d,e,f)
    plt.contour(x, y,tt + f, [0], colors='b')
    print('boundary drawn')
    plt.title('Decision Boundary different Covariance')

    lin_decision_boundary(X,sigma, mu0, mu1)
    plt.show()


# In[18]:



plot_quad_graph(X,Y,m,mu0, mu1)





# In[ ]:




