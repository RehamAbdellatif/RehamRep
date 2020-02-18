#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot


# In[53]:


excel_file = 'house_prices_data_training_data.csv'
housePrices = pd.read_csv("house_prices_data_training_data.csv", error_bad_lines=False )


# In[54]:


housePrices.tail()


# In[55]:


PricesNoEmpty=housePrices.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


# In[56]:


train, validate, test = np.split(PricesNoEmpty.sample(frac=1), [int(.6*len(PricesNoEmpty)), int(.8*len(PricesNoEmpty))])


# In[57]:


def plotcol (dataset):
    Price = dataset["price"] 
    bedrooms= dataset["bedrooms"]
    PriceList=[]
    bedroomList=[]
    PriceList=list(Price)
    bedroomList=list(bedrooms)
    plt.plot(bedroomList, PriceList, 'ro', ms=10, mec='k')
    plt.xlabel('bedrooms->')
    plt.ylabel('Prices->')
    plt.title('House Prices')
    plt.show()


# In[58]:


def featureNormalize(X):

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


# In[59]:


def costmulti(theta, X, y, h, lambda_,m):
  
    m = y.shape[0] # number of training examples
    
  
    J = 0
    
   
    J = np.dot((h - y), (h - y)) / (2 * m) + ((lambda_/(2 * m))* np.sum(np.dot(theta, theta)))

    return J


# In[60]:


def costfx(y, h, m):
    J= np.dot((h - y), (h - y)) / (2 * m)
    return J


# In[61]:


def firstgradientDescent(X, y, theta, alpha, num_iters, lambda_):
  
    m = y.shape[0] # number of training examples
    
    
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
      
        alphabym = alpha / m
        h = np.dot(X, theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costmulti(theta, X, y, h, lambda_,m))

    
    return theta, J_history


# In[62]:


def secondgradientDescent(X, y, theta, alpha, num_iters , lambda_):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []  # Use a python list to save cost in every iteration

    for i in range(num_iters):
        alphabym = alpha / m
        h = np.dot(np.power(X,2), theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costmulti(theta, X, y, h, lambda_,m))

    return theta, J_history


# In[63]:


def thirdgradientDescent(X, y, theta, alpha, num_iters ,lambda_):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []  # Use a python list to save cost in every iteration

    for i in range(num_iters):
        alphabym = alpha / m
        h = np.dot(hyp3X, theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costmulti(theta, X, y, h, lambda_,m))

    return theta, J_history


# In[64]:


def getnormalizedSet (Setnorm):
    
    Setcopy= Setnorm.copy()
    Setnorm, mu, sigma = featureNormalize(Setcopy.iloc[0::,3::])
    y=Setcopy.iloc[0::,2]
    m=y.size
    finalSet = np.concatenate([np.ones((m, 1)), Setnorm], axis=1)
    
    return finalSet 


# In[65]:


def hypothesis (Set, hypnumber):
    if hypnumber == "hyp1":
        hyp = np.dot(Set, first_theta)
    elif hypnumber == "hyp2":
        hyp = np.dot(np.power(Set,2) , first_theta)
    else:
        hyp3X= Set.copy()
        hyp3X[:, 2] = np.power(hyp3X[:, 2], 3)
        hyp = np.dot(hyp3X , first_theta)


# In[66]:


kfold=[1,2,3]
lambdaList = [0.02, 0, 0.005, 1 , 5]
alpha=0.05
alpha2=0.0029
alpha3=0.02
for k in kfold:
    train2, validate2, test2 = np.split(PricesNoEmpty.sample(frac=1), [int(.6*len(PricesNoEmpty)), int(.8*len(PricesNoEmpty))])
    plotcol(train2)
    normalizedtrain=getnormalizedSet(train2)
    normalizedvali=getnormalizedSet(validate2)
    first_theta = np.zeros(normalizedtrain.shape[1])
    hypoth1=hypothesis(normalizedtrain,"hyp1")
    hypoth2=hypothesis(normalizedtrain,"hyp2")
    hypoth3=hypothesis(normalizedtrain,"hyp3")
    arrayHk1 =[]
    arrayHk2 =[]
    arrayHk3 =[]
    hyp3vali= normalizedvali.copy()
    hyp3vali[:, 2] = np.power(hyp3vali[:, 2], 3)
    for i in lambdaList:
        theta, J_history = firstgradientDescent(normalizedtrain,y, first_theta, alpha, iterations, i )

        theta2, J_history2 = secondgradientDescent(normalizedtrain,y, first_theta, alpha2, iterations, i)

        theta3, J_history3 = thirdgradientDescent(normalizedtrain,y, first_theta, alpha3, iterations, i)
    
        h1validating = np.dot(normalizedvali, theta)
    
        h2validating = np.dot(np.power(normalizedvali, 2), theta2)
    
        h3validating = np.dot(hyp3vali, theta3)
    
        arrayHk1.append(costfx(yV, h1validating, mV))
    
        arrayHk2.append(costfx(yV, h2validating, mV))
    
        arrayHk3.append(costfx(yV, h3validating, mV))

    lambda_1= lambdaList[min(range(len(arrayHk1)), key=arrayHk1.__getitem__)]
    lambda_2= lambdaList[min(range(len(arrayHk2)), key=arrayHk2.__getitem__)]
    lambda_3= lambdaList[min(range(len(arrayHk3)), key=arrayHk3.__getitem__)]
    theta, J_history = firstgradientDescent(normalizedtrain,y, first_theta, alpha, iterations,lambda_1)
    theta2, J_history2 = secondgradientDescent(normalizedtrain,y, first_theta, alpha2, iterations,lambda_2)
    theta3, J_history3 = thirdgradientDescent(normalizedtrain,y, first_theta, alpha3, iterations,lambda_3)
    plt.figure()
    plt.plot(np.arange(len(J_history)), J_history,'r', lw=2, label='h1')
    plt.plot(np.arange(len(J_history2)), J_history2, lw=2, label='h2')
    plt.plot(np.arange(len(J_history3)), J_history3,'g', lw=2, label='h3')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost Function')
    plt.show()

