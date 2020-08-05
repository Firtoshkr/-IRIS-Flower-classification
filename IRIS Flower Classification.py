#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = np.array(pd.read_csv("C:\\Users\\firto\\IrisFlower.csv"))


# In[3]:


X = data[:,:4]


# In[4]:


X = X.reshape(150,2,2)


# In[5]:


Y = data[:,-1]


# In[6]:


def distance(X, querypoint):
    return np.sqrt(np.sum((querypoint-X)**2))

def knn(X, Y, querypoint, k = 5):
    m = X.shape[0]
    queryp = querypoint
    dis = []
    for i in range(m):
        xi = X[i]
        xi = xi.reshape((4,))
        dist = distance(xi,queryp)
        dis.append((dist,Y[i]))
    dis = sorted(dis, key = lambda x:x[0])[:k]
    dis = np.array(dis)
    dis = np.unique(dis[:,1],axis=0,return_counts = True)
    index = dis[1].argmax()
    result = dis[0][index]
    return result


# In[7]:


querypoint = np.array([[5.0, 2.0],
        [3.5, 1.0]])
querypoint = querypoint.reshape((4,))


# In[8]:


you = knn(X, Y, querypoint)


# In[9]:


you = np.array(you)


# In[10]:


you


# In[11]:


X[0]


# In[12]:


data


# In[ ]:




