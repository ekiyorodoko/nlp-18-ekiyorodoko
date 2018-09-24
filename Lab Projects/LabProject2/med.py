#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np


# In[2]:


def min_edit_distance(source, target):
    n = len(source)
    m = len(target)
    
    # Distance matrix D for source and target strings
    D = np.zeros((n+1,m+1), dtype=int)
    
    # Initialize base case values
    D[0][0] = 0
    
    # Create boundary for two base cases
    # Base case 1: source substring to empty target
    for i in range(1, n+1):
        D[i][0] = D[i-1][0]+1
    # Base case 2: empty source to target substring
    for j in range(1, m+1):
        D[0][j] = D[0][j-1]+1
    
    # Compute distance for every point in D
    for i in range(1, n+1):
        for j in range(1, m+1):
            dist = [D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+sub_cost(source[i-1], target[j-1])]
            D[i][j] = dist[np.argmin(dist)]
 
    return D[n][m]


# In[3]:


def sub_cost(s, t):
    if s==t:
        return 0
    else:
        return 2


# In[4]:


def main(source, target):
    med = min_edit_distance(source, target)
    print("Minimum edit distance between %s and %s is %d" %(source,target,med)) 


# In[ ]:


main(sys.argv[1], sys.argv[2])
