#!/usr/bin/env python
# coding: utf-8

# # Binary Tabu Search
# 
# > This code performs the Binary Tabu Search for the binary case where it is desired to find the set of pareto in a given set of points.

# ## 1. Importing

# In[10]:


import random
import numpy as np
import math
import time
from deap import benchmarks


# ## 2. Helpers of helpers

# In[ ]:





# ## 3. Tabu Search Operators

# In[11]:


# Check Domination
def dominates(row, candidateRow, evalfunction, weights):
    '''
    Introduction:
        Function to determine with candidateRow dominates row
    ---
    
    Input:
        row: Iterable object (list, set, ...)
            It's dominated?
        
        candidateRow : Iterable object (list, set, ...)
            It dominates?
            
        evalfunction : Function
            Function to map both rows to the space
            where they will be compared
            
        weights : Iterable object (list, set, ...)
            Weights of each dimension of the output
            They will will be used to maximize or
            minimize each feature
    ---
    
    Output:
        Boolean :
            True if it's dominated, false otherwise
    '''
    rrow = evalfunction(row)
    rcandidaterow = evalfunction(candidateRow)

    if len(rcandidaterow) != len(weights):
        raise Exception('Output of function different from number of weights')
    
    return sum([rrow[x]*weights[x] <= rcandidaterow[x]*weights[x] for x in range(len(rrow))]) == len(rrow)


# ## 4. Binary Tabu Search

# In[12]:


def binary_pareto_tabu_search(obj, weights, X):
    '''
    Introduction:
        Function to find the pareto set of the points in X
    ---
    
    Input:
        obj: Function
            Multi-objective function to optimize
            
        weights : Iterable object (list, set, ...)
            Weights of each dimension of the output
            They will will be used to maximize or
            minimize each feature
            
        X : List
            List of all points
    ---
    
    Output:
        The pareto set of X
    '''
    # Init first pareto front
    dim = len(X)
    ceil_value = [1]*dim
    S = [1] + [0]*(dim-1) # Pareto front
    
    Tabu = set([])
    
    # All possible moves
    M = [int(x) for x in range(len(X)) if x not in Tabu and S[x] != 1]

    for m in M:
        nomDominated = True
        for i in range(len(S)):
            # Solution in Pareto Front
            if S[i] == 1:
                if dominates(X[m], X[i], obj, weights):
                    nomDominated = False
                    Tabu.add(m)
                elif dominates(X[i], X[m], obj, weights):
                    S[i] = 0
                    Tabu.add(i)
                # If not dominated, add to Solution
                if nomDominated:
                    # Move
                    S[m] = 1
                # S' is a better solution than S, so we modifie directly in S
    sol =  []
    for i in range(len(S)):
        if S[i] == 1:
            sol.append(X[i])
    
    return sol

