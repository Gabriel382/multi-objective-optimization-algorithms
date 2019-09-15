#!/usr/bin/env python
# coding: utf-8

# # PSA - PARETO SIMULATED ANNEALING
# 
# > This code performs the Pareto Simulated Annealing for the binary case.

# ## 1. Importing

# In[3]:


import random
import numpy as np
import copy
import time
import math
from functools import partial


# ## 2. Helpers of helpers

# In[4]:


def binToInt(individual):
    '''
    Input:
        individual: deap.toolbox.individual
            bit iterable (tuple, list, ...).
    ---
    Output:
        Converts bit to int
    ''' 
    out = 0
    for bit in individual:
        out = (out << 1) | int(bit)
    return out


# In[5]:


def feasibility(X_size, x):
    '''
    Intro:
        Default feasibility function to check if
        x belongs to the boundaries of X
    ---
    
    Inputs:
        x : Iterable object (list, set, ...)
            Point which we want to check
        
        X_size : Integer
            Number of elements in X
            
    ---
    
    Outputs:
        True if x belongs to the boundaries, false otherwise
    '''
    if(binToInt(x) < (X_size)):
        return True
    else:
        return False


# In[6]:


def regulatedMultiFunction(obj, X, x):
    '''
    Intro:
        Function that put the multi-objective function
        in the default way necessary to the PSA function
    ---
    
    Inputs:
        x : Iterable object (list, set, ...)
            Point which we want to check
        
        X : List
            List of all points
        
        obj : Function
            Function that we want to find the pareto
            front
    ---
    
    Output:
        The result of the mapped function (get position of point in
        X by binary label in x and uses function to get result)
            
    ---
    
    Outputs:
        True if x belongs to the boundaries, false otherwise
    '''
    pos = binToInt(x)
    return obj(X[pos])


# In[1]:


def createEnvironment(obj, X):
    '''
    Intro:
        Function that returns the normalized function
        that we want to find the pareto front and
        the default feasibility funtion to check
        if binary label x points to a place in X
    ---
    
    Inputs:
        obj : Function
            Function that we want to find the pareto
            front
        
        X : List
            List of all points
            
    ---
    
    Outputs:
        Returns multi-objective function and feasibility function
    '''
    
    feasibilityNormalized = partial(feasibility,len(X))
    multiObj = partial(regulatedMultiFunction,obj, X)
    
    return (feasibilityNormalized,multiObj)


# In[1]:


def getPopulationByLabel(X,labels):
    '''
    Intro:
        Function to get population by labels
    ---
    
    Inputs:
        X : List
            List of all points
        
        labels : List
            List of binary labels
            
    ---
    
    Outputs:
        Returns points of X referenced by each label
    '''
    Xlb = []
    for m in labels:
        Xlb.append(X[binToInt(m)])
    
    return Xlb


# ## 3. Classes

# In[8]:


class solution:
    '''
    Introduction:
        Defining the solution class which will store
        the results
    '''
    def __init__(self):
        self.optimizer=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.maxiters=0
        self.pop = None


# ## 4. Pareto Simulated Annealing Operators

# In[9]:


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


# In[10]:


def mutate(x, p_bit, n_bits=None):
    '''
    Intro:
        Get a neighbor of x by mutation
    ---
    
    Input: 
        x : Iterable object (list, set, ...)
            Individual who we want to get the neighbor
        
        p_bit : Float
            Probability to change the bit
        
        n_bits : Integer
            Maximum number of bits to flip
    ---
    
    Output:
        A neighbor of x
    '''
    
    if n_bits is None:
        n_bits=len(x)
    
    y = []
    cout = 0
    for xi in x:
        if p_bit > random.uniform(0,1) and cout < n_bits:
            y.append((xi+1)%2)
            cout += 1
        else:
            y.append(xi)
    return y


# In[2]:


def initPop(s, dim, p, feasib_function):
    '''
    Intro:
        Create colony with s feasible solutions
    ---
    
    Input:
        s : Integer
            Number of solutions
        
        dim : Integer
            Dimension of each solution
            
        p : Float
            Probability of a certain number being one
            
        feasib_function : Function
            Function to check if our solution is good
    ---
    
    Output:
        Feasible solutions in a list
    '''
    S = []
    while len(S) < s:
        indv = []
        for i in range(dim):
            if p > random.uniform(0,1):
                indv.append(1)
            else:
                indv.append(0)
                
        # If point feasible
        if feasib_function(indv):
            S.append(indv)
    return S


# In[12]:


def updateParetoFront(obj,weights,paretoSet, point):
    '''
    Intro:
        Function that check if a given point
        belongs to the pareto front
    ---
    
    Input:
        obj : Function
            Function to map the points to the dimension
            where we want to compare them
        
        weights : Iterable object (list, set, ...)
            Weights of each dimension of the output
            They will will be used to maximize or
            minimize each feature
        
        paretoSet : List of points(iterable objects)
            Pareto set where we want to check the nez point
            
        point : Iterable object (list, set, ...)
            Point which we want to compare with the Pareto Front set 
    ---
    
    Output:
        The updated pareto set
    '''
    nonDominated = True
    i = 0
    while i < len(paretoSet):
        arch_point = paretoSet[i]
        if dominates(point, arch_point, obj, weights):
            nonDominated = False
        elif dominates(arch_point, point, obj, weights):
            paretoSet.remove(arch_point)
            continue
        i += 1
    if nonDominated:
        paretoSet.append(copy.deepcopy(point))
        
    return paretoSet


# In[13]:


def bestNeighbor(M,x):
    '''
    Intro:
        Function to find the best neighbor of x in the
        neighborhood
    ---
    
    Input:
        M : List of points
            The neighborhood
        
        x : Iterable object (list, set, ...)
            The point which we want the best neighbor
    ---
    Output:
        The best neighbor of x
    '''
    closestV = float('inf')
    point = None
    foundOne = False
    for xnb in M:
        if not x == xnb:
            h_distance = sum([abs(v1-v2) for v1,v2 in zip(x,xnb)])
            if h_distance < closestV:
                closestV = h_distance
                point = xnb
                foundOne = True
    if foundOne:
        return xnb
    else:
        return None


# ## 5. Pareto Simulated Annealing Algorithm

# In[1]:


def PSA(obj, weights,feasib_function, 
        X,K,s=10,p=0.5,b=0.93,a=1.1,iters=250,T=98.8,
        bit_flip=None,prob_bitflip=0.5, showprogress=True,
        dim=None):
    '''
    Intro:
        Pareto Simulated Annealing to binary multi-objective problems
    ---
    
    Inputs:
        obj : Function
            Function that we want to found the pareto front
        
        weights : Iterable object (list, set, ...)
            Weights of each dimension of the output
            They will will be used to maximize or
            minimize each feature
            
        feasib_function : Function
            Function used to check if a point belongs
            to the space of the input of our problem
            
        X : List of points
            List of our real points that we want to
            find the set of the best ones
            
        s : Integer
            Number of feasible points of pareto front
            
        K : Integer:
            Number of objectives
            
        p : Float
            Probability if choosing 1-bit when generating an
            initial feasible solution in S
            
        b : Float
            Temperature reduction factor (smaller and close to one)
        
        a : Float
            Weight modification factor (greater than 1)
            
        iters : Integer
            Max number of iterations
            
        T : Float
            Temperature value (the higher the slower the solutions
            will get to change)
            
        bit_flip : Integer
            How many bits to flip (at maximum) to get the neighbor
            If None, every bit has the possibility of be flipped
            
        prob_bitflip : Float
            Probability of flipping a bit
            
        dim : Integer
            Dimension of each feasible solution
        ---
        
        Output:
            Object of solution class with all informations
            about the process
    '''
    
    # Initialisation of vars
    w = np.zeros((s,K))
    if dim is None:
        dim = len(str(bin(len(X)))[2:]) # Case to find pareto by labels of real points
    
    # Initialisation of S
    S = initPop(s,dim,p,feasib_function)
    
    # solution set (i.e., set of all proposed efficient solutions
    # in current iteration)
    M = []
    
    sol=solution()
    
    timerStart=time.time() 
    sol.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # ==================================================================
    # 
    for point in S:
        if not feasib_function(point):
            continue
        M = updateParetoFront(obj,weights,M,point)
    
    # Repeat until criterion is true
    for l in range(iters):
        if showprogress:
            print("Iteration : ", l)
        for i in range(s):
            x = S[i]
            y = None
            # Construct a y feasible neighbor of x
            while True:
                y = mutate(x,prob_bitflip,bit_flip)
                if feasib_function(y):
                    break
            
            if dominates(x, y, obj, weights):
                M = updateParetoFront(obj,weights,M,y)
            xnb = bestNeighbor(M,x)
            
            if xnb == None or l == 0:
                w[i][:] = np.random.uniform(0,1,(1,K))
                w[i][:] =  w[i][:]/sum(w[i][:])
            else:
                for k in range(K):
                    if obj(x)[k]*weights[k] >= obj(xnb)[k]*weights[k]:
                        w[i][k] = a*w[i][k]
                    else:
                        w[i][k] = w[i][k]/a
                w[i][:] =  w[i][:]/sum(w[i][:])
            

            dotProduct = [wik*(ukx - uky) for wik,ukx,uky in zip(
                w[i][:].tolist(), list(obj(x)), list(obj(y)) )]
            
            try:
                expo = math.exp(sum(dotProduct)/T)
            except OverflowError:
                expo = 1
                
            if min(1, expo) > random.uniform(0,1):
                    S[i] = y
            
        T = b*T
    # ==================================================================
    timerEnd=time.time()  
    sol.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    sol.executionTime=timerEnd-timerStart
    sol.maxiters = iters
    sol.Algorithm="ParetoSimulatedAnnealing"
    sol.pop = M
    return sol

