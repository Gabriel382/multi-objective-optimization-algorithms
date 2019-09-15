#!/usr/bin/env python
# coding: utf-8

# # GeneralCases
# 
# > This code has functions for general cases, mainly involving manipulations with the database.

# ## 1. Importing

# In[1]:


import math
import pandas as pd
import numpy as np
import os
import pygmo as pg


# ## 2. Functions

# In[1]:


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


# In[2]:


def distance(x):
    '''
    Intro:
        Function that calculates the distance between two points.
        
    ''' 
    d = 0
    for e in x:
        d += (e)**2
    return math.sqrt(d)


# In[3]:


def normalisedListOfList(X):
    '''
    Intro:
        Returns a normalized list
    ---
    Input:
        X: List of List
            List of Points
    ---
    Output:
        List of normalized points
    ''' 
    Xnp = np.array(X)
    biggestValues = Xnp.max(axis=0)
    for j in range(Xnp.shape[1]):
        Xnp[:,j] = Xnp[:,j]/biggestValues[j]
    return Xnp.tolist()


# In[4]:


def normalizedSorting(X):
    '''
    Intro:
        Sorts the points by ordering them using
        the distance between them and the origin,
        but only after normalization.
    ---
    Input:
        X: List of List
            List of Points.
    ---
    Output:
        The normalized list.
        
    ''' 
    Xnm = normalisedListOfList(X)
    Y = [distance(x) for x in Xnm]
    Xnew = [x for _,x in sorted(zip(Y,X))]
    return Xnew


# ## 3. Statistical tools

# In[20]:


def diversity(first_front, first, last):
    """Given a Pareto front `first_front` and the two extreme points of the 
    optimal Pareto front, this function returns a metric of the diversity 
    of the front as explained in the original NSGA-II article by K. Deb.
    The smaller the value is, the better the front is.
    """
    df = math.hypot(first_front[0][0] - first[0],
               first_front[0][1] - first[1])
    dl = math.hypot(first_front[-1][0] - last[0],
               first_front[-1][1] - last[1])
    dt = [math.hypot(first[0] - second[0],
                first[1] - second[1])
          for first, second in zip(first_front[:-1], first_front[1:])]

    if len(first_front) == 1:
        return df + dl

    dm = sum(dt)/len(dt)
    di = sum(abs(d_i - dm) for d_i in dt)
    delta = (df + dl + di)/(df + dl + len(dt) * dm )
    return delta

def convergence(first_front, optimal_front):
    """Given a Pareto front `first_front` and the optimal Pareto front, 
    this function returns a metric of convergence
    of the front as explained in the original NSGA-II article by K. Deb.
    The smaller the value is, the closer the front is to the optimal one.
    """
    distances = []
    
    for ind in first_front:
        distances.append(float("inf"))
        for opt_ind in optimal_front:
            dist = 0.
            for i in range(len(opt_ind)):
                dist += (ind[i] - opt_ind[i])**2
            if dist < distances[-1]:
                distances[-1] = dist
        distances[-1] = math.sqrt(distances[-1])
        
    return sum(distances) / len(distances)


# In[6]:


def hypervolume(front, ref):
    """
    Intro:
        Returns the front hypervolume
        relative to the reference point.
    ---
    Input:
        front: List of List
            List of Points.
        ref: List
            Lists the coordinates of the
            reference point.
    ---
    Output:
        The hypervolume value.
    """
    hv = pg.hypervolume(front)
    return hv.compute(ref, hv_algo=pg.hvwfg())


# ## 4. Save statistics

# In[38]:


def save_dict(dictionary, folderpath, filename):
    """
    Intro:
        Function that saves dictionary in csv
    ---
    Input:
        dictionary: Dict
            Dictionary of values to be saved (keys
            are the columns and key values are the
            values in the matrix)
        folderpath: String
            Path to folder where the
            file will be saved.
        filename: String
            Name of file to save
    ---
    Output:
        The hypervolume value.
    """
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        
    path = folderpath + "/" + filename
    df = pd.DataFrame(dictionary, columns=dictionary.keys(), index=[0])
    if not os.path.isfile(path):
        df.to_csv(path, header=dictionary.keys(), index=None)
    else:
        df.to_csv(path, mode="a", header=False, index=None)


# In[43]:


def open_csv(filepath):
    '''
    Intro:
        Recovers csv file
    ---
    Input:
        filepath: String
            Path to file
    ---
    Output:
        A DataFrame of the csv File
    '''
    if not os.path.isfile(filepath):
        raise Exception('This path does not exist: ' + filepath)
    else:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_colwidth', -1)
        df = pd.read_csv(filepath)
        display(df)

