#!/usr/bin/env python
# coding: utf-8

# # jupyterHelpers
# 
# > This code aims to group different codes and classes to efficiently manage genetic algorithms.

# ## 1. Importing

# ### 1.0 Classic Libs

# In[1]:


import matplotlib.pyplot as plt
from deap.benchmarks.tools import diversity, convergence
import sys


# In[2]:


# Appending system path to codes
sys.path.insert(0, 'pyCodes/')


# ### 1.1 DataGenerator And BenchmarkData

# In[3]:


import DataGenerator as DataGenerator
import BenchmarkData as bData


# ### 1.2 GeneralCases

# In[4]:


import GeneralCases as GeneralCases


# ### 1.3 NSGAII

# In[6]:


import NSGAII as nsga2


# ### 1.4 NSGAII

# In[8]:


import NSGAIII as nsga3


# ### 1.5 SPEA2

# In[9]:


import SPEA2 as spea2


# ### 1.6 GSA

# In[9]:


#import ipynb.fs.full.MethodGSA as gsa


# ### 1.7 Simple Cull

# In[10]:


import SimpleCull as sCull


# ### 1.8 Pareto Simulated Annealing

# In[11]:


import methodPSA as psa


# ### 1.9 Binary Tabu Search

# In[12]:


import BinaryTabuSearch as bTabuSearch


# ### 1.10 MOGSA 

# In[13]:


import methodMOGSA as mogsa


# ### 1.11 Full Non-Dominated Sorting and Ranking

# In[14]:


import FullNonDominatedSortingAndRanking as fullnsr


# ### 1.11 Search Parameter Manager

# In[16]:


import SearchParameterManager as searchManager


# ### 1.12 Fake Data Manager

# In[17]:


import FakeDataManager as fakeDataManager


# ### 1.13 MapManager

# In[18]:


import MapManager as mapManager


# ## 2. Measures

# In[18]:


def getBasicMeasures(pop, optimal_front):
    pop.sort(key=lambda x: x.fitness.values)
    convergenceValue = convergence(pop, optimal_front)
    diversityValue = diversity(pop, optimal_front[0], optimal_front[-1])
    print("Convergence: ", convergenceValue)
    print("Diversity: ", diversityValue)
    return (convergenceValue, diversityValue)


# In[19]:


def plotFrontgraphics(front, optimal_front, title = "",xlabel="x",ylabel="y"):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r",label='Real Pareto Front')
    plt.scatter(front[:,0], front[:,1], c="b",label='Pareto Front found')
    plt.legend(loc='upper right')
    plt.axis("tight")
    plt.show()


# ## 3. Helpers for other cases

# In[ ]:




