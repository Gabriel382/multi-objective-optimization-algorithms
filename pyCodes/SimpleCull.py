#!/usr/bin/env python
# coding: utf-8

# # SimpleCull
# 
# > This code performs the Simple Cull algorithm for a group of given points.

# ## 1. Importing

# In[4]:


import random


# ## 2. Helpers

# In[5]:


def dominates(row, candidateRow, evalfunction, weights):
    '''
    Intro:
        Function that tests if one point dominates another.
    ---
    Input:
        row: List of numerical values
            Point that will be checked if it is being dominated.
        candidateRow: List of numerical values
            Point that will be checked if it's dominating.
        evalfunction: Function
            Multi-objective function that will take
            value lists to output space to check dominance.
        weights: Tuple of values
            Values that will balance (to maximize / minimize) 
            the values of each list position in the output space.
    ---
    Output:
        True if candidateRow dominates, false otherwise
    ''' 
    rrow = evalfunction(row)
    rcandidaterow = evalfunction(candidateRow)
    
    if len(rcandidaterow) != len(weights):
        raise Exception('Output of function different from number of weights')
    
    return sum([rrow[x]*weights[x] <= rcandidaterow[x]*weights[x] for x in range(len(rrow))]) == len(rrow)


# ## 3. Simple Cull

# In[1]:


def simple_cull(AllPoints, evalfunction, weights):
    '''
    Intro:
        Function that returns dominated and 
        non-dominated points (pareto front) 
        of all points
    ---
    Input:
        AllPoints: List of List
            All points (as list of numerical values)
        evalfunction: Function
            Multi-objective function that will take
            value lists to output space to check dominance.
        weights: Tuple of values
            Values that will balance (to maximize / minimize) 
            the values of each list position in the output space.
    ---
    Output:
        Returns dominated and 
        non-dominated points (pareto front) 
        of all points
    ''' 
    inputPoints = AllPoints.copy()
    
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()    
    
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row, evalfunction, weights):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow, evalfunction, weights):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
            
    return [list(x) for x in paretoPoints], [list(x) for x in dominatedPoints],

