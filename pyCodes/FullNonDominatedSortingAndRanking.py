#!/usr/bin/env python
# coding: utf-8

# # Full Non-Dominated Sorting and Ranking
# > This code performs a non-dominated sorting and ranking in the entire dataset so that we can have all points
# labeled by their ranking of (pareto) front.

# ## 1. Importing

# In[70]:


import time
import pandas as pd
import random
from deap import benchmarks


# In[85]:


import copy
import numpy as np
import math


# ## 2. Helpers for our Helpers

# In[71]:


def distance(x,y):
    '''
    Intro:
        Function that calculates the distance between two points.
        
    ''' 
    d = 0
    for r in range(len(x)):
        d += (y[r] -x[r])**2
    d = math.sqrt(d)
    return d


# In[72]:


# Check Domination
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


# In[112]:


# Update Archive
def crowding_distance(archive,evalfunction):
    '''
    Intro:
        Calculates file crowding distance.
    ---
    Input:
        archive: List of individuals
            Front Point List
        evalfunction: Function
            Multiobjectif function being evaluated
    ---
    Output:
        the crowding distance.
    ''' 
    # Getting measures
    degree_of_output = len(evalfunction(archive[0]))
    len_of_archive = len(archive)
    
    # Setting sorted positions by function value
    sorted_positions_matrix = []
    
    # Setting pos and all output values
    position_archive = [int(x) for x in range(len(archive))]
    output_values = []
    
    # Getting all outputs from points
    for i in range(len(archive)):
        # Get output
        for j in range(degree_of_output):
            # First person
            if i == 0:
                output_values.append([])
            # Append feature j of person i
            output_values[j].append(evalfunction(archive[i])[j])
        
    # Crowding list
    crowding = [0]*len_of_archive
    crowding[0] = -float('inf')
    crowding[-1] = float('inf')
    
    # Sorting
    for i in range(degree_of_output):
        sorted_by_one_feature = [x for _,x in sorted(zip(output_values[i],position_archive))]
        sorted_positions_matrix.append(sorted_by_one_feature)
    
    # Finally calculating the crowding distance
    for i in range(0,len(sorted_positions_matrix)):
        for j in range(1,len_of_archive-1):
            ft_before = evalfunction(archive[sorted_positions_matrix[i][j-1]])[i]
            ft_after = evalfunction(archive[sorted_positions_matrix[i][j+1]])[i]
            crowding[sorted_positions_matrix[i][j]] += (ft_after - ft_before)**2
    
    for j in range(1,len_of_archive-1):
        crowding[j] = math.sqrt(crowding[j])
    
    return crowding
        
def spreadFactor(archive,crowdingDist,evalfunction):
    '''
    Intro:
        Calculates spread factor to decise
        the best points of the archive
    ---
    Input:
        archive: List of individuals
            Front Point List
        crowdingDist: Float
            The crowding distance of the archive
        evalfunction: Function
            Multiobjectif function being evaluated
    ---
    Output:
        The spread factor.
    ''' 
    spread = 0
    list_of_values = []
    len_of_archive = len(archive)
    mean = 0
    
    # Getting mean and list of values
    for i in range(1,len_of_archive-1):
        mean += crowdingDist[i]
        list_of_values.append(crowdingDist[i])
    mean = mean/len_of_archive
        
    # Calculating spread factor
    for i in range(len(list_of_values)):
        spread += abs(list_of_values[i]-mean)
        
    spread = spread/( (len_of_archive -len(evalfunction(archive[0])))*mean )
    return spread
    
def regulate_archive(old_archive,nmax_archive,evalfunction):
    '''
    Intro:
        Chop-down the size of the array
    ---
    Input:
        old_archive: List of individuals
            Old front point List
        nmax_archive: Float
            Maximum number of points in new list
        evalfunction: Function
            Multiobjectif function being evaluated
    ---
    Output:
        A file with fewer points but the most relevant
        ones from the previous list
    ''' 
    # Measures
    len_of_archive = len(old_archive)
    
    # Calculates crowding distance
    crowdingDist = crowding_distance(old_archive,evalfunction)
    
    # Copping array
    archive = old_archive.copy()
    
    # Calculates distance of each point
    distance_matrix = np.full((len_of_archive, len_of_archive), np.inf)
    
    for i in range(0,len_of_archive):
        for j in range(i+1,len_of_archive):
            distance_matrix[i,j] = distance(evalfunction(archive[i]),evalfunction(archive[j]))
    
    
    while len(archive) > nmax_archive:
        all_positions = np.where(distance_matrix == np.min(distance_matrix))
        (i,j) = (all_positions[0][0],all_positions[1][0])
        distance_matrix[i,j] = float('inf')
        if((i == 0 and j == (len(archive)-1)) or
          (i == (len(archive)-1) and j == 0)):
            continue

        d1 = spreadFactor(archive[:i]+archive[i+1:],crowdingDist,evalfunction)
        d2 = spreadFactor(archive[:j]+archive[j+1:],crowdingDist,evalfunction)
        
        if (i == 0 and d1 > d2) or (i == (len(archive)-1) and d1 > d2):
            distance_matrix[j,:] = float('inf')
            distance_matrix[:,j] = float('inf')
            el = old_archive[j]
            archive.remove(el)
        elif(j == 0 and d1 < d2) or (j == (len(archive)-1) and d1 < d2):
            distance_matrix[i,:] = float('inf')
            distance_matrix[:,i] = float('inf')
            el = old_archive[i]
            archive.remove(el)
        elif d1 > d2:
            distance_matrix[i,:] = float('inf')
            distance_matrix[:,i] = float('inf')
            el = old_archive[i]
            archive.remove(el)
        else:
            distance_matrix[j,:] = float('inf')
            distance_matrix[:,j] = float('inf')
            el = old_archive[j]
            archive.remove(el)
            
    return archive
    
    
    
    
def smart_removal(pop, nmax_points, evalfunction, weights):
    '''
    Intro:
        Uptade archive with pop 
        (if population larger than the limit).
    ---
    Input:
        pop: List of individuals
            Points list.
        nmax_archive: Float
            Maximum number of points in new list
        evalfunction: Function
            Multiobjectif function being evaluated
        weights: Tuple of values
            Values that will balance (to maximize / minimize) 
            the values of each list position in the output space.
    ---
    Output:
        If population larger than the limit, it will return
        a file with fewer points but the most relevant ones 
        from the previous list.
    ''' 
    archive = copy.deepcopy(pop)
    
    if nmax_points > len(pop):
        return archive
    
    # Sorting results
    distance_array = [math.sqrt(sum([x**2 for x in evalfunction(p)])) for p in archive]
    archive = [x for _,x in sorted(zip(distance_array,archive))]
    
    # Checking size
    if len(archive) > nmax_points:
        archive = regulate_archive(archive,nmax_points,evalfunction)
        distance_array = [math.sqrt(sum([x**2 for x in evalfunction(p)])) for p in archive]
        archive = [x for _,x in sorted(zip(distance_array,archive))]
        
    return archive


# # 3. Code

# In[74]:


def fnon_dominated_sorting_and_ranking(X,evalFunction, weights,rank_start=1):
    '''
    Intro:
        This code performs a non-dominated sorting and ranking in
        the entire dataset so that we can have all points
        labeled by their ranking of (pareto) front.
    ---
    
    Input:
        X : List
             List with all input points
        evalFunction : Function
            Multi-objective function that we want to optimize
        weights : Tuple
            Weight values of each output of the multi-objective function
        rank_start : Integer
            First Front Rank
    ---
    
    Output:
        A dictionary that has as key the rank of the layer and a list of the points (only
        the integers that indicate the rows of X) of the respective rank as value.
    
    '''
    sol = {}
    
    # Dominate
    S=[[] for i in range(len(X))]
    
    # Each Front
    front = [[]]
    
    # Dominated by
    n=[0 for i in range(len(X))]
    
    # Rank
    rank = [0 for i in range(len(X))]
    
    # Update dominations
    for p in range(0,len(X)):
        S[p] = []
        n[p]=0
        for q in range(0,len(X)):
            if evalFunction(X[p]) == evalFunction(X[q]):
                continue
            elif dominates(X[q], X[p], evalFunction, weights):
                if q not in S[p]:
                    S[p].append(q)
            elif dominates(X[p], X[q], evalFunction, weights):
                n[p] += 1
        
        if n[p]==0:
            rank[p] = rank_start
            if p not in front[0]:
                front[0].append(p)
    
    sol[rank_start] = front[0]
    
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if( n[q] == 0):
                    rank[q]=rank_start+i+1
                    if q not in Q:
                        Q.append(q)
                        
        i = i+1
        front.append(Q)
        if len(Q) != 0:
            sol[rank_start+i] = Q
        
    
    del front[len(front)-1]
    return sol

