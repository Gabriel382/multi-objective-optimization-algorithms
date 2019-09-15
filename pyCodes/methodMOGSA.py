#!/usr/bin/env python
# coding: utf-8

# # MOGSA - Multi-Objective Gravitational Search Algorithm
# 
# > This code performs genetic algorithms of the NSGAII type for the continuous or binary case.

# ## Summary
# 
# ### 1. [Importing libs](#Introduction)
# 
# ### 2. [Classes](#Classes)
# 
# ### 3. [Non-dominated functions](#Non-dominated)
# 
# ### 4. [Gravitational search functions](#GS_functions)
# 
# ### 5. [Continuous MOGSA helper functions](#Continuous_GS_functions)
# 
# ### 6. [Continuous MOGSA](#Continuous_MOGSA)
# 
# ### 7. [Binary MOGSA helper functions](#Binary_GS_functions)
# 
# ### 8. [Binary MOGSA](#Binary_MOGSA)

# ---
# <a id='Introduction'></a>
# ## 1. Importing libs

# In[1]:


import random
import numpy as np
import math
import time
from deap import benchmarks
import copy
from functools import partial


# ---
# <a id='Classes'></a>
# ## 2. Classes

# In[2]:


class solution:
    '''
    Introduction:
        Defining the solution class
    '''
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0
        self.pop = None
        self.archive = None


# In[3]:


class point:
    '''
    Intro:
        Class to represent the points
    
    '''
    def __init__(self, ind, mass, pos, vel, acc, obj, fit, rank, crowding):
        self.ind = ind
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.obj = obj
        self.fit = fit
        self.rank = rank
        self.crowding = crowding
        
    def calculateFit(self):
        self.fit = self.obj(self.pos)
    
    # Can change however you want
    def __lt__(self,other):
        return 1
  


# In[4]:


class cpoint(point):
    '''
    Intro:
        Class to represent the points in the continuous space
    
    '''
    def __init__(self, ind, mass, pos, vel, acc, obj, fit, rank, crowding):
        self.ind = ind
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.obj = obj
        self.fit = fit
        self.rank = rank
        self.crowding = crowding
        
        
    def distance(self, other_point):
        d = 0
        for r in range(len(self.fit)):
            d += (other_point.pos[r] -self.pos[r])**2
        d = math.sqrt(d)
        return d


# In[5]:


class bpoint(point):
    '''
    Intro:
        Class to represent the points in the binary space
    
    '''
    def __init__(self, ind, mass, pos, vel, acc, obj, fit, rank, crowding):
        self.ind = ind
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.obj = obj
        self.fit = fit
        self.rank = rank
        self.crowding = crowding
        
        
    def distance(self, other_point):
        # Hamming Distance
        d = 0
        for r in range(len(self.fit)):
            d += abs(self.fit[r] - other_point.fit[r])
        return d


# ---
# <a id='Non-dominated'></a>
# ## 3. Non-dominated functions

# In[6]:


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


# In[1]:


# Update Archive
def crowding_distance(archive):
    '''
    Intro:
        Calculates file crowding distance.
    ---
    Input:
        archive: List of individuals
            Front Point List
    ---
    Output:
        the crowding distance.
    '''  
    # Getting measures
    degree_of_output = len(archive[0].fit)
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
            output_values[j].append(archive[i].fit[j])
        
        # Reset crowding
        archive[i].crowding = 0
    archive[0].crowding = -float('inf')
    archive[-1].crowding = float('inf')
    
    # Sorting
    for i in range(degree_of_output):
        sorted_by_one_feature = [x for _,x in sorted(zip(output_values[i],position_archive))]
        sorted_positions_matrix.append(sorted_by_one_feature)
    
    # Finally calculating the crowding distance
    for i in range(0,len(sorted_positions_matrix)):
        for j in range(1,len_of_archive-1):
            ft_before = archive[sorted_positions_matrix[i][j-1]].fit[i]
            ft_after = archive[sorted_positions_matrix[i][j+1]].fit[i]
            archive[sorted_positions_matrix[i][j]].crowding += (ft_after - ft_before)**2
    
    for j in range(1,len_of_archive-1):
        archive[j].crowding = math.sqrt(archive[j].crowding)
    
    return archive
        
def spreadFactor(archive):
    '''
    Intro:
        Calculates spread factor to decise
        the best points of the archive
    ---
    Input:
        archive: List of individuals
            Front Point List
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
        mean += archive[i].crowding
        list_of_values.append(archive[i].crowding)
    mean = mean/len_of_archive
        
    # Calculating spread factor
    for i in range(len(list_of_values)):
        spread += abs(list_of_values[i]-mean)
        
    spread = spread/( (len_of_archive -len(archive[0].fit))*mean )
    return spread
    
def regulate_archive(old_archive,nmax_archive):
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
    old_archive = crowding_distance(old_archive)
    
    # Copping array
    archive = old_archive.copy()
    
    # Calculates distance of each point
    distance_matrix = np.full((len_of_archive, len_of_archive), np.inf)
    
    for i in range(0,len_of_archive):
        for j in range(i+1,len_of_archive):
            distance_matrix[i,j] = archive[i].distance(archive[j])
    
    
    while len(archive) > nmax_archive:
        all_positions = np.where(distance_matrix == np.min(distance_matrix))
        (i,j) = (all_positions[0][0],all_positions[1][0])
        distance_matrix[i,j] = float('inf')
        if((i == 0 and j == (len(archive)-1)) or
          (i == (len(archive)-1) and j == 0)):
            continue

        d1 = spreadFactor(archive[:i]+archive[i+1:])
        d2 = spreadFactor(archive[:j]+archive[j+1:])
        
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
    
    
    
    
def update_archive(pop, archive, nmax_archive, weights):
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
    for point in pop:
        nonDominated = True
        i = 0
        while i < len(archive):
            arch_point = archive[i]
            if dominates(point.pos, arch_point.pos, point.obj, weights):
                nonDominated = False
            elif dominates(arch_point.pos, point.pos, point.obj, weights):
                archive.remove(arch_point)
                continue
            i += 1
        if nonDominated:
            archive.append(copy.deepcopy(point))
    
    # Sorting results
    distance_array = [math.sqrt(sum([x**2 for x in p.fit])) for p in archive]
    archive = [x for _,x in sorted(zip(distance_array,archive))]
    
    # Checking size
    if len(archive) > nmax_archive:
        archive = regulate_archive(archive,nmax_archive)
        distance_array = [math.sqrt(sum([x**2 for x in p.fit])) for p in archive]
        archive = [x for _,x in sorted(zip(distance_array,archive))]
        
    return archive


# In[8]:


def non_dominated_sorting(pop,dim_of_output,weights,rank_start=3):
    '''
    Intro:
        This code performs a non-dominated sorting and ranking in
        the entire POPULATION so that we can have all points
        labeled by their ranking of (pareto) front.
    ---
    
    Input:
        pop : List of points
             List with all input points
        dim_of_output : Integer
            Size of the output.
        weights : Tuple
            Weight values of each output of the multi-objective function
        rank_start: Integer
            Rank of the first front (For the MOGSA, it starts at 3)
    ---
    
    Output:
        A Front List and Rank List
    
    '''
    # Dominate
    S=[[] for i in range(len(pop))]
    
    # Each Front
    front = [[]]
    
    # Dominated by
    n=[0 for i in range(len(pop))]
    
    # Rank
    rank = [0 for i in range(len(pop))]
    
    # Update dominations
    for p in range(0,len(pop)):
        S[p] = []
        n[p]=0
        for q in range(0,len(pop)):
            if pop[p].fit == pop[q].fit:
                continue
            elif dominates(pop[q].pos, pop[p].pos, pop[p].obj, weights):
                if q not in S[p]:
                    S[p].append(q)
            elif dominates(pop[p].pos, pop[q].pos, pop[p].obj, weights):
                n[p] += 1
        
        if n[p]==0:
            rank[p] = rank_start
            pop[p].rank = rank_start
            if p not in front[0]:
                front[0].append(p)
    
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if( n[q] == 0):
                    rank[q]=rank_start+i+1
                    pop[q].rank=rank_start+i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return (front,rank)


# ---
# <a id='GS_functions'></a>
# ## 4. MOGSA helper functions

# In[9]:


# Calculating Fitness
def calFitness(pop):
    '''
    Intro:
        Update fitness value of each individual
    ---
    
    Input:
        pop : List of points
             List with all points of the population
    ---
    
    Output:
        All points with updated fitness function
    
    '''
    for i in range(len(pop)):
        pop[i].calculateFit()
    return pop


# In[1]:


def updateListOfParticles(pop,archive,front,m,p_elistism,nmax_pop=None):
    '''
    Intro:
        Upgrades all point values so that after velocity and
        acceleration vectors can be calculated
    ---
    
    Input:
        pop : List of points
             List with all input points
        archive : List of points
            List of the best points
        front : List of lists
            All fronts of the population (calculated with the
            Non-dominated sorting and ranking algorithm)
        m: Integer
            Numbers of extreme points to get from the file to
            put in the population.
        p_elistism: Float
            Likelihood of adding other points of the file that
            are not at the extremes.
        nmax_pop: Integer or None
            Maximum number of individuals in the population. 
            If None, the number of individuals in the population
            will remain constant.
    ---
    Output:
        Updated population.
    
    '''
    # Getting max number of points in population
    if nmax_pop is None:
        nmax_pop = len(pop)
    
    # Get crowding distances
    archive = crowding_distance(archive)
        
    # Get m extremes
    extremes_archive =  copy.deepcopy(archive[:m//2]) + copy.deepcopy(archive[-m//2:])
    
    # Get m less crowded area
    all_crowding_dist = [p.crowding for p in archive[m//2:-m//2]]
    m_lesscrowded = [x for _,x in sorted(zip(all_crowding_dist,copy.deepcopy(archive[m//2:-m//2])))][:m]
    
    # Get the rest
    not_extremes_archive = copy.deepcopy(archive[m//2:-m//2])
    for p in m_lesscrowded:
        if p in not_extremes_archive:
            not_extremes_archive.remove(p)

    # Update ranks and velocities
    vel_dim = len(archive[0].vel)
    for p in extremes_archive:
        p.vel = [0]*vel_dim
        p.rank = 1
    for p in m_lesscrowded:
        p.vel = [0]*vel_dim
        p.rank = 1
    for p in not_extremes_archive:
        p.vel = [0]*vel_dim
        p.rank = 2
    
    pop += extremes_archive
    pop += m_lesscrowded
    
    for p in not_extremes_archive:
        if random.uniform(0,1) < p_elistism:
            pop.append(p)
    
    rank_order = [p.rank for p in pop]
    new_pop = [x for _,x in sorted(zip(rank_order,pop))][:nmax_pop]
    pop = new_pop
    
    return pop


# In[11]:


def gConstant(l,iters, G0=100):
    '''
    Intro:
        This functions gives G, the gravitational
        constant that will decrease along the time
    '''
    return G0*(1-l/iters)


# In[12]:


def massCalculation(pop):
    '''
    Introduction:
        Calculates the mass of the entire population
        
    ---
    Input:
        pop: List of points
            population to calculate mass
            
    ---
    Output:
        A numpy array with the mass of each member of the population
    '''
    # Get list of ranks
    fit = [p.rank for p in pop]
    
    # Calculates necessary values for further calculations
    Fmax = max(fit)
    Fmin = min(fit)
    
    # Check max and min to see with the denominator would be 0
    # In this case, we would have only persons with mass 1
    if Fmax == Fmin:
        for p in pop:
            p.mass = 1
    else:
        Msum=0
        # Brute mass
        for p in pop:
            mass = (Fmax-p.rank)/(Fmax-Fmin) + np.finfo(float).eps
            p.mass = mass
            Msum += mass
            
    # Normalized mass
    Msum=Msum/len(pop)
    for p in pop:
        p.mass = p.mass/Msum

    return pop


# ---
# <a id='Continuous_GS_functions'></a>
# ## 5. Continuous MOGSA helper functions

# In[13]:


# Population
def cinit_population(n,lb,ub,dim,obj):
    '''
    Intro:
        function to initialise the population
        for the continuous case.
    ---
    Input:
        n: Integer
            Number of individuals.
        lb: Float
            Lower limit of the value of each point position.
        ub: Float
            Upper limit of the value of each point position.
        dim: Integer
            Dimension of the individuals.
        obj: Funcion
            Multi-objectif function.
    ---
    Output:
        A randomly created population.
    '''
    pop = []
    for i in range(n):
        pos = np.random.uniform(0,1,(dim,))*(ub-lb) + lb
        one_point = cpoint(i, 0, pos.tolist(), [0.]*dim, [0.]*dim,
                          obj, 0., 0., 0.)
        pop.append(one_point)
    
    return pop


# In[14]:


def cgravitational_search_constant(x_lb,x_ub, B):
    '''
    Intro:
        This functions gives G0, the biggest gravitational
        constant that will decrease along the time.
    ---
    Input:
        x_lb: List of values
            Highest possible value for an individual.
        x_ub: List of values
            Lowest possible value for an individual.
        B: Float
            Constant to multiply the result.
    ---
    Output:
        The gravitational search constant for the continuous
        case.
    '''
    if len(x_lb) != len(x_ub):
        raise Exception('Inferior limit and Superior limit with different sizes')
    
    biggest_value = -float('inf')
    for i in range(len(x_lb)):
        x = abs(x_ub[i]-x_lb[i])
        if x > biggest_value:
            biggest_value = x
    
    return B*biggest_value


# In[15]:


def cgField(pop,l,iters,G,ElitistCheck,Rpower, randomseed=None):
    '''
    Introduction:
        Calculates the Force and acceleration
        
    ---
    Input:
        pop : List of points
            Contains each point of the population
            
        l : Integer
            Number of the current iteration
            
        iters : Integer
            Max number of iteration
            
        G : Integer
            Gavitational constant
            
        ElitistCheck : Integer
            If 1, change the number of persons that interacts with others by time
            Otherwise, everyone interacts with everyone
            
        Rpower : Integer
            The power of R (distant between two points)
        
        randomseed : Integer
            The seed for the random process
    ---
    Output:
        A DimXPopSize-Dimensional numpy array with the acceleration of each member of the population
    '''
    # Set random seed
    PopSize = len(pop)
    random.seed(randomseed)
    
    # Set elitism
    final_per = 2
    if ElitistCheck == 1:
        kbest = final_per + (1-l/iters)*(100-final_per)
        kbest = round(PopSize*kbest/100)
    else:
        kbest = PopSize
            
    # Get index of kbest
    kbest = int(kbest)
    ds = sorted(range(len(pop)), key=lambda k: pop[k].mass,reverse=True)
    
    #Getting force
    Force = np.zeros((len(pop),len(pop[0].pos)))
    
    for r in range(len(pop)):
        for ii in range(kbest):
            z = ds[ii]
            R = 0
            if z != r:
                # Get position of two points
                x=pop[r].pos
                y=pop[z].pos
                esum=0
                imval = 0
                # Calculate distance
                for t in range(len(pop[0].pos)):
                    imval = ((x[t] - y[t])** 2)
                    esum = esum + imval
                    
                R = math.sqrt(esum)
                
                for k in range(len(pop[0].pos)):
                    randnum=random.random()
                    Force[r,k] = Force[r,k]+randnum*(pop[z].mass)*(
                        (y[k]-x[k])/(R**Rpower+np.finfo(float).eps))
    
    for x in range(len(pop)):
        for y in range (len(pop[0].pos)):
            pop[x].acc[y]=Force[x,y]*G
    return pop


# In[16]:


def cmove(pop,lb,ub, ps_m, pr_m,w0,w1,l,iters, randomseed=None):
    '''
    Introduction:
        Defines the move Function for calculating the updated position of
        each member of population.

    ---
    Input:
        pop : List of points
            Contains each point of the population.
        
        lb : Float
            Lower limit value of an axis of position.
        
        ub : Float
            Upper limit value of an axis of position.
            
        ps_m : Float
            Sign Mutation probability for each value
            of each individual.
            
        pr_m : Float
            Reordering Mutation probability for each.
            individual.
            
        w0 and w1 : Float
            Speed related constants.
            
        l : Integer
            Current Iteration.
            
        iters: Integer
            Max number of iterations.
            
        randomseed : Integer
            Seed for the random process.
    ---
    Output:
        Two DimXPopSize-Dimensional numpy array with the position
        and velocity of each member of the population.
    '''
    
    random.seed(randomseed)
        
    for i in range(len(pop)):
        vel_temp = copy.deepcopy(pop[i].vel)
        
        # Calculating the velocities with mutation
        for j in range (len(pop[0].pos)):
            # weighting coefficient 
            r1=w0 - (w0-w1)*(l/iters)
            pop[i].vel[j] = r1*pop[i].vel[j] + pop[i].acc[j]
            
            vel_temp[j] = pop[i].vel[j]
            # Sign Mutation
            if random.uniform(0,1) < ps_m:
                vel_temp[j] = pop[i].vel[j]*(-1)
        
        # Reordering Mutation
        if random.uniform(0,1) < pr_m:
            vel_temp = random.sample(vel_temp, len(vel_temp))
        
        
        # Calculating the velocities
        for j in range (len(pop[0].pos)):
            pop[i].pos[j] += vel_temp[j]
            # Putting within the limits
            if pop[i].pos[j] > ub:
                pop[i].pos[j] = ub
            elif pop[i].pos[j] < lb:
                pop[i].pos[j] = lb

    return pop


# ---
# <a id='Continuous_MOGSA'></a>
# ## 6. Continuous MOGSA

# In[17]:


def Continuous_MOGSA(objf, lb, ub ,dim, PopSize, iters, weights,
        n_archive, p_elistim=0.5, ElitistCheck = 1,
        ps_m=0.2, pr_m=0.1, w0=0.9, w1=0.5,
        Rpower = 1, B=2.5, randomseed = None,showprogress = True):
    '''
    Introduction:
        Defines the Gravitational Search Algorithm(GSA) Function for minimizing the Objective Function

    ---
    Input:
        objf : Function
            Function to minimize

        lb : Float
            lower limit

        ub : Float
            upper limit

        dim : Integer
            Number of dimensions

        PopSize : Integer
            Number of persons in the population

        Iters : Integer
            Max number of iterations

        ElististCheck : Integer
            If 1, change the number of persons that interacts with others by time
            Otherwise, everyone interacts with everyone

        Rpower : Integer
            The power of R (distance between to persons)

    ---
    Output:
        The best solution of the type solution
    '''

    random.seed(randomseed)

    s=solution()

    # Initializations
    pop = cinit_population(PopSize,lb,ub,dim,objf)
    archive = []
    
    len_of_output = len(objf(np.zeros((dim,))))
  
    convergence_curve=np.zeros(iters)
    
    if showprogress:
        print("============= MOGSA is optimizing  \""+objf.__name__+"\" ============= ")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    G0 = cgravitational_search_constant([lb]*dim,[ub]*dim,B)
    
    for l in range(0,iters):
        # ==========================================================================
        if showprogress:
            print('Iteration ',l)
        
        # Calculating Fitness
        pop = calFitness(pop)
        
        
        # Update archive
        archive = update_archive(pop,archive,n_archive,weights)
        
        
        # Non-dominated-Sorting
        (front, rank) = non_dominated_sorting(pop,len_of_output,weights)
        # Update the list of moving particles
        pop = updateListOfParticles(pop,archive,front,len_of_output,p_elistim)

        # Calculating Mass
        pop = massCalculation(pop)

        # Calculating Gfield        
        G = gConstant(l,iters, G0)
        
        # Calculating Gfield
        pop = cgField(pop,l,iters,G, ElitistCheck = 1, Rpower = 1, randomseed=randomseed)

        # Calculating Position
        pop = cmove(pop,lb,ub, ps_m, pr_m,w0,w1,l,iters)

        # ==========================================================================

    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.Algorithm="Continuous MOGSA"
    s.objectivefunc=objf.__name__
    s.pop = pop
    s.archive = archive

    return s


# ---
# <a id='Binary_GS_functions'></a>
# ## 7. Binary MOGSA helper functions
# 

# In[3]:


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

def feasibility_labelcase(X_size, x):
    '''
    Intro:
        Default feasibility function to check if
        x belongs to the boundaries of the binary
        labels of X
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

def regulatedMultiFunction(obj, X, x):
    '''
    Intro:
        Function that put the multi-objective function
        in the default way necessary to the PSA function
    ---
    
    Inputs:
        x : cpoint
            Binary point which we want to check
        
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
    
    feasibilityNormalized = partial(feasibility_labelcase,len(X))
    multiObj = partial(regulatedMultiFunction,obj, X)
    
    return (feasibilityNormalized,multiObj)


# In[2]:


# Population
def binit_population(n,dim,obj,feasibility):
    '''
    Intro:
        function to initialise the population
    ---
    
    Inputs:
        n : Integer
            Number of individuals in the population.
            
        dim : Integer
            Size of each individual.
            
        obj : Function
            Function that we want to find the pareto
            front.
        
        feasibility : Function
            Function that tests if individual belongs
            to the limits of the input space.
            
    ---
    
    Outputs:
        Returns randomly created population.
    '''
    pop = []
    
    i = 0
    while i < n:
        pos = np.random.randint(0,2,(dim,))
        one_point = bpoint(i, 0, pos.tolist(), [0.]*dim, [0.]*dim,
                          obj, 0., 0., 0.)
        
        if feasibility(one_point.pos):
            pop.append(one_point)
            i += 1
    
    return pop


# In[20]:


def bgravitational_search_constant(B):
    '''
    Intro:
        This functions gives G0, the biggest gravitational
        constant that will decrease along the time
    '''
    return B*(1-0)


# In[21]:


def bgField(pop,l,iters,G,ElitistCheck,Rpower, randomseed=None):
    '''
    Introduction:
        Calculates the Force and acceleration
        
    ---
    Input:
        pop : List of points
            Contains each point of the population
            
        l : Integer
            Number of the current iteration
            
        iters : Integer
            Max number of iteration
            
        G : Integer
            Gavitational constant
            
        ElitistCheck : Integer
            If 1, change the number of persons that interacts with others by time
            Otherwise, everyone interacts with everyone
            
        Rpower : Integer
            The power of R (distant between two points)
    ---
    Output:
        A DimXPopSize-Dimensional numpy array with the acceleration of each member of the population
    '''
    PopSize = len(pop)
    
    # Set random seed
    random.seed(randomseed)
    
    # Set elitism
    final_per = 2
    if ElitistCheck == 1:
        kbest = final_per + (1-l/iters)*(100-final_per)
        kbest = round(PopSize*kbest/100)
    else:
        kbest = PopSize
            
    # Get index of kbest
    kbest = int(kbest)
    ds = sorted(range(len(pop)), key=lambda k: pop[k].mass,reverse=True)
    
    #Getting force
    Force = np.zeros((len(pop),len(pop[0].pos)))
    
    for r in range(len(pop)):
        for ii in range(kbest):
            z = ds[ii]
            R = 0
            if z != r:
                # Get position of two points
                x=pop[r].pos
                y=pop[z].pos
                R=0
                # Calculate distance
                for t in range(len(pop[0].pos)):
                    R += abs(x[t] - y[t])
                # Calculate force
                for k in range(len(pop[0].pos)):
                    randnum=random.random()
                    Force[r,k] = Force[r,k]+randnum*(pop[z].mass)*(
                        round(y[k]-x[k])/(R**Rpower+np.finfo(float).eps))
    
    for x in range(len(pop)):
        for y in range (len(pop[0].pos)):
            pop[x].acc[y]=Force[x,y]*G
    return pop


# In[33]:


def bmove(pop, ps_m, pr_m,w0,w1,l,iters, feasibility, randomseed=None,vmax=None,flipProb=None):
    '''
    Introduction:
        Defines the move Function for calculating the updated position of
        each member of population

    ---
    Input:
        pop : List of points
            Contains each point of the population
        ps_m : Float
            Sign Mutation probability for each value.
            of each individual.
            
        pr_m : Float
            Reordering Mutation probability for each.
            individual.
            
        w0 and w1 : Float
            Speed related constants.
            
        l : Integer
            Current Iteration.
            
        iters: Integer
            Max number of iterations.
        
        feasibility : Function
            Function that tests if individual belongs
            to the limits of the input space.
            
        randomseed : Integer
            Seed for the random process.
            
        vmax : Float
            The maximum speed the point can reach 
            (if None, there will be no maximum speed)
            
        flipProb : Float
            Likelihood of changing the bit value of
            each bit of all individuals.
    ---
    Output:
        Updated population.
    '''
    
    
    random.seed(randomseed)
    
    for i in range(len(pop)):
        # Create copy of point at last position
        old_point = copy.deepcopy(pop[i])
        vel_temp = copy.deepcopy(pop[i].vel)
        
        # Calculating the velocities with mutation
        for j in range (len(pop[0].pos)):
            # weighting coefficient 
            r1=w0 - (w0-w1)*(l/iters)
            pop[i].vel[j] = r1*pop[i].vel[j] + pop[i].acc[j]
            
            vel_temp[j] = pop[i].vel[j]
            # Sign Mutation - Not necessary, just to comparation
            if random.uniform(0,1) < ps_m:
                vel_temp[j] = pop[i].vel[j]*(-1)
        
        # Reordering Mutation
        if random.uniform(0,1) < pr_m:
            vel_temp = random.sample(vel_temp, len(vel_temp))
        
        # Calculating the velocities
        for j in range (len(pop[0].pos)):
            sv = vel_temp[j]
            if vmax is not None:
                sv = np.sign(sv)*min(abs(sv),vmax)
            sv = abs(math.tanh(sv))
            
            if random.uniform(0,1) < sv:
                pop[i].pos[j] = int((pop[i].pos[j]+1)%2)
                
            if flipProb is not None:
                if random.uniform(0,1) < flipProb:
                    pop[i].pos[j] = int((pop[i].pos[j]+1)%2)
                    
        if not feasibility(pop[i].pos):
            pop[i] = old_point
    
    return pop


# ---
# <a id='Binary_MOGSA'></a>
# ## 8. Binary MOGSA

# In[23]:


def Binary_MOGSA(objf , feasibility, dim, PopSize, iters, weights,
        n_archive, p_elistim=0.5, ElitistCheck = 1,
        ps_m=0.2, pr_m=0.1, w0=0.9, w1=0.5, flipProb = 0.01,vmax=None,
        Rpower = 1, B=2.5, randomseed = None, showprogress = True):
    '''
    Introduction:
        Defines the Binary Gravitational Search Algorithm(GSA) Function for minimizing the Objective Function

    ---
    Input:
        objf : Function
            Function to minimize
            
        feasibility : Function
            Must be a function that receives only a list (or iterable object)
            and returns true if the solution is feasible

        lb : Float
            lower limit

        ub : Float
            upper limit

        dim : Integer
            Number of dimensions

        PopSize : Integer
            Number of persons in the population

        Iters : Integer
            Max number of iterations

        ElististCheck : Integer
            If 1, change the number of persons that interacts with others by time
            Otherwise, everyone interacts with everyone

        Rpower : Integer
            The power of R (distance between to persons)

    ---
    Output:
        The best solution of the type solution
    '''

    random.seed(randomseed)

    s=solution()

    # Initializations
    pop = binit_population(PopSize,dim,objf,feasibility)
    archive = []
    
    len_of_output = len(objf(np.zeros((dim,))))
  
    convergence_curve=np.zeros(iters)

    if showprogress:
        print("============= Binary MOGSA is optimizing ==============")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    G0 = bgravitational_search_constant(B)
    
    
    for l in range(0,iters):
        # ==========================================================================
        if showprogress:
            print('Iteration ',l)
        
        # Calculating Fitness
        pop = calFitness(pop)
        
        
        # Update archive
        archive = update_archive(pop,archive,n_archive,weights)
        
        # Non-dominated-Sorting
        (front, rank) = non_dominated_sorting(pop,len_of_output,weights)
        # Update the list of moving particles
        pop = updateListOfParticles(pop,archive,front,len_of_output,p_elistim)

        # Calculating Mass
        pop = massCalculation(pop)

        # Calculating Gfield        
        G = gConstant(l,iters, G0)
        
        # Calculating Gfield
        pop = bgField(pop,l,iters,G, ElitistCheck = 1, Rpower = 1, randomseed=randomseed)

        # Calculating Position
        pop = bmove(pop, ps_m, pr_m,w0,w1,l,iters,feasibility=feasibility,flipProb=flipProb)

        # ==========================================================================
    # Last update
    pop = calFitness(pop)
    archive = update_archive(pop,archive,n_archive,weights)
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.Algorithm="Binary MOGSA"
    s.objectivefunc="Multi_Objective"
    s.pop = pop
    s.archive = archive

    return s

