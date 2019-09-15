#!/usr/bin/env python
# coding: utf-8

# # NSGAIII
# 
# > This code performs genetic algorithms of the NSGAIII type for the continuous or binary case.

# ## 1. Importing

# In[2]:


from math import factorial
import random
import matplotlib.pyplot as plt
import numpy as np
import pymop.factory
import array

from deap import benchmarks
from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools


# ## 2. Helpers for our Helpers

# In[ ]:





# ## 3. Continuous Case

# In[5]:


def create_Environment_For_NSGAIII_Continuous_Case(BOUND_LOW, BOUND_UP, NDIM,
                                                 attr_individual_function, evaluate_function, weights,
                                                 etaMate = 30.0,etaMutate = 20.0, indpb=None,
                                                 NOBJ=2, P=12):
    '''
    Intro:
        This function returns an environment to use
        in the NSGAIII_Continuous_Case.
    ---
    Input:
        BOUND_LOW: Float
            Lower limit of each value of individual.
        BOUND_UP: Float
            Uper limit of each value of individual.
        NDIM: Integer
            Number of dimensions of each individual.
        attr_individual_function: Funtion
            Function to create individual.
        evaluate_function: Function
            Function to evalulate our points.
        weights: Tuple
            Weights for each output of the evaluate_function
        etaMate: Float
            Crowding degree of the crossover. A high eta will 
            produce children resembling to their parents, 
            while a small eta will produce solutions much more different.
        etaMutate: Float
            Crowding degree of the mutate. A high eta will 
            produce mutations resembling to their parents, 
            while a small eta will produce solutions much more different.
        indpb: Float
            Probability of mutation
        NOBJ: Integer
            Number of objectives
        P: Integer
            Number of partitions
    ---
    Output:
        A tuple with (toolbox, stats, logbook, refpoints)
        
    ''' 
    
    if indpb is None:
        indpb = 1.0/NDIM
    
    # Create uniform reference point
    ref_points = tools.uniform_reference_points(NOBJ, P)

    # Create classes
    creator.create("FitnessCustom", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessCustom)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", attr_individual_function, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    
    toolbox.register("evaluate", evaluate_function)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=etaMate)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=etaMutate, indpb=indpb)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    
    # Creating stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Creating log book
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    return (toolbox, stats, logbook, ref_points)


# In[6]:


def NSGAIII_Continuous_Case(envrironment, NGEN=400, MU=100, CXPB=0.9, MUTPB=1.0, seed=None,showprogress = True):
    '''
    Intro:
        This function returns an the best population and some
        statistics of a pareto front.
        
    ---
    Input:
        envrironment: Tuple
            A tuple with (toolbox, stats, logbook).
        NGEN: Integer
            Number of generations.
        MU: Integer
            Number of instances.
        CXPB: Float
            Cross over probability.
        MUTPB: Float
            Probability to allows or not one alteration of the
            population by mutation
        seed: Integer
            Set function to get same results.
        showprogress: Boolean
            If we want to show the progress of each generation.
            
    ---
    Output:
        A tuple with (deap.toolbox.population, deap.Logbook)
        
    '''
    
    if MU%4 != 0:
        raise Exception('MU should not be a multiple of 4.')

    random.seed(seed)
        
    (toolbox, stats, logbook, _) = envrironment
    
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if showprogress:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if showprogress:
            print(logbook.stream)

    return pop, logbook


# ## 4. Discrete Case

# ### 4.1 Helpers for Discrete Case

# In[9]:


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

def intToIndividual(cindividual, bL, n):
    '''
    Intro:
        Transforms a number in a individual for the population.
    ---
    
    Input:
        cindividual: deap.toolbox.individual
            Function to transform list in individual.
        bL: Integer
            Size of binary number in the output.
        n: Integer
            The integer that we want to binarize
    ---
    Output:
        A individual for a discrete case of NSGAIII.
        
    ''' 
    nb = ('{:0' + str(bL) +'b}').format(n)
    return cindividual([int(x) for x in nb])

def getPop(toolindv, n_database, n_points, random_labels=False, seed=None):
    '''
    Intro:
        Create a list of labels for each entry of the matrix of real values
        and creates a individual over each label to feed the NSGAIII.
    ---
    
    Input:
        cindividual: deap.toolbox.individual
            Function to transform value in individual.
        n_database: Integer
            Size of our database.
        n_points: Integer
            Size of the output database of individuals.
        random_labels: Boolean
            
            
    ---
    Output:
        A List of individuals for a discrete NSGAIII case to
        create a population
    ''' 
    random.seed(seed)
    pop = []
    
    # Include points linearly
    if random_labels == False:
        if n_points < n_database:
            raise Exception('n_points less than number of elements of the database with random labels False')
        
        counter = n_points
        while(True):
            for i in range(n_database):
                pop.append(toolindv(i))
                counter -= 1
                if counter <= 0:
                    break
            if counter <= 0:
                    break

        return pop
    
    # Include points randomly
    else:
        for i in range(n_points):
            pop.append(toolindv(random.randint(0,n_database-1)))
        return pop

    
def evaluate(evfunction, X, individual):
    '''
    Intro:
        Function to evaluate a single individual. The function
        gets the binary label of the individual, transforms in
        integer and then get the real value in X. After this,
        we use evfunction to evaluate.
    ---
    
    Input:
        evfunction: function
            Function to evaluate.
        X: List of real values
            Ceil value.
        Individual: deap.toolbox.individual
            The Individual that we want to measure.   
    ---
    Output:
        The value of the evaluation of the individual by
        our function.
    ''' 
    out = binToInt(individual)
    return evfunction(X[out])


def ceilling(createindiv, maxvalue, individual):
    '''
    Intro:
        Check if individual have value bigger than the
        ceil value, and then returns ceil if it is the
        case.
    ---
    
    Input:
        createindiv: deap.toolbox.individual
            Function to transform value in individual.
        maxvalue: Integer
            Ceil value.
        Individual: deap.toolbox.individual
            An Individual.
            
            
    ---
    Output:
        The same individual if value less than maxvalue,
        otherwise a individual with ceiling value
    ''' 
    if binToInt(individual) > maxvalue:
        individual = createindiv(maxvalue)
    return individual


# ### 4.2 Discrete Case Code

# In[11]:


def create_Environment_For_NSGAIII_Discrete_Case(X, bss,
                                                weights, evaluate_function,
                                                random_labels=False,
                                                indpb = None, seed=None,
                                                NOBJ=2, P=12):
    '''
    Intro:
        This function returns an environment to use
        in the NSGAIII_Discrete_Case.
        
    ---
    Input:
        X: List
            List of real values
        bss: Integer
            Number of bits for each individual.
        weights: Tuple of integers
            Uper limit of each value of individual.
        indpb: Integer
            Weights for each output of the evaluate_function.
        evaluate_function: Function
            Function to evaluate individuals.
        random_labels: Boolean
            If we want to get randomly or linearly the points of X.
            
    ---
    Output:
        A tuple with (toolbox, logbook)
        
    '''
    
    random.seed(seed)
    
    if indpb is None:
        indpb = 1.0/(bss*10)

    # Create uniform reference point
    ref_points = tools.uniform_reference_points(NOBJ, P)
        
    # Creating individuals
    creator.create("FitnessCustom", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessCustom)

    # Creating toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual",
                 intToIndividual,
                 creator.Individual,
                 bss)
    toolbox.register("population", getPop, toolbox.individual, len(X),
                     random_labels=random_labels,seed=seed)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    
    # Defining ceil
    toolbox.register("ceiling",
                 ceilling,
                 toolbox.individual,
                 len(X)-1)
    
    # Defining evaluate function
    toolbox.register("evaluate", evaluate, evaluate_function, X)
    
    # Creating stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Creating log book
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    return (toolbox, stats, logbook, ref_points)


# In[12]:


def NSGAIII_Discrete_Case(envrironment, NGEN=100, MU=200, CXPB=0.93,MUTPB=1.0, seed=None, showprogress = True):
    '''
    Intro:
        This function returns an the best population and some
        statistics of a pareto front for a discrete case.
    
    Attention:
        the toolbox envrironment must have a function ceiling to
        put the values that are of the limits of the dataset back
        in the limits.
    ---
    Input:
        envrironment: Tuple
            A tuple with (toolbox, stats, logbook).
        NGEN: Integer
            Number of generations.
        MU: Integer
            Number of instances.
        CXPB: Float
            Cross over probability.
        seed: Integer
            Set function to get same results.
        showprogress: Boolean
            If we want to show the progress of each generation.
            
    ---
    Output:
        A tuple with (deap.toolbox.population, deap.Logbook)
        
    '''
    random.seed(seed)
    
    (toolbox, stats, logbook, _) = envrironment

    pop = toolbox.population(MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if showprogress:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        offspring = [toolbox.ceiling(ind) for ind in offspring]
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)
        
        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if showprogress:
            print(logbook.stream)
    
    if showprogress:
        print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook


# In[ ]:




