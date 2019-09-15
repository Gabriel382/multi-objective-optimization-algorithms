# multi-objective-optimization-algorithms

We will see here an analysis of several methods of optimization algorithms focusing on multi-objective optimization. For each method, we will have an introduction, a methodology, an implementation, a conclusion and a bibliography.

## The Problem

When you have a database where you want to find the points that belong to the pareto curve within the database, deterministic algorithms like Simple Cull can solve the problem. However, the larger the database, the longer the deterministic algorithms will take. In this environment, it is interesting to use stochastic algorithms that have almost constant excitation times for certain parameters and a given problem.

## The compared algorithms

Here, many algorithms will be tested (some with their continuous version and all with a version that solves our problem). Here is a list of the methods:

* Non-dominated Sorting Genetic Algorithm II
* Non-dominated Sorting Genetic Algorithm III
* SPEA2
* Non-dominated Sorting Gravitational Search Algorithm
* Simple Cull algorithm
* Pareto Simulated Annealing
* Binary Multi-Objective Tabu Search
* Full Non-dominated Sorting and Ranking
