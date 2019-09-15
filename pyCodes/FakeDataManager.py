#!/usr/bin/env python
# coding: utf-8

# # FakeDataManager
# 
# > This code generates fake data and can save and loat it in a csv

# ## 1. Importing

# In[4]:


import numpy as np
import pandas as pd
import os
import random


# ## 2. Helpers of helpers

# In[2]:


def save_dict(dictionary, folderpath, filename, mode='a'):
    '''
    Intro:
        Save the dictionary to a csv
    ---
    Input:
        dictionary: dict
            Dictionary of save values (keys
            are column names and dictionary
            values are csv values)
        folderpath: String
            Folder where file will be saved
        filename: String
            File name
        mode: character
            Size 1 string with save mode:
                a: add new values
                w: overwrites previous values
        
    '''
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        
    path = folderpath + "/" + filename
    df = pd.DataFrame(dictionary, columns=dictionary.keys())
    if not os.path.isfile(path):
        df.to_csv(path, header=dictionary.keys(), index=None)
    else:
        if mode == 'w':
            df.to_csv(path, header=dictionary.keys(), mode=mode, index=None)
        elif mode == 'a':
            df.to_csv(path, header=None, mode=mode, index=None)
        else:
            raise Exception('Unrecognized mode')


# In[9]:


def display_csv(filepath):
    '''
    Intro:
        Show values in a csv
    ---
    Input:
        filepath: String
            Path to file
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


# In[10]:


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
        df = pd.read_csv(filepath)
        return df


# ## 3. Create Fake Data

# In[5]:


def createFakeNumericData(namesOfColumns, lowerlimits, upperlimits, PopSize):
    '''
    Intro:
        Functions for creating lists of false numeric values
        and placing them in a dictionary (in an easy-to-pass
        format for a DataFrame)
    ---
    Input:
        namesOfColumns: List of String
            Name of each dictionary key
        lowerlimits: List
            List of lower values for each key.
        upperlimits: List
            List of upper values for each key.
        PopSize: Integer
            Number of false values per list
    ---
    Output:
        False value dictionary that can easily be embedded in a DataFrame
    '''
    if len(namesOfColumns) == len(lowerlimits) and len(lowerlimits) == len(upperlimits):
        fakeDict = {}
        for i in range(len(namesOfColumns)):
            fakeDict[namesOfColumns[i]] = np.random.uniform(0,1,(PopSize)) *(upperlimits[i]-lowerlimits[i])+lowerlimits[i]
        return fakeDict
    else:
        raise Exception('every array must have same len')


# In[7]:


def createFakeObjectData(namesOfColumns, arrayOfElementsForEachColumn,PopSize):
    '''
    Intro:
        Function that creates lists of random values 
        (between a group of predefined values)
        and places them in a dictionary (in an easy-to-pass
        format for a DataFrame)
    ---
    Input:
        namesOfColumns: List of String
            Name of each dictionary key
        arrayOfElementsForEachColumn: List of Lists
            List of allowed value lists (from each 
            list, the individual will choose only one 
            random value)
        PopSize: Integer
            Number of values per list
    ---
    Output:
        False value dictionary that can easily be embedded in a DataFrame
    '''
    if len(namesOfColumns) == len(arrayOfElementsForEachColumn):
        fakeDict = {}
        for i in range(len(namesOfColumns)):
            temp = []
            for j in range(PopSize):
                temp.append(random.choice(arrayOfElementsForEachColumn[i]))
            fakeDict[namesOfColumns[i]] = np.array(temp)
        return fakeDict
    else:
        raise Exception('every array must have same len')

