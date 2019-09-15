#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sin, cos, sqrt, atan2, radians, exp
from datetime import datetime
import pandas as pd
import numpy as np
import copy
from functools import partial


# In[84]:


class SearchParameterManager:
    '''
    Intro:
        The purpose of this class is to facilitate
        the research, storing the variables and realizing calculations about them.
    Input:
        companyGPS : Tuple of Floats of coordinates in degree
            company coordinates
        typeOfMachine : String
            The type of the machine that we want to buy
        availabilityDates : Tuple of String
            Has the date of start and the date of end for the duration
            when the company need the machine.
        dictOfParams : Dictionary of Lists
            The key must be name the name of the column and
            the value must be one list: if there is one value,
            then during the filtering will select only the
            machines that have exactly this value. If there
            are two values, then the filtering will select
            only the machines that have a value between the
            two values.
    
    '''
    def __init__(self, companyGPS, typeOfMachine, availabilityDates, dictOfParams):
        if self.distanceBetweenDates(availabilityDates[0],availabilityDates[1]) < 0:
                raise Exception('Dates of machine are not consistent (end before start)')
        self.companyGPS = companyGPS
        self.typeOfMachine = typeOfMachine
        self.availabilityDates = availabilityDates
        self.dictOfParams = dictOfParams
        
    def gpsToRadian(self):
        '''
        Intro:
            Returns gps coordinates in radians in a tuple
        '''
        return (radians(self.companyGPS[0]),radians(self.companyGPS[1]))
    
    def calculateDistance(self,gpsInDegree):
        '''
        Intro:
            Function to calculate the distance between two GPS coordinates
        ---
        Inputs: 
            gpsInDegree: Tuple of Floats
                Coordinate of the machine
        ---
        Outputs:
            Returns the distance (from company to machine)
        '''
        # approximate radius of earth in km
        R = 6373.0
        mygpsx,mygpsy = self.gpsToRadian()
        othergpsx,othergpsy = radians(gpsInDegree[0]),radians(gpsInDegree[1])
        dlon = othergpsy - mygpsy
        dlat = othergpsx - mygpsx

        a = sin(dlat / 2)**2 + cos(mygpsx) * cos(othergpsx) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        
        return distance
    
    def distanceBetweenDates(self,date1,date2):
        '''
        Intro:
            Function that calculates distance between dates.
        ---
        Inputs: 
            date1 and date2: Strings
                Dates in day-month-year format.
        ---
        Outputs:
            Returns distance between dates in integer.
        '''
        date_format = "%d-%m-%Y"
        a = datetime.strptime(date1, date_format)
        b = datetime.strptime(date2, date_format)
        delta = b - a
        return int(delta.days)
    
    def distanceOverlapFormula(self, dist):
        '''
        Intro:
            Function that calculates Temporal Temporal Overlap (formula)
        ---
        Inputs: 
            dist: Integer
                Distance between dates
        ---
        Outputs:
            Returns the Temporal Temporal Overlap
        '''
        return -1.0/(exp(1.0/abs(dist)))
    
    def calculateOverlapCoef(self, startavmachine, endavmachine):
        '''
        Intro:
            Function that calculates Temporal Overlap Factor Between Two Dates.
        ---
        Inputs: 
            startavmachine: String (Dates in day-month-year format)
                Availability Start Date.
            endavmachine: String (Dates in day-month-year format)
                Availability End Date.
        ---
        Outputs:
            Returns the Temporal Overlap Coefficient between the 
            availability date of the machine and the period the
            company needs the machine.
        '''
        if self.distanceBetweenDates(startavmachine,endavmachine) < 0:
            raise Exception('Dates of machine are not consistent (end before start)')
        
        duration = self.distanceBetweenDates(self.availabilityDates[0],self.availabilityDates[1])
        # Before
        if self.distanceBetweenDates(endavmachine,self.availabilityDates[0]) > 0:
            return self.distanceOverlapFormula(self.distanceBetweenDates(self.availabilityDates[0],endavmachine))
        # After
        elif self.distanceBetweenDates(self.availabilityDates[1], startavmachine) > 0:
            return self.distanceOverlapFormula(self.distanceBetweenDates(startavmachine,self.availabilityDates[1]))
        else:
            # Inf
            if self.distanceBetweenDates(startavmachine,self.availabilityDates[0]) >= 0:
                return min(duration,self.distanceBetweenDates(self.availabilityDates[0],endavmachine))/duration
            # Sup
            elif self.distanceBetweenDates(self.availabilityDates[1],endavmachine) >= 0:
                return min(duration,self.distanceBetweenDates(startavmachine, self.availabilityDates[1]))/duration
            # Overlap
            else:
                return min(duration,self.distanceBetweenDates(startavmachine, endavmachine))/duration
    
    def filterDatabase(self, df_origin):
        '''
        Intro:
            Function to filter the database from company preferences.
        ---
        Inputs: 
            df_origin: DataFrame
                DataFrame that will be filtered.
        ---
        Outputs:
            Returns filtered DataFrame.
        '''
        df = copy.deepcopy(df_origin)
        numberOfLines = df.shape[0]
        todaysDay = datetime.today().strftime('%d-%m-%Y')
        
        overlapCoef = []
        distances = []      
        age = []
        
        # Adding Pending Columns
        for i in range(numberOfLines):
            distances.append(self.calculateDistance((df.loc[i]['Latitude'],df.loc[i]['Longitude'])))
            overlapCoef.append(self.calculateOverlapCoef(df.loc[i]['Available from'],df.loc[i]['Available until']))
            age.append(self.distanceBetweenDates(df.loc[i]['Purchase date'],todaysDay)/365.0)
        df['Distance'] = np.array(distances)
        df['Temporal Overlap Coefficient'] = np.array(overlapCoef)
        df['Age'] = np.array(age)
        
        # Filter data
        i = 0
        linesToDrop = set([])
        while i < df.shape[0]:
            if df.loc[i]['Type of machine'] != self.typeOfMachine:
                linesToDrop.add(i)
            else:
                for key in self.dictOfParams.keys():
                    if key in df.columns:
                        if len(self.dictOfParams[key]) == 1:
                            if df.loc[i][key] != self.dictOfParams[key][0]:
                                linesToDrop.add(i)
                                break
                        elif len(self.dictOfParams[key]) == 2:
                            if df.loc[i][key] < self.dictOfParams[key][0] or df.loc[i][key] > self.dictOfParams[key][1]:
                                linesToDrop.add(i)
                                break
            i += 1
            
        df = df.drop(list(linesToDrop)).reset_index(drop=True)
        
        return df
    
    def pseudoFunc(self,dicOfColumns, x):
        '''
        Intro:
            Multi-objective function to be optimized.
        ---
        Inputs: 
            dicOfColumns: Dict
                DataFrame that will be filtered.
            x: List
                Individual.
        ---
        Outputs:
            Returns the result of the function performed on element.
        '''
        output = []
        for key in dicOfColumns:
            output.append(x[key])
        return tuple(output)

    def createInputOfOptAlgorithms(self,df_origin, dictColumnPrefs):
        '''
        Intro:
            The purpose of this function is to create the input data
            needed to the optimization algorithms, using a dataframe
            with the search data and one dictionary with the
            weights of each column to optimize
        Input:
            df_origin : DataFrame
                DataFrame with all data that were going to be analysed to
                find the best machine among them.
            dictColumnPrefs : Dict
                Dictionary where the keys are the name of the columns which
                we want to optimize and their values are the weights which
                the optimization algorithm will use to prioritize one
                criteria (column) over another.
        Output:
            df : DataFrame
                Copy of df_origin, but with the indexes so that it will
                be possible to localize the machines by their 
                integer/binary labels
            X : Array
                Array of df data which will be used in the optimization 
                algorithms
            dim_of_labels : Integer    
                Binary label size for each element.
            funcOutput : Function
                Function that will be optimized to find the best machine
                in the array.

        '''
        
        df = copy.deepcopy(df_origin)
        df.reset_index(level=0, inplace=True)
        listOfParams = ['index']
        for key in dictColumnPrefs.keys():
            if dictColumnPrefs[key] != 0:
                listOfParams.append(key)
        df = df[listOfParams]
        weights = {}
        columns_list = list(df.columns)
        for key in dictColumnPrefs.keys():
            ind = columns_list.index(key)
            weights.update({ind:dictColumnPrefs[key]})

        funcOutput = partial(self.pseudoFunc,weights)
        X = np.array(df)
        dim_of_labels = len(str(bin(len(X)-1))[2:])
        
        return (df,X,dim_of_labels,funcOutput)

