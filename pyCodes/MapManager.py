#!/usr/bin/env python
# coding: utf-8

# In[1]:


import folium
import numpy as np
import random
import webbrowser


# In[ ]:


class DataFrameToMapManager:
    '''
    This class has the purpose of using data
    from a dataframe to feed a map with its 
    information concentrated in points.
    '''
    def __init__(self,df):
        self.df = df
        
    def indexes_DF(self, indexes):
        '''
        Intro:
            Function to filter the number of points in the DataFrame.
        ---
        Input:
            indexes: List of integers
                Row indices to filter DataFrame.
        '''
        self.df = self.df.loc[np.unique(np.array(indexes)), :]
        
    def displayInMap(self, center,columnsToShow,xupperlimit=0.00009,xlowerlimit=0.00001,yupperlimit=0.00009,ylowerlimit=0.00001):
        '''
        Intro:
            Function to plot informations in a map
        ---
        Input:
            center : List
                List with Latitude and Longitude
            columnsToShow : List
                List with the names of the columns to show
            xs and ys limits: Floats
                Lower and upper values that will be used to
                choose random number to push the point found
                so that no two points can overlap
                
        '''
        # Setting the map
        m = folium.Map(location=center, zoom_start=2)
        
        # Setting the tooltip
        tooltip = "Click For More Info"
        
        # Setting the company point
        folium.Marker(center,
            popup='<strong>My Company</strong>',
            tooltip="My Company",
            icon=folium.Icon(color='purple')).add_to(m)
        
        # Marking the points in the map
        for i in range(self.df.shape[0]):
            line = self.df.iloc[i]
            
            # Constructing string
            stringToShow = "<strong>"
            for column in columnsToShow:
                if column in self.df:
                    stringToShow += column + " : " + str(line[column]) + "<br>" 
            stringToShow += "</strong>"
            
            # Get random signal for each direction
            if 0.5 < random.uniform(0.0, 1.0):
                sx = -1
            else:
                sx = 1
            if 0.5 < random.uniform(0.0, 1.0):
                sy = -1
            else:
                sy = 1
            
            # Plot
            lat = line["Latitude"]
            lon = line["Longitude"]
            folium.Marker([lat + sx*((xupperlimit-xlowerlimit)*random.uniform(0.0, 1.0) + xlowerlimit),
                   lon + sy*((yupperlimit-ylowerlimit)*random.uniform(0.0, 1.0) + ylowerlimit)],
            popup=stringToShow,
            tooltip=tooltip).add_to(m)
            
        # Saving it
        m.save('map.html')
        
        # Loading it
        url = './map.html'
        webbrowser.open_new_tab(url)

