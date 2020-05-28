# -*- coding: utf-8 -*-
"""
This script is part of Bloemendaal et al, Estimation of global tropical cyclone wind probabilities using the STORM dataset (in review)
This script calculates the return periods for all tropical cyclones within 100 km of 18 pre-defined coastal cities.

The TC tracks are taken from the STORM model, see 

Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)

    Parameters
    ----------
    lon1,lat1 : coordinates location 1
    lon2,lat2 : coordinates location 2

    Returns
    -------
    distance in km.

    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

wmin,wmax=18,75

thresholds=np.linspace(wmin,wmax,int(wmax-wmin)+1)

#details of the 18 chosen cities
overview={i:[] for i in range(0,6)}
overview[0]=[['EP','Honolulu',21.3281101,-157.8340924],['EP','San Diego',32.824763,-117.2490598],['EP','Acapulco',16.8355396,-99.8973279]]
overview[1]=[['NA','San Juan',18.465540,-66.0954918],['NA','Houston',29.760427,-95.369804],['NA','Miami',25.761681,-80.191788]]
overview[2]=[['NI','Mumbai',19.075983,72.877655],['NI','Chittagong',22.356852,91.783180],['NI','Muscat',23.613540,58.594110]]
overview[3]=[['SI','Beira',-19.830500,34.843021],['SI','Saint-Denis',-20.882980,55.450020],['SI','Toamasina',-18.185568,49.36846]]
overview[4]=[['SP','Brisbane',-27.3812533,152.7130144],['SP','Nadi',-17.802191,177.419540],['SP','Noumea',-22.271072,166.441650]]
overview[5]=[['WP','Ho Chi Minh',10.7553411,106.4150407],['WP','Taipei',25.032969,121.565414],['WP','Tokyo',35.686958,139.749466]]

#%%

wind_dict={i:[] for i in range(0,6)}
for i in range(0,6):
    wind_dict[i]={j:[] for j in range(0,3)}

radius=100
for basin,basinid in zip(['EP','NA','NI','SI','SP','WP'],range(0,6)):
    for index in range(0,10): 
        #load the STORM datasets. Make sure the directory is set right!
        data=np.loadtxt('STORM_DATA_IBTRACS_'+str(basin)+'_1000_YEARS_'+str(index)+'.txt',delimiter=',')
        
        #extract necessary parameters
        time,lat,lon,wind=data[:,3],data[:,5],data[:,6],data[:,8]
        del data
            
        indices=[i for i,x in enumerate(time) if x==0]
        indices.append(len(time))
        
        i=0
        #loop over all TCs in the dataset
        while i<len(indices)-1:
            start=indices[i]
            end=indices[i+1]
            
            latslice=lat[start:end]
            lonslice=lon[start:end]
            windslice=wind[start:end]
                    
            for l in range(0,3): #for every city in the basin
                wind_loc=[]
                lat_loc=overview[basinid][l][2]
                lon_loc=overview[basinid][l][3]
                if lon_loc<0.:
                    lon_loc+=360    
            
                for j in range(len(latslice)):
                    #calculate the distance between the track and the city
                    distance=haversine(lonslice[j],latslice[j],lon_loc,lat_loc)
                  
                    if distance<=radius:
                        wind_loc.append(windslice[j])
                
                if len(wind_loc)>0.:
                    if np.max(wind_loc)>=18.:
                        wind_dict[basinid][l].append(np.max(wind_loc)) #store the maximum wind speed for the TC
            i=i+1

#%%
"""
Here, please define at which return periods you'd like to evaluate the wind speeds. This needs to be 
stored in an array called returnperiods.
In addition, please define at which wind speeds you'd like to evluate the return periods. This needs
to be stored in an array called wind_items. 
Please find examples of the returnperiods and wind_items lists below. These were used in the paper.
"""
            
returnperiods=[]
returnperiods.extend(np.linspace(10,100,10))
returnperiods.extend(np.linspace(200,1000,9))
returnperiods.extend(np.linspace(2000,10000,9))

wind_items=[18,20,25,30,33,35,40,42,45,50,55,58,60,65,70,75]

for i in range(0,6):
    for j in range(0,3):
        name=overview[i][j][1]

        df=pd.DataFrame({'Wind': wind_dict[i][j]})
        df['Ranked']=df['Wind'].rank(ascending=0)
        df=df.sort_values(by=['Ranked'])
        ranklist=df['Ranked'].tolist()
        windlist=df['Wind'].tolist() 
        
        rpwindlist=[]  
            
        for m in range(len(ranklist)):
            weibull=(ranklist[m])/(len(ranklist)+1.) #weibulls plotting formula. This yields the exceendance probability per event set
            r=weibull*(len(ranklist)/10000.) #multiply by the average number of events per year to reach the exceedence probability per year
            rpwindlist.append(1./r) #convert to year instead of probability
  				
        rpwindlist=rpwindlist[::-1]         
        windlist=windlist[::-1] 
        
        for rp in returnperiods:
            if np.max(rpwindlist)>=rp and np.min(rpwindlist)<=rp:            
                #Interpolate to the desired return level
                windint=np.interp(rp,rpwindlist,windlist)
                print(name,'Return period '+str(rp),'Wind speed '+str(windint))

        for w in wind_items:
            rp_int=np.interp(w,windlist,rpwindlist)
            print(name,'Wind speed '+str(w),'Return period '+str(rp_int))
        
