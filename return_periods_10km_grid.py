# -*- coding: utf-8 -*-
"""
This script is part of Bloemendaal et al, Estimation of global tropical cyclone wind probabilities using the STORM dataset (in review)
This script calculates the empirical return periods at 10 km resolution.
The TC tracks are taken from the STORM model, see 

Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""

import numpy as np
import pandas as pd
import sys
from SELECT_BASIN import Basins_WMO #see STORM module
from scipy import spatial
import scipy.stats

basin='EP'

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m+h, m-h

#==============================================================================
# Make a list of different return periods
#==============================================================================
returnperiods=[]
returnperiods.extend(np.linspace(10,100,10))
returnperiods.extend(np.linspace(200,1000,9))
returnperiods.extend(np.linspace(2000,10000,9))

lat0,lat1,lon0,lon1=Basins_WMO(basin)[2:]
# =============================================================================
# Make a list of all lon/lat points in the basin
# =============================================================================

res=0.1
if lat0>0:
    latspace=np.arange(lat0+res/2.,lat1+res/2.,res)
else:
    latspace=np.arange(lat0-res/2.,lat1-res/2.,res)

lonspace=np.arange(lon0+res/2.,lon1+res/2.,res)

points=[(i,j) for i in latspace for j in lonspace]

all_data={i:[] for i in range(len(points))}

#Please check the definition of wind speed! If it's 1-minute sustained wind, use the following: 
#wind_items=[20,25,30,33,35,40,42,45,50,55,58,60,65,70,75,80,85]

#In STORM, the wind speeds are 10-min average, so the Saffir-Simpson category thresholds need to be converted from 1-min to 10-min: 
wind_items=[20,25,29,30,35,37.6,40,43.4,45,50,51.1,55,60,61.6,65,70,75]

# =============================================================================
# Open the datasets
# =============================================================================
for i in range(0,10):
    for j in range(0,10):
      dataset = np.load("STORM_WINDSPEEDS_"+str(basin)+"_"+str(index)+"_"+str(yearslice)+".npy",allow_pickle=True).item()       
      for idx in range(len(dataset)):
          if len(dataset[i][j][idx])>0:
              for w in dataset[i][j][idx]:
                  all_data[idx].append(w)
      del dataset

rp_slices={i:[] for i in range(len(points))}
wind_slices={i:[] for i in range(len(points))}

for i in range(len(dataset)):
    rp_slices[i]={j:[] for j in returnperiods}
    wind_slices[i]={j:[] for j in wind_items}
	
    data=all_data[i]
	
    rp_list={j:[] for j in returnperiods}
    wind_rp={j:[] for j in wind_items}
    
    (lat,lon)=points[i]
	
    if len(data)>0.:
# =============================================================================
#         Bootstrap
# =============================================================================
        for iteration in range(0,1000):
            random_draw=np.random.choice(data,size=len(data),replace=True)
  			
            df1=pd.DataFrame({'Wind': random_draw})
            df1['Ranked']=df1['Wind'].rank(ascending=0) 
            df1=df1.sort_values(by=['Ranked'])    			     
            ranklist=df1['Ranked'].tolist()        
            windlist=df1['Wind'].tolist() 
  				
            rpwindlist=[]
            for j in range(len(ranklist)):
                weibull=(ranklist[j])/(len(ranklist)+1.) #weibulls plotting formula. This yields the exceendance probability per event set
                r=weibull*(len(ranklist)/10000.) #multiply by the average number of events per year to reach the exceedence probability per year
                rpwindlist.append(1./r) #convert to year instead of probability
  				
            rpwindlist=rpwindlist[::-1]         
            windlist=windlist[::-1] 
  				
# =============================================================================
#             Calculate the wind speed value per return period for every bootstrap iteration, save in rp_list-array
# =============================================================================
            for rp in returnperiods:
                if np.max(rpwindlist)>rp:            
                    #Interpolate to the desired return level
                    windint=np.interp(rp,rpwindlist,windlist)
                    rp_list[rp].append(windint)
# =============================================================================
#             Calculate the return period for every wind speed for every bootstrap iteration, save in wind_rp-array
# =============================================================================
            for w in wind_items:
                if np.max(windlist)>=w and w>=np.min(windlist):            
                    #Interpolate to the desired return level
                    rpint=np.interp(w,windlist,rpwindlist)
                    wind_rp[w].append(rpint) 		
			
# =============================================================================
# After the bootstrap is finished, we calculate the mean, standard deviation,  				
# minimum, maximum, median, 25- and 75-percentile and 95-percentage confidence interval values
# =============================================================================
        for rp in returnperiods:
            if len(rp_list[rp])>0.:
                mean=np.mean(rp_list[rp])
                std=np.std(rp_list[rp])
                maximum=np.max(rp_list[rp])
                minimum=np.min(rp_list[rp])
                median=np.median(rp_list[rp])
  					
                perc_25=np.percentile(rp_list[rp],25)
                perc_75=np.percentile(rp_list[rp],75)
  					
                conf_95,conf_5=confidence_interval(rp_list[rp])
  					
                rp_slices[i][rp]=[lat,lon,mean,std,median,minimum,maximum,perc_25,perc_75,conf_95,conf_5,len(data)]
            else:
                rp_slices[i][rp]=[lat,lon,np.nan]

        for w in wind_items:
            if len(wind_rp[w])>0.:
                mean=np.mean(wind_rp[w])
                std=np.std(wind_rp[w])
                maximum=np.max(wind_rp[w])
                minimum=np.min(wind_rp[w])
                median=np.median(wind_rp[w])
  					
                perc_25=np.percentile(wind_rp[w],25)
                perc_75=np.percentile(wind_rp[w],75)
  					
                conf_95,conf_5=confidence_interval(wind_rp[w])
  					
                wind_slices[i][w]=[lat,lon,mean,std,median,minimum,maximum,perc_25,perc_75,conf_95,conf_5,len(data)]
            else:
                wind_slices[i][w]=[lat,lon,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,0]             

    else:
        for rp in returnperiods:
            rp_slices[i][rp]=[lat,lon,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,0]

        for w in wind_items:
            wind_slices[i][w]=[lat,lon,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,0]     
	            
print(datetime.now()-starttime)
    
np.save('Bootstrap_return_periods_'+str(basin)+'.npy',rp_slices)
np.save('Bootstrap_wind_speeds_'+str(basin)+'.npy',wind_slices)
