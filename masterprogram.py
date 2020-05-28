# -*- coding: utf-8 -*-
"""
This script is part of Bloemendaal et al, Estimation of global tropical cyclone wind probabilities using the STORM dataset (in review)
The script has been developed by Nadia Bloemendaal, Job Dullaart and Sanne Muis. 
This script is the master program complementary to holland_model.py. The methodology is heavily inspired by 

Lin, N., and Chavas, D. ( 2012), On hurricane parametric wind and applications in storm surge modeling, 
J. Geophys. Res., 117, D09120, doi:10.1029/2011JD017126.

The TC tracks are taken from the STORM model, see 

Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Copyright (C) 2020 Nadia Bloemendaal. All versions released under GNU General Public License v3.0.
"""
import numpy as np
import holland_model as hm
import math
from scipy import spatial

def Basins_WMO(basin):
    if basin=='EP': #Eastern Pacific
        lat0,lat1,lon0,lon1=5,60,180,285
    if basin=='NA': #North Atlantic
        lat0,lat1,lon0,lon1=5,60,255,359
    if basin=='NI': #North Indian
        lat0,lat1,lon0,lon1=5,60,30,100
    if basin=='SI': #South Indian
        lat0,lat1,lon0,lon1=-60,-5,10,135
    if basin=='SP': #South Pacific
        lat0,lat1,lon0,lon1=-60,-5,135,240
    if basin=='WP': #Western Pacific
        lat0,lat1,lon0,lon1=5,60,100,180
    
    return lat0,lat1,lon0,lon1

# =============================================================================
# Please define a basin (EP,NA,NI,SI,SP,WP), index (0-9), and yearslice of 100 years (0-9)
# =============================================================================
basin='EP'
index=0
yearslice=0

year_int=100

#==============================================================================
# Constants from literature
#==============================================================================
alpha=0.55              #Deceleration of surface background wind - Lin & Chavas 2012
beta_bg=20.             #Angle of background wind flow - Lin & Chavas 2012
SWRF=0.85               #Empirical surface wind reduction factor (SWRF) - Powell et al 2005 
CF=0.915                #Wind conversion factor from 1 minute average to 10 minute average - Harper et al (2012)

#==============================================================================
# Other pre-defined constants
#==============================================================================
tc_radius=1000.            #Radius of the tropical cyclone, in km
max_distance=tc_radius/110. #Maximum distance (in degrees) to look for coastal points in the full file
Patm=101325.

#==============================================================================
# Spyderweb specifications 
#==============================================================================
n_cols=36                   #number of gridpoints in angular direction
n_rows=1000                 #number of gridpoints in radial direction

#==============================================================================
# Create a list of points at 0.1deg resolution, spanning the whole basin
#==============================================================================
lat0,lat1,lon0,lon1=Basins_WMO(basin) #see basin boundary definitions in line 23
res=0.1
if lat0>0:
    latspace=np.arange(lat0+res/2.,lat1+res/2.,res)
else:
    latspace=np.arange(lat0-res/2.,lat1-res/2.,res)

lonspace=np.arange(lon0+res/2.,lon1+res/2.,res)

points=[(i,j) for i in latspace for j in lonspace]
   
wind_field={i:[] for i in range(len(points))} #the index corresponds to the index of the 
#points-list. So the lon/lat of index i is given by points[i], the wind data by wind_field[i] 

tree=spatial.cKDTree(points)

#==============================================================================
# Open the STORM dataset
#==============================================================================
#Please make sure to point to the right directory!
data=np.loadtxt('STORM_DATA_IBTRACS_'+str(basin)+'_1000_YEARS_'+str(index)+'.txt',delimiter=',')
yearall,timeall,latall,lonall,presall,windall,rmaxall=data[:,0],data[:,3],data[:,5],data[:,6],data[:,7],data[:,8],data[:,9]

year=[y for y in yearall if y>=yearslice*year_int and y<(yearslice+1)*year_int]
year_begin=year[0]
year_end=year[-1]
begin=np.searchsorted(yearall,year_begin)
end=np.searchsorted(yearall,year_end)   

indiceslist=[i for i,x in enumerate(timeall[begin:end]) if x==0]  #a new TC always starts at time step 0 
i=0   
#loop over the different TCs
while i<len(indiceslist)-1:  
    start=indiceslist[i]+begin
    end=indiceslist[i+1]+begin
    
    latslice=latall[start:end]
    lonslice=lonall[start:end]
    windslice=windall[start:end]
    presslice=presall[start:end]
    timeslice=timeall[start:end] 
    rmaxslice=rmaxall[start:end]
    
    shadowlist={kk:[] for kk in range(len(points))}
    for j in range(1,len(latslice)):
      lat0,lat1,lon0,lon1,t0,t1=latslice[j-1],latslice[j],lonslice[j-1],lonslice[j],timeslice[j-1],timeslice[j]
      U10,Rmax,P=windslice[j],rmaxslice[j],presslice[j]
      dt=(t1-t0)*3600*3.                            
      
      #Generate the seperate list of coastal points that are in the spyderweb
      distances, indices = tree.query((lat1,lon1),k=len(points), p=2,distance_upper_bound=max_distance)                    
      points_to_save=[points[indices[k]] for k in range(len(distances)) if distances[k]<max_distance]
      
      #Spyderweb step 1: Generate the spyderweb mesh --> predefined function!
      
      rlist,thetalist,xlist,ylist=hm.Generate_Spyderweb_mesh(n_cols,n_rows,tc_radius,lat0)
      
      latlist,lonlist=hm.Generate_Spyderweb_lonlat_coordinates(xlist,ylist,lat1,lon1)
      
      #Spyderweb step 2: Calculate the background wind --> predefined function!
      
      [bg,ubg,vbg]=hm.Compute_background_flow(lon0,lat0,lon1,lat1,dt)
      
      #Spyderweb step 3: Subtract the background flow from U10 (tropical cyclone's 10-meter wind speed)
      #For this, first convert U10 to surface level using the SWRF-constant
      #next, subtract a fraction alpha of the background flow.
      
      Usurf=(U10/SWRF)-(bg*alpha) #1-minute maximum sustained surface winds
      
      P_mesh=np.zeros((xlist.shape))
      Pdrop_mesh=np.zeros((xlist.shape))
      
      up=np.zeros((xlist.shape))
      vp=np.zeros((xlist.shape))
      
      #Spyderweb step 4: Calculate wind  and pressure profile using the Holland model
      for l in range(1,n_rows):
          r=rlist[0][l]
          Vs,Ps=hm.Holland_model(lat1,P,Usurf,Rmax,r)
          Vs=Vs*SWRF      #Convert back to 10-min wind speed
          
          P_mesh[:,l].fill(Ps/100.)    #in Pa
          Pdrop_mesh[:,l].fill((Patm-Ps)/100.)   #in Pa
          
          beta=hm.Inflowangle(r,Rmax,lat0)
          
          for k in range(0,n_cols):
            ubp=alpha*(ubg*math.cos(math.radians(beta_bg))-np.sign(lat0)*vbg*math.sin(math.radians(beta_bg)))
            vbp=alpha*(vbg*math.cos(math.radians(beta_bg))+np.sign(lat0)*ubg*math.sin(math.radians(beta_bg)))
                                 
            up[k,l]=-Vs*math.sin(thetalist[:,0][k]+beta)+ubp
            vp[k,l]=-Vs*math.cos(thetalist[:,0][k]+beta)+vbp
                
      u10=CF*up
      v10=CF*vp
      windfield=np.sqrt(u10**2.+v10**2.)
        
      spy_points=[]
      wind_points=[]
        
      for k in range(n_cols):
        for l in range(n_rows):
          spy_points.append((latlist[k,l],lonlist[k,l])) 
          wind_points.append(windfield[k,l])
        
      tree2=spatial.cKDTree(spy_points)
      #overlay the spyderweb grid with the regular grid
      for (lat,lon),idx in zip(points_to_save,range(len(points_to_save))):
        local_dist, local_ind = tree2.query((lat,lon),k=1, p=2,distance_upper_bound=max_distance)    
        shadowlist[indices[idx]].append(wind_points[local_ind])
            
    for m in range(len(shadowlist)):
      if len(shadowlist[m])>0.:
        if np.max(shadowlist[m])>=18.:
          wind_field[m].append(np.max(shadowlist[m])) #this dictionary contains the max wind speeds per location
                        
    i=i+1   

#Store the dictionary with max wind speeds per location for 100 years of data                 
np.save("STORM_WINDSPEEDS_"+str(basin)+"_"+str(index)+"_"+str(yearslice)+".npy",wind_field)