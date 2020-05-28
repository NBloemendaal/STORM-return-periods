# -*- coding: utf-8 -*-
"""
This script is part of Bloemendaal et al, Estimation of global tropical cyclone wind probabilities using the STORM dataset (in review)
This script calculates the basin-wide return periods empirically and using EV distributions.
The TC tracks are taken from the STORM model, see 

Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Parts of this script originates from https://www.linkedin.com/pulse/beginners-guide-carry-out-extreme-value-analysis-2-chonghua-yin/
Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""
import numpy as np
import pandas as pd
import lmoments
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

#==============================================================================
# Calculate return periods in STORM dataset
#==============================================================================
#Please select basin
basin='SP'
max_windspeed=[]
for index in range(0,10): 
    data=np.loadtxt("STORM\\STORM_DATA_IBTRACS_"+str(basin)+"_1000_YEARS_"+str(index)+".txt",delimiter=',')
    year,time,wind=data[:,0],data[:,3],data[:,8]

    indices=[i for i,x in enumerate(time) if x==0]  #a new TC always starts at time step 0 
    indices.append(len(year))
    i=0   
    
    while i<len(indices)-1:
        
        start=indices[i]
        end=indices[i+1]

        windslice=wind[start:end]
        timeslice=time[start:end]
        
        max_windspeed.append(max(windslice))
        
        i=i+1

#==============================================================================
#    Group the data in a dataframe so that the data can be ranked    
#==============================================================================
df=pd.DataFrame({'Max U10':max_windspeed})
#we want the highest wind speeds to get the highest rank
df['Ranked']=df['Max U10'].rank(ascending=0) 
df=df.sort_values(by=['Ranked'])
ranklist=df['Ranked'].tolist()
windlist=df['Max U10'].tolist()

rpwindlist=[]
for i in range(0,len(ranklist)):
    weibull=(ranklist[i])/(10000.+1)
    r=weibull*(len(ranklist)/10000.)
    rpwindlist.append(1./r)

#The lists need to be reversed for the interpolation  
ratio=len(windlist)/(1000.*selection)  
rpwindlist=rpwindlist[::-1]         
windlist=windlist[::-1] 

T=np.arange(0.1,25000.1,0.1)+1
#interpolate the wind to the desired return periods
windint=[]
for rp in T:
    windint.append(np.interp(rp,rpwindlist,windlist))

np.savetxt('STORM_wind_'+str(basin)+'.txt',windint)  
tlist=T
bootstrapgev={i:[] for i in range(0,len(tlist))}
bootstrapexp={i:[] for i in range(0,len(tlist))}
bootstrapgum={i:[] for i in range(0,len(tlist))}
bootstrapwei={i:[] for i in range(0,len(tlist))}
bootstrappar={i:[] for i in range(0,len(tlist))}
bootstraplog={i:[] for i in range(0,len(tlist))}
bootstrappea={i:[] for i in range(0,len(tlist))}
bootstrapgam={i:[] for i in range(0,len(tlist))}

#==============================================================================
# Select 1000 times a 38 years -period from the STORM dataset
#==============================================================================
ratiolist=[]
for ro in range(0,10):
    index=random.sample(range(0,10),1)[0]
    data=np.loadtxt("STORM_DATA_IBTRACS_"+str(basin)+"_1000_YEARS_"+str(index)+".txt",delimiter=',')
    year,time,wind=data[:,0],data[:,3],data[:,8]
    
    print(ro)
    year_unique=np.unique(year) 
    loop=0
    while loop<100:
        max_wind_30=[]
        r=random.sample(range(len(year_unique[0:-39])),1)[0]
        year_begin=year_unique[r]    
        year_list=np.linspace(int(year_begin),int(year_begin)+37,38)
        year_end=min(i for i in year if i>year_list[-1])
        
        begin=np.searchsorted(year,year_begin)
        end=np.searchsorted(year,year_end)    
        
        indices=[i for i,x in enumerate(time[begin:end]) if x==0] #new storm always starts at 0
        indices.append(len(year))
        i=0
        while i<len(indices)-1:
            start=begin+indices[i]
            end=begin+indices[i+1]
            
            windslice=wind[start:end]
                
            max_wind_30.append(max(windslice))
    
            i=i+1    
    
        df1=pd.DataFrame({'Max U10':max_wind_30})
        #we want the highest wind speeds to get the highest rank
        df1['Ranked']=df1['Max U10'].rank(ascending=0) 
        df1=df1.sort_values(by=['Ranked'])
        ranklist1=df1['Ranked'].tolist()
        windlist1=df1['Max U10'].tolist()
    
        windlist1=windlist1[::-1]  
        ratio=len(windlist1)/38.
        #fit the different extreme value distributions
        try:
            LMU=lmoments.samlmu(windlist1)
       
            gevfit = lmoments.pelgev(LMU)      
            expfit = lmoments.pelexp(LMU)
            gumfit = lmoments.pelgum(LMU)
            weifit = lmoments.pelwei(LMU)
            gpafit = lmoments.pelgpa(LMU)            
            
            gevST = lmoments.quagev(1.0-1./T, gevfit)
            expST = lmoments.quaexp(1.0-1./T, expfit)  
            gumST = lmoments.quagum(1.0-1./T, gumfit)
            weiST = lmoments.quawei(1.0-1./T, weifit)
            gpaST = lmoments.quagpa(1.0-1./T, gpafit)
            ratiolist.append(ratio)
            for t,tl in zip(range(0,len(bootstrapgev)),tlist):
                bootstrapgev[t].append(np.interp(tl,T,gevST))
                bootstrapexp[t].append(np.interp(tl,T,expST))
                bootstrapgum[t].append(np.interp(tl,T,gumST))
                bootstrapwei[t].append(np.interp(tl,T,weiST))
                bootstrappar[t].append(np.interp(tl,T,gpaST))
            
        except TypeError:
            loop=loop+1
    
        loop=loop+1
# =============================================================================
# Store the different bootstrap-dictionaries. We do this as both parts (previous + following)
# take very long. 
# =============================================================================
np.savetxt('ratios_'+str(basin)+'.txt',ratiolist)
np.save('bootstrapgev_'+str(basin)+'.npy',bootstrapgev)
np.save('bootstrapexp_'+str(basin)+'.npy',bootstrapexp)
np.save('bootstrapgum_'+str(basin)+'.npy',bootstrapgum)
np.save('bootstrapwei_'+str(basin)+'.npy',bootstrapwei)
np.save('bootstrappar_'+str(basin)+'.npy',bootstrappar)

# =============================================================================
# Create the figure
# =============================================================================
fig=plt.figure(figsize=(4*3.13,4*3.13))
gs=gridspec.GridSpec(3,2,wspace=0.25,hspace=0.5)
ax=[[] for _ in range(0,6)] 
plt.rcParams.update({'font.size':12})

for basin,basinlabel,idx,name,gs1,gs2 in zip(['EP','NA','NI','SI','SP','WP'],['Eastern Pacific','North Atlantic','North Indian','South Indian','South Pacific','Western Pacific'],range(0,6),['a)','b)','c)','d)','e)','f)'],[0,0,1,1,2,2],[0,1,0,1,0,1]):

    ax[idx]=plt.subplot(gs[gs1,gs2])

    windint=np.loadtxt('STORM_wind_'+str(basin)+'.txt') 
    ax[idx].text(-0.20,0.95,name,fontweight='bold',transform=ax[idx].transAxes)  
    
    for bootstraplist,color,label in zip(['bootstrapgev','bootstrapexp','bootstrapgum','bootstrapwei','bootstrappar'],['mediumpurple','cadetblue','tan','darkseagreen','darksalmon'],['GEV','Exponential','Gumbel','Weibull','Pareto']):
            bootstrap=np.load(str(bootstraplist)+'_'+str(basin)+'.npy',allow_pickle=True,encoding='latin1').item()
            ratio=np.loadtxt('ratios_'+str(basin)+'.txt')
            meanlist=[]
            ci_under=[]
            ci_upper=[]
            for t in range(0,len(bootstrap)):
                meanlist.append(np.mean(bootstrap[t]))
                ci_under.append(np.mean(bootstrap[t])-1.96*np.std(bootstrap[t]))
                ci_upper.append(np.mean(bootstrap[t])+1.96*np.std(bootstrap[t]))
            
            del bootstrap
            tlist=np.arange(0.1,round(np.mean(ratio)*1010,1),0.1)+1
            ax[idx].plot(tlist/np.mean(ratio),meanlist,color,linewidth=2,label=label)

            ax[idx].fill_between(np.array(tlist/np.mean(ratio)),np.array(ci_upper),np.array(ci_under),facecolor=color,alpha=0.2)
 
    ax[idx].set_xlim([0.1,1000])
    ax[idx].set_xscale('log')
    ax[idx].set_xlabel('Return period (yr)')
    ax[idx].set_ylabel('Maximum wind speed (m/s)')
    ax[idx].set_title(basinlabel)
    ax[idx].plot(T,windint,color='black', label='Empirical\n10,000 years')  
   
ax[5].legend(bbox_to_anchor=(1.05,1.03),loc='upper left')
plt.show()