#Damon Luk
#script that graphs cumulative number of COVID cases versus poverty level 

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import math 

povertyAndConfirmed = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/models/CelticClassic/Poverty_Confirmed_Merged.xlsx")
maxTrainingWeek = povertyAndConfirmed.trainingweek.max()

subsetPoverty = povertyAndConfirmed[(povertyAndConfirmed.trainingweek == maxTrainingWeek)]
subsetPoverty = subsetPoverty[ ["fips", "epiweek", "POVALL_2018", "MEDHHINC_2018", "cdcili__ili","dohweb__numnewpos", "census", "trainingweek"] ]

#print (subsetPoverty.head(20)) #checking to see that the subset is working correctly

def addCumulativeCases(subset):
    subset.loc[:,'cumulativeCases'] = np.cumsum(subset.dohweb__numnewpos)
    return subset

subsetPoverty = subsetPoverty.groupby(['fips']).apply(addCumulativeCases)

#print (subsetPoverty.head(20)) #checking to see if cumulative cases got added
maxEpiWeek = subsetPoverty.epiweek.max()
subsetPoverty = subsetPoverty[(subsetPoverty.epiweek == maxEpiWeek)]

fig,axs = plt.subplots(1,2)
plt.style.use('fivethirtyeight')

ax = axs[0]
for fip, data in subsetPoverty.groupby(['fips']):
    ax.scatter(data.POVALL_2018, data.cumulativeCases, linewidth = 2, alpha = 0.55 )

ax.tick_params(which = 'both', labelsize = 8)
ax.set_ylabel('Cumulative Number of COVID-19 cases')
ax.set_xlabel('Estimate of people of all ages in poverty in 2018')

ax = axs[1]
for fip, data in subsetPoverty.groupby(['fips']):
    ax.scatter(data.MEDHHINC_2018, data.cumulativeCases, linewidth = 2, alpha = 0.55)

ax.tick_params(which = 'both', labelsize = 8)
ax.set_ylabel('Cumulative Number of COVID-19 cases')
ax.set_xlabel('Estimate of median household 2018')


fig.set_tight_layout(True)
fig.set_size_inches(10,5)

plt.show()


