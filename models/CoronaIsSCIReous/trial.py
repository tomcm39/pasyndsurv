import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------


#To get population we need to get data from the population source and this will be our N 

popData = pd.read_csv('../../data/populationEstimates/PApopdata.csv')

countyPopulation = int(popData.POP[popData.countyfips == 42095]) 



#---------------------------------------------------------------------------------------

#To get the deaths in a specific county we need to get data from the deaths source and this will be our R for now

deathData = pd.read_csv('../../models/CoronaIsSCIReous/jhuCSSEdata_2020-06-18-22.csv')

#To get the deaths of a specific county as of the latest date (june-17) (the number 3 is the index to get the count)
#changed the last column in jhucsse data from 'count' to 'deaths'

specificCountyDeaths = np.array((deathData[deathData.FIPS==42095].deaths))

#To get the data weekly
specificCountyDeathsWeekly = [] 

for i in range (len(specificCountyDeaths)):
    if(i%7 == 0):
        specificCountyDeathsWeekly.append(specificCountyDeaths[i])
weeklyDeaths = np.array(specificCountyDeathsWeekly)
#---------------------------------------------------------------------------------------

#This is to get the information of active [infected] cases for only Northampton County[fips =42095 ] 

data = pd.read_csv('../../data/cases/PATrainingDataCases.csv')

# I will only consider data available as of 2020W23 (the training week) 
# and focus on data in the current 2020 season, data more recent than epidemic week 2019W52

specificTrainingWeek = data[(data.trainingweek==202023) & (data.epiweek>201952)] 
specificTrainingWeek = specificTrainingWeek.replace(np.nan,0)


'''
To find the most recent epiweek and most recent training week
#mostRecentEW=data.epiweek.max()
#mostRecentTW=data.trainingweek.max()
'''
specificRegion = specificTrainingWeek[specificTrainingWeek.fips==42095] 
newCases = np.array(specificRegion.covidtracker__numnewpos)

#print(newCases)

#---------------------------------------------------------------------------------
#Graph of the real-time data 

N = countyPopulation
I = newCases
R = weeklyDeaths[2:21]

#in order to get the susceptibles the arrays I and R need to be exactly of the same size


'''
S = np.array([R + I])
SN = N - S
print(SN)
plt.xlabel("EpiWeek")


plt.plot(SN)


plt.show()

'''

