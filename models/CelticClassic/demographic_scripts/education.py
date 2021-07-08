#Damon Luk
#Script that graphs number of new cases vs education data in the most recent week

import pandas as pd 
import matplotlib.pyplot as plt 
import math

educationAndConfirmed = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/models/CelticClassic/demo_confirmed(excel)/Education_Confirmed_Merged.xlsx")
maxTrainingWeek = educationAndConfirmed.trainingweek.max()
maxEpiWeek = educationAndConfirmed.epiweek.max()

subsetEducation = educationAndConfirmed [(educationAndConfirmed.trainingweek == maxTrainingWeek) & (educationAndConfirmed.epiweek == maxEpiWeek)]
subsetEducation = subsetEducation[ ["fips", "2013 Rural-urban Continuum Code", "2013 Urban Influence Code", "epiweek", "cdcili__ili","dohweb__numnewpos", "census", "trainingweek", "Percent of adults with less than a high school diploma, 2014-18", "Percent of adults with a high school diploma only, 2014-18", "Percent of adults with a bachelor's degree or higher, 2014-18"]]
subsetEducation.rename(columns = {"Percent of adults with less than a high school diploma, 2014-18": "percentLessHigh", "Percent of adults with a high school diploma only, 2014-18": "percentOnlyHigh", "Percent of adults with a bachelor's degree or higher, 2014-18": "percentBachelor" }, inplace = True)

fig, axes = plt.subplots(1,3)
plt.style.use('fivethirtyeight')

ax = axes[0]
for fip, data in subsetEducation.groupby(['fips']):
    ax.scatter(data.percentLessHigh, data.dohweb__numnewpos, linewidth = 2, alpha = 0.55 )

ax.tick_params(which = 'both', labelsize = 8)
ax.set_ylabel('Number of new COVID-19 cases')
ax.set_xlabel('Per. adults with less than a high school diploma')

ax = axes[1]
for fip, data in subsetEducation.groupby(['fips']):
    ax.scatter(data.percentOnlyHigh, data.dohweb__numnewpos, linewidth = 2, alpha = 0.55)

ax.tick_params(which = 'both', labelsize = 8)
ax.set_ylabel('Number of new COVID-19 cases')
ax.set_xlabel('Per. adults with a high school diploma only')

ax = axes[2]
for fip, data in subsetEducation.groupby(['fips']):
    ax.scatter(data.percentBachelor, data.dohweb__numnewpos, linewidth = 2, alpha = 0.55)

ax.tick_params(which = 'both', labelsize = 8)
ax.set_ylabel('Number of new COVID-19 cases')
ax.set_xlabel('Per. adults with a bachelor degree or higher')


#fig.set_tight_layout(True)
fig.set_size_inches(10,5)
fig.suptitle(maxEpiWeek)

plt.show()




