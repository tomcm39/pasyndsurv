#Damon Luk
#script that graphs new number of COVID-19 cases versus poverty levels in the past four weeks

import pandas as pd 
import matplotlib.pyplot as plt 
import math

povertyAndConfirmed = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/models/CelticClassic/Poverty_Confirmed_Merged.xlsx")
maxTrainingWeek = povertyAndConfirmed.trainingweek.max()
maxEpiWeek = povertyAndConfirmed.epiweek.max() #most recent week

#i want to write a for loop that goes from max epi week to 4 weeks before that and each time I have to figure out a way to subset the data to only that epiweek and graph it. 

for data in range(4):

    subsetPoverty = povertyAndConfirmed [(povertyAndConfirmed.trainingweek == maxTrainingWeek) & (povertyAndConfirmed.epiweek == maxEpiWeek)]
    subsetPoverty = subsetPoverty[ ["fips", "epiweek", "POVALL_2018", "MEDHHINC_2018", "cdcili__ili","dohweb__numnewpos", "census", "trainingweek"] ]
    #print(subsetPoverty.head(10)) #to test the above 

    fig,axs = plt.subplots(1,2)
    plt.style.use('fivethirtyeight')

    ax = axs[0]
    for fip, data in subsetPoverty.groupby(['fips']):
        ax.scatter(math.log10(data.POVALL_2018), data.dohweb__numnewpos, linewidth = 2, alpha = 0.55 )

    ax.tick_params(which = 'both', labelsize = 8)
    ax.set_ylabel('Number of new COVID-19 cases')
    ax.set_xlabel('Log10 of Estimate of people of all ages in poverty in 2018')

    ax = axs[1]
    for fip, data in subsetPoverty.groupby(['fips']):
        ax.scatter(math.log(data.MEDHHINC_2018), data.dohweb__numnewpos, linewidth = 2, alpha = 0.55)

    ax.tick_params(which = 'both', labelsize = 8)
    ax.set_ylabel('Number of new COVID-19 cases')
    ax.set_xlabel('Log10 of Estimate of median household 2018')


    fig.set_tight_layout(True)
    fig.set_size_inches(10,5)
    fig.suptitle(maxEpiWeek)

    plt.show()

    maxEpiWeek -= 1





