#mcandrew, nelson, liu

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    allData = pd.read_csv("data/cases/PATrainingDataCases.csv")
# iterating the columns 
    for col in allData.columns: 
        print(col) 

    #CODE TO SUBSET TO INDIVIDUAL COUNTIES
    #mostrecentweek = allData.trainingweek.max()
    #singleCounty = allData[ (allData.fips == 42091) & (allData.trainingweek==mostrecentweek)  ]

    #run loop to get N C & S through each county
    for (fips,subsetdata) in allData.groupby(["fips"]):
        newPos = subsetdata.dohweb__numnewpos
        census = subsetdata.census
        totPos = np.cumsum(newPos)
 
        #create var N: new cases per population
        N = newPos/census

        #create var C: cumulative cases per population
        C = totPos/census
    
        #create var S : cumulative non-cases per population
        S = 1 - C

