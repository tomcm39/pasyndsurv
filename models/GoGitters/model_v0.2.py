#mcandrew, nelson, liu

#new python script in folder
#load in pa training data
#load in census data
#create var N cumulative number of new positives new positives / total people in census
#create var C cumulative new pos / census
#to proportion of people in census – number of new positives/ number of people in census

#S = 1-C

#plot some C vars of different fips
#matplotlib

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    allData = pd.read_csv("data/cases/PATrainingDataCases.csv")

    #subset data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[ (allData.fips == 42091) & (allData.trainingweek==mostrecentweek)  ]

    newPos = singleCounty.dohweb__numnewpos
    census = singleCounty.census
    totPos = np.cumsum(newPos)
 
    #create var N: new cases per population
    N = newPos/census

    #create var C: cumulative cases per population
    C = totPos/census
    
    #create var S : cumulative non-cases per population
    S = 1 - C

    print(N, C, S)