
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#goal(week of 9/7): make an equally weighted ensemble for one week ahead, one fip for all the case bins with the most recent training week

#pull sample forecast data from git repo
forecastData = pd.read_csv("/Users/damonluk924/Desktop/pasyndsurv/scores/fulldens.csv")
maxTrainingWeek = forecastData.forecastTW.max()
#print (forecastData.head(20)) #check to see if it was read in correctly

#subset sample forecast data
forecastDataEz = forecastData[ (forecastData.weekahead == 1) & (forecastData.fips == 42001) & (forecastData.forecastTW == maxTrainingWeek)] #subsets the data to what we need for our goal
#print(forecastDataEz.head(20)) #check to see our subset

midbin = []
#this for loop gets us the mid point of all the bins
for (left_bin,right_bin), data in forecastDataEz.groupby(['numnewcases_leftbin','numnewcases_rightbin']):
    numnewcases_mid = (left_bin + right_bin) / 2.0
    midbin.append(numnewcases_mid)

#print(midbin)



#numnewcases_leftbin = forecastData.numnewcases_leftbin #sets variable to all the left bins


#numnewcases_rightbin = forecastData.numnewcases_rightbin #sets variable to all the right bins


#prob = forecastData.prob

#fips = forecastData.fips

#equally weighted ensemble model equation
#input sum of the probabilities and the number of models
#def equalEnsemble(sumProb, numModels):
    #ensembleProb = (1/numModels) * sumProb
    #return ensembleProb