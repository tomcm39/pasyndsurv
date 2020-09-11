
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#goal(week of 9/7): make an equally weighted ensemble for one week ahead, one fip for all the case bins with the most recent training week

#pull sample forecast data from git repo
forecastData = pd.read_csv("../../scores/fulldens.csv")
maxTrainingWeek = forecastData.forecastTW.max()
#print (forecastData.head(20))

#subset sample forecast data
forecastDataEz = forecastData[ (forecastData.weekahead == 1) & (forecastData.fips == 42001) & (forecastData.forecastTW == maxTrainingWeek)] #subsets the data to what we need for our goal
#print(forecastDataEz.head(20))

#equally weighted ensemble model equation
#input sum of the probabilities and the number of models
def equalEnsemble(sumProb, numModels):
    ensembleProb = sumProb / numModels
    return ensembleProb

midpoint = []
average =[]
#this for loop gets us the mid point of all the bins
for (left_bin,right_bin), data in forecastDataEz.groupby(['numnewcases_leftbin','numnewcases_rightbin']):
    numnewcases_mid = (left_bin + right_bin) / 2.0
    midpoint.append(numnewcases_mid)
    averageProb_OneBin = equalEnsemble(data.prob.sum(), 4)
    #print(averageProb_OneBin)
    average.append(averageProb_OneBin)

midbin = pd.DataFrame(midpoint, columns = ['numnewcases_mid'])
averageProb = pd.DataFrame(average, columns = ['averageProb'])

#print(midbin)
#print(averageProb)

ensembleDataEz = pd.merge(midbin, averageProb, left_index = True, right_index = True)
print(ensembleDataEz)

sns.set()

sns.lineplot(x = 'numnewcases_mid', y = 'averageProb', data = ensembleDataEz).set_title("Equally Weighted Ensemble for 1 Week ahead in FIP 42001")
plt.xlim(0 , 80)
plt.show()

#creates data frame with same format as other models
#temporarily takes first 100 since we are only look at fips 42001 with weekahead 1
ensembleForecast = pd.DataFrame({
    'numnewcases_leftbin' : forecastData.numnewcases_leftbin[0:100],
    'numnewcases_rightbin' : forecastData.numnewcases_rightbin[0:100],
    'weekahead' : forecastData.weekahead[0:100],
    'prob' : average,
    'fips' : forecastData.fips[0:100]})
#print(ensembleForecast)
