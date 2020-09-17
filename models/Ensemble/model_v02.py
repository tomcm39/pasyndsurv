# luk, donnachie (equal contrib), mcandrew

import sys
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#goal(9/14): create equally weighted ensemble for all fips for all weeks ahead using all TW

def equalEnsemble(probs):
    """ 
        Equally Weighted ensemble model equation
        Inputs: probs is a list of forecast probabilities 
        Outputs: the EW ensemble prob
    """

    numModels = len(probs)  # number of models
    probs = np.array(probs) # convert to numpy array

    if np.any( np.isnan(probs) ): #originally written as np.nan but I think it's supposed to be np.isnan
        print( "A forecast contained a NAN" )
    else:
        ensembleProb = sum(probs) / numModels #add up the number of probabilities and divide by to find average
        return ensembleProb

if __name__ == "__main__":

    forecastData = pd.read_csv("../../scores/fulldens.csv") #pull sample forecast data from git repo
    forecastData = forecastData.replace(np.nan, 0)

    # we need to fill up this dictionary with forecasts
    singleEWForecast = { "numnewcases_leftbin"   :[]
                         ,"numnewcases_rightbin" :[]
                         ,"numnewcases_midbin"   :[]
                         ,"fips"                 :[]
                         ,"weekahead"            :[]
                         ,"forecastTW"           :[]
                         ,"prob"                 :[]
    }

    # lets start with iterated for loops:
    # 1. forecastTW
    # 2. Fips
    # 3. weekahead
    # in the innermost loop you'll include your below code
    #for TW, data in forecastData.groupby(['forecastTW']):
        #singleEWForecast['forecastTW'].append(TW)

    for forecastTW, data in forecastData.groupby(['forecastTW']):

        for fips, data in forecastData.groupby(['fips']):

            for weekahead, data in forecastData.groupby(['weekahead']):

                #this for loop gets us the mid point of all the bins
                for (left_bin,right_bin), data in forecastData.groupby(['numnewcases_leftbin','numnewcases_rightbin']):
                    singleEWForecast["numnewcases_leftbin"].append(left_bin) #append data to dictionary above 
                    singleEWForecast["numnewcases_rightbin"].append(right_bin)
                    numnewcases_mid = (left_bin + right_bin) / 2.0
                    singleEWForecast["numnewcases_midbin"].append(numnewcases_mid)
                    singleEWForecast['fips'].append(fips)
                    singleEWForecast['weekahead'].append(weekahead)
                    singleEWForecast['forecastTW'].append(forecastTW)
                    averageProb_OneBin = equalEnsemble(data.prob)
                    singleEWForecast["prob"].append(averageProb_OneBin)


    
    # we can just use this code
    singleEWForecast = pd.DataFrame(singleEWForecast)
    #singleEWForecast.to_excel("Ensemble.xlsx")

    

