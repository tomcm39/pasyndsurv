#Andrew Donnachie, Damon Luk

import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#pull sample forecast data from git repo
forecastData = pd.read_csv("../../forecasts/theScienceDogsForecast.csv")

#subset sample forecast data
numnewcases_leftbin = forecastData.numnewcases_leftbin
numnewcases_rightbin = forecastData.numnewcases_rightbin

prob = forecastData.prob

fips = forecastData.fips

#equally weighted ensemble model equation
#input sum of the probabilities and the number of models
def equalEnsemble(sumProb, numModels):
    ensembleProb = (1/numModels) * sumProb
    return ensembleProb