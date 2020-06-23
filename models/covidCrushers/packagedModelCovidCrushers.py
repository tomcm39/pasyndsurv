# mcandrew,kline,lin

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
#from statsmodels.tsa.holtwinters import ExponentialSmoothing as eSmooth  # X marks the import statement!
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as eSmooth

class covidCrushers(object):
    # Copy this section 
    def __init__(self):
        pass

    def addTrainingData(self,trainingdata):
        self.trainingdata = trainingdata

    # Copy this function
    def modeldesc(self):
        print("The covidCrushers Holt_winters model")
    
    # Copy this function 
    def makeForecasts(self):
        forecastsForAllRegions = pd.DataFrame()
        for fip in self.trainingdata.fips.unique():
            if np.isnan(fip):
                continue
            
            print('Forecasting FIPS = {:05d}'.format(int(fip)))
            
            regiondata = self.trainingdata[self.trainingdata.fips==fip]

            if regiondata.shape[0]==0:
                continue
            
            if np.all(np.isnan(regiondata.dohweb__numnewpos)):
                continue
            
            forecastData = self.makeForecastForOneRegion(regiondata)
            
            forecastData['fips'] = fip
            
            forecastsForAllRegions = forecastsForAllRegions.append(forecastData)
        return forecastsForAllRegions
    
    # -------------------------------This is your model code------------------------
    def makeForecastForOneRegion(self,regiondata):
        import scipy
        
        regiondata= regiondata.replace(np.nan, 0.)
    
        # Documentation on how to use Holt-Winters for Python is here
        # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
        # notice the import state above marked with an X

        trainingData = regiondata[['modelweek', 'dohweb__numnewpos']].set_index('modelweek')
        covidCrushersModel = eSmooth(trainingData, trend='add')
        fittedCovidCrushersModel = covidCrushersModel.fit()

        # predictions for the next 4 weeks
        _4WeekAheadForecast    = fittedCovidCrushersModel.forecast(4)
        _4WeekAheadForecastObj = fittedCovidCrushersModel.get_forecast(4)

        # build a list of forecasted epiweeks
        lastWeekOfData = trainingData.index.max()
        forecastedEpiweeks = [lastWeekOfData+x for x in np.arange(1, 4+1)]

        # add in confidence intervals
        confidenceIntervals = _4WeekAheadForecastObj.conf_int()
        standardErrors      = _4WeekAheadForecastObj.se_mean

        _4WeekAheadForecast = list(_4WeekAheadForecast)
        standardErrors = list(standardErrors)
        
        forecastData = {'numnewcases_leftbin':[],'numnewcases_rightbin':[],'weekahead':[],'prob':[]}
        for weekahead in np.arange(1,4+1,1):
            dist = scipy.stats.norm(loc= _4WeekAheadForecast[weekahead-1] , scale = standardErrors[weekahead-1] )
            
            stepsize=5
            for numOfNewCases in np.arange(0,5*10**2,stepsize):
                probability = float(dist.cdf(numOfNewCases+stepsize) - dist.cdf(numOfNewCases) )

                forecastData['numnewcases_leftbin'].append(numOfNewCases)
                forecastData['numnewcases_rightbin'].append(numOfNewCases+stepsize)
                forecastData['prob'].append(probability)
                forecastData['weekahead'].append(weekahead)
        forecastData = pd.DataFrame(forecastData)

        # i restricted the range of values to forecast between 0 and 1000.
        def normalize(x):
            x['prob'] = x.prob/x.prob.sum()
            return x
        forecastData = forecastData.groupby(['weekahead']).apply(normalize).reset_index()
        forecastData['modelname'] = 'coviCrushers_HW'
        return forecastData

if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[(allData.fips == 42095) & (allData.trainingweek == mostrecentweek)]

    # Looks like a few NANS show up in the data. I'm not sure why yet, but those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0
    singleCounty = singleCounty.replace(np.nan, 0.)
    
    covidCrushers = forecastModel()
    covidCrushers.addTrainingData(singleCounty)

    f = covidCrushers.makeForecastForOneRegion(singleCounty)
