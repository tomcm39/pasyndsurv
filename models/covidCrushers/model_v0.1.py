# mcandrew,kline,lin

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
#from statsmodels.tsa.holtwinters import ExponentialSmoothing as eSmooth  # X marks the import statement!
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as eSmooth

if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[(allData.fips == 42095) & (allData.trainingweek == mostrecentweek)]

    # Looks like a few NANS show up in the data. I'm not sure why yet, but those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0
    singleCounty = singleCounty.replace(np.nan, 0.)
    
    # Documentation on how to use Holt-Winters for Python is here
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    # notice the import state above marked with an X

    trainingData = singleCounty[['modelweek', 'dohweb__numnewpos']].set_index('modelweek')
    covidCrushersModel = eSmooth(trainingData, trend='add')
    fittedCovidCrushersModel = covidCrushersModel.fit()

    # predictions for the next 4 weeks
    _4WeekAheadForecast    = fittedCovidCrushersModel.forecast(4)
    _4WeekAheadForecastObj = fittedCovidCrushersModel.get_forecast(4)
 
    # NOW is the hard part.
    # lets look at a plot of the data and the forecast
    
    fig, ax = plt.subplots()  # setup a plot environment
    ax.plot(trainingData.index   , trainingData.dohweb__numnewpos, color='b', alpha=0.50, label="DOH data")
    ax.scatter(trainingData.index, trainingData.dohweb__numnewpos, s=30, color='b', alpha=0.50)

    # build a list of forecasted epiweeks
    lastWeekOfData = trainingData.index.max()
    forecastedEpiweeks = [lastWeekOfData+x for x in np.arange(1, 4+1)]
    
    ax.plot(forecastedEpiweeks, _4WeekAheadForecast, color='k', label="prediction")  # now I'll plot the forecasted weeks and the predictions

    # add in confidence intervals
    confidenceIntervals = _4WeekAheadForecastObj.conf_int()
    standardErrors      = _4WeekAheadForecastObj.se_mean
    
    ax.fill_between(x = forecastedEpiweeks, y1= confidenceIntervals.iloc[:,0], y2 = confidenceIntervals.iloc[:,1],color='blue', alpha=0.20)
    
    # i should label my axes. No one likes bare axes, no one!
    ax.set_xlabel("Model week")
    ax.set_ylabel("Number of new COVID cases")

    ax.set_ylim(-5,800)

    # lets check out my legend too. Note the two "label" statements above
    ax.legend()
    
    # and then I'll take a look at what we did!
    plt.show()

    # TODOS:
    # Holt Winter's is a point prediction. How will you turn it into a probabilistic forecast, a probability distribution over future values?
    # Take a look at FIPS = 42045. Anything we need to fix about this model?
    # So many more for Alex and Kenny to list
