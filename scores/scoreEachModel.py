#mcandrew

import sys
import numpy as np
import pandas as pd
import itertools

def removeModelFromPath():
    sys.path = sys.path[:-1]

def subset2EW(tws,ew):
    subsetofTws = []
    for tw in tws:
        if tw >= ew:
            subsetofTws.append(tw)
    return subsetofTws

sys.path.append("../data/weekpp/")
from week import week 

def subsetMostRecentData24weeks(mostRecentData,MW):
      trueData = mostRecentData[ (mostRecentData.modelweek <= MW+4) & (mostRecentData.modelweek > MW) ]
      trueData['weekahead'] = trueData.modelweek - MW
      trueData = trueData.drop(columns=['modelweek'])

      return trueData

def scoreModelForecasts(forecasts, trueData):
    forecastAndTruth = forecasts.merge( trueData, on = ['fips','weekahead'] )
    forecastAndTruth = forecastAndTruth[ (forecastAndTruth.numnewcases_leftbin <= forecastAndTruth.dohweb__numnewpos) & (forecastAndTruth.numnewcases_rightbin > forecastAndTruth.dohweb__numnewpos)   ]

    forecastAndTruth['score'] = np.log(forecastAndTruth.prob)
    forecastAndTruth = forecastAndTruth[['modelname','fips','weekahead','prob','score']]

    return forecastAndTruth

if __name__ == "__main__":

    #models
    #---------------------------------------------------------------------------
    sys.path.append('../models/theScienceDogs/')
    from packagedModelTheScienceDogs import theScienceDogs
    
    sys.path.append('../models/covidCrushers/')
    from packagedModelCovidCrushers  import covidCrushers

    sys.path.append('../models/CovidSeers/')
    from packagedModelCovidSeers  import covidSeers

    sys.path.append('../models/SIR/')
    from packagedModelBasicSIR  import basicSIR


    models = [theScienceDogs, covidCrushers, covidSeers, basicSIR]
    #---------------------------------------------------------------------------
    
    casesData = pd.read_csv("../data/cases/PATrainingDataCases.csv")

    fips   = list( casesData.fips.unique() )

    trainingWeeks = sorted( casesData.trainingweek.unique() )
    trainingWeeksFrom202020 = subset2EW(trainingWeeks,202020)

    maxTrainingWeek = max(trainingWeeksFrom202020)
    mostRecentData = casesData.loc[ casesData.trainingweek == maxTrainingWeek, ["modelweek","dohweb__numnewpos","fips"] ]
    
    firsttimewrite = 1
    for model in models:

        # instantiate model
        forecastmodel = model()
        forecastmodel.modeldesc()
        
        for TW in trainingWeeksFrom202020:
  
            # subset to specific training data and add to model
            trainingDataForThisWeek = casesData[casesData.trainingweek==TW]
            trainingDataForThisWeek.loc[:,'dohweb__numnewpos'] = trainingDataForThisWeek.dohweb__numnewpos.replace(np.nan,0.)

            forecastmodel.addTrainingData( trainingDataForThisWeek )

            # make forecasts for every fip in the training data
            forecasts = forecastmodel.makeForecasts()
            forecasts['forecastTW'] = TW

            try:
                forecasts = forecasts.drop(columns=['index'])
            except KeyError:
                pass

            # Find the most recent Epiweek and corresponding model week
            mostRecentEW = TW-1
            MW = week(epiweek=str(mostRecentEW)).modelweek

            # subset the most recent data to training week +1, 2, 3, and 4.
            # Then consider the most recent data the truth and score the forecasts
            trueData = subsetMostRecentData24weeks(mostRecentData,MW)
            forecastAndTruth = scoreModelForecasts( trueData, forecasts  )
            forecastAndTruth['trainingweek'] = TW

            # write out densitiies and scores
            if firsttimewrite==1:
                # write out full probabilistic densities
                forecasts.to_csv("./fulldens.csv",header=True,mode="w",index=False)

                # write out scores for model
                forecastAndTruth.to_csv("./scores.csv",header=True,mode="w",index=False)
                
                firsttimewrite=0
            else:
                forecasts.to_csv("./fulldens.csv",header=False,mode="a",index=False)
                forecastAndTruth.to_csv("./scores.csv",header=False,mode="a",index=False)
