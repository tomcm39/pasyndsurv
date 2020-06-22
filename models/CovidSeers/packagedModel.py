# mcandrew,andrew,poplar

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy
from statsmodels.tsa.arima_model import ARIMA # note the import here (you may need to pip3 install statsmodel)

class CovidSeers(object):
    # Copy this section 
    def __init__(self):
        pass

    def addTrainingData(self,trainingdata):
        self.trainingdata = trainingdata

    # Copy this function
    def modeldesc(self):
        print("The CovidSeers KNN-ARIMA model")
    
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
            
            forecastData = self.makeForecastForOneRegion(regiondata,self.trainingdata)
            
            forecastData['fips'] = fip
            
            forecastsForAllRegions = forecastsForAllRegions.append(forecastData)
        return forecastsForAllRegions
    
    # -------------------------------This is your model code------------------------
    def makeForecastForOneRegion(self,regiondata, singleTrainingWeek):
        import scipy

        trainingData = regiondata[['modelweek','cdcili__wili','covidtracker__numnewtest','jhucsse__numnewpos','dohweb__numnewpos']].set_index('modelweek')
        trainingData = trainingData.replace(np.nan,0.)
        
        targetFIP = float(regiondata.fips.iloc[0])

        # We need to develop a matrix D that has rows and columns  equal to the number of counties in PA.
        # For entry i,j of the matrix D[i,j], we will store the distance between the time series for FIPS=i and FIPS=j
        # I am going to pick the easiest distance. You should mess around with different distance metrics!

        def euclideanDistance(timeSeries1, timeSeries2):
            """ Compute the Euclidean distance between two time series.
                The Euclidean distance between ts1 and ts2 is defined as the sum of the squared distance between corresponding entries of the time series. 
            """
            squaredDiffs =  [ (t1-t2)**2 for (t1,t2) in zip(timeSeries1,timeSeries2)  ] # square all the differences between time series values
            sumOfSquaredDiffs = sum(squaredDiffs)
            return sumOfSquaredDiffs

        # inside len, outputs the number of unique fips and len answers the question "how long is that vector?"
        FIPS = singleTrainingWeek.fips.unique()
        numberOfFIPS = len(FIPS) 
        D = np.zeros( (numberOfFIPS, numberOfFIPS) ) # build D matrix with FIPS number of rows and FIPS number of columns

        print(D.shape) # print out the number of rows and number of columns ofour Distance matrix D (good check).

        # now we need to be careful about how we enter values in this matrix.
        # I'm making a policy here the FIPS in rows and columns are sorted.

        FIPS = sorted(FIPS)

        # i want to build a map from the index of my matrix D to the FIPS number.
        # To do this, I'll use a dictionary. Take a look at dictionaries here = https://docs.python.org/3/tutorial/datastructures.html#dictionaries
        FIPS2index = { fip:x for (x,fip) in enumerate(FIPS)  }
        index2FIPS = { x:fip for (x,fip) in enumerate(FIPS)  }

        # This loop will run through every combination of county time series and compute the euclidean distance between county i and county j

        for (i,fipI) in enumerate(FIPS): # take a look at what the enumerate function does here = https://book.pythontips.com/en/latest/enumerate.html
            for (j,fipJ) in enumerate(FIPS):
                # find time series for FIP i
                fipItimeSeries = singleTrainingWeek[ singleTrainingWeek.fips==fipI]

                # find time series for FIP j
                fipJtimeSeries = singleTrainingWeek[ singleTrainingWeek.fips==fipJ]

                D[i,j] = euclideanDistance( fipItimeSeries.dohweb__numnewpos, fipJtimeSeries.dohweb__numnewpos )

        # now we will build a forecast for each of the 5 closest counties to our "target" county.

        K=5
        KclosestCounties = np.argsort( D[:, FIPS2index[targetFIP]] )[:K] # argsort = https://numpy.org/devdocs/reference/generated/numpy.argsort.html

        closestFIPS = [ index2FIPS[index] for index in KclosestCounties]

        stepsize = 5
        probBins = np.arange(0,5*10**2,stepsize)
        distOfForecasts = np.zeros( (K, len(probBins),4) )
        for i,fip in enumerate(closestFIPS):
            fipData = singleTrainingWeek[ singleTrainingWeek.fips==fip ] # take a single FIPS worth of data
            fipData = fipData[['modelweek','dohweb__numnewpos']].set_index(['modelweek']) #  only need these two columns

            # build an ARIMA model
            # Documentation for ARIMA from Statmodels is here
            # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html

            fipData = fipData.replace(np.nan,0.)
            arimaModel = ARIMA( fipData, order=(1,1,1) )
            fittedModel = arimaModel.fit()
                
            lastWeekOfData = fipData.index.max()
            mn,stderr,confint = fittedModel.forecast(steps=4)

            for weekahead in np.arange(1,4+1,1):
                dist = scipy.stats.norm(loc = mn[weekahead-1] , scale = stderr[weekahead-1] )

                probs = [ float(dist.cdf(numOfNewCases+stepsize) - dist.cdf(numOfNewCases) ) for numOfNewCases in np.arange(0,5*10**2,stepsize)]
                distOfForecasts[i,:,weekahead-1] = probs

        distOfForecasts = distOfForecasts.mean(axis=0) # average K prob dists
        distOfForecasts = distOfForecasts / distOfForecasts.sum(0) # normalize prob dists

        nBins = len(probBins)
        allForecastData = pd.DataFrame()
        for weekahead in np.arange(1,4+1,1):
            forecastData = { 'numnewcases_leftbin': probBins
                             ,'numnewcases_rightbin':np.arange(stepsize,5*10**2+stepsize,stepsize)
                             ,'weekahead':[weekahead]*nBins
                             ,'prob':distOfForecasts[:,weekahead-1]}
            allForecastData = allForecastData.append( pd.DataFrame(forecastData)  )
        return allForecastData
        
if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[(allData.fips == 42095) & (allData.trainingweek == mostrecentweek)]
    singleTW = allData[(allData.trainingweek == mostrecentweek)]

    # Looks like a few NANS show up in the data. I'm not sure why yet, but those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0
    singleCounty = singleCounty.replace(np.nan, 0.)
    
    covidseers = CovidSeers()
    covidseers.addTrainingData(singleCounty)

    f = covidseers.makeForecastForOneRegion(singleCounty,singleTW)

