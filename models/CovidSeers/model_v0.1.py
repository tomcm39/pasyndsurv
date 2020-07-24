#mcandrew, Poplar, and Andrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from statsmodels.tsa.arima_model import ARIMA # note the import here (you may need to pip3 install statsmodel)


if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()

    singleTrainingWeek = allData[ (allData.trainingweek==mostrecentweek)  ]
    singleTrainingWeek = singleTrainingWeek.replace(np.nan,0.)

    targetFIP = 42007
    singleCounty = allData[ (allData.fips == targetFIP) & (allData.trainingweek==mostrecentweek)  ]

    # Looks like a few NANS show up in the data. I'm not sure why yet, but those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0
    singleCounty = singleCounty.replace(np.nan,0.)

    # ---------------------------------------------------------------------------
    # TRAIN MODEL Below
    # Your model should take in data from Single County and output forecasts (predictions) for at least 1 week ahead. 
    # ---------------------------------------------------------------------------
    
    trainingData = singleCounty[['modelweek','cdcili__wili','covidtracker__numnewtest','jhucsse__numnewpos','dohweb__numnewpos']].set_index('modelweek')

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
    
    # why are all the diagonal entries of D zero? Seems weird, or does it?

    
    # now we will build a forecast for each of the 5 closest counties to our "target" county.
    
    K=5
    KclosestCounties = np.argsort( D[:, FIPS2index[targetFIP]] )[:K] # take a look at what argsort does here = https://numpy.org/devdocs/reference/generated/numpy.argsort.html

    closestFIPS = [ index2FIPS[index] for index in KclosestCounties]
    
    fourWeekAheadForecasts = np.zeros( (K,5) ) # 5 closest time series by 4 week ahead forecasts (plus a 0 week ahead forecast)
    for i,fip in enumerate(closestFIPS):
        fipData = singleTrainingWeek[ singleTrainingWeek.fips==fip ] # take a single FIPS worth of data
        fipData = fipData[['modelweek','dohweb__numnewpos']].set_index(['modelweek']) #  only need these two columns
        
        # build an ARIMA model
        # Documentation for ARIMA from Statmodels is here
        # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html
        arimaModel = ARIMA( fipData, order=(1,0,0) )
        fittedModel = arimaModel.fit()

        lastWeekOfData = fipData.index.max()
        forecasts = fittedModel.predict(start=len(fipData),end=len(fipData)+4) # some issues will arise here when we are in december
        
        fourWeekAheadForecasts[i,:] = forecasts
        
    # lets take the average forecast from all 5 models as the forecasts for our county.
    mostRecentModelWeek = fipData.index.max() # the index is modelweek
    avgForecast = fourWeekAheadForecasts.mean(0)

    fig,ax = plt.subplots()

    ax.plot( singleCounty.modelweek, singleCounty.dohweb__numnewpos, color='blue',alpha=0.70 )

    ax.plot( np.arange(mostRecentModelWeek,mostRecentModelWeek+4+1) , avgForecast, color ='red',alpha=0.80, label = "Avg Forecast")
    for i,closeCountyForecast in enumerate(fourWeekAheadForecasts):
        if i==0:
            clr='purple'
            lbl="Target FIP forecast"
            alpha=1.0
        elif i==1:
            clr='gray'
            lbl = "Close County Forecast"
            alpha=0.5
        else:
            clr="gray"
            lbl="_none_"
            alpha=0.5
        ax.plot( np.arange(mostRecentModelWeek,mostRecentModelWeek+4+1) , closeCountyForecast, color =clr,alpha=alpha, label = lbl)

    ax.set_xlabel("Model week (number of epiweeks since 1970W1)")
    ax.set_ylabel("Number of new COVID-19 cases")

    ax.legend()
    
    plt.show()

    # TODO
    # these forecasts are POINT forecasts. We need a proabbilistic distribuion over the number of new cases.
    # lets think about how to do that
    
