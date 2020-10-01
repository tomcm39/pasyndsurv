#mcandrew

import sys
import numpy as np
import pandas as pd

class ARLinRegRegress(object):
    # Copy this section 
    def __init__(self):
        pass

    def addTrainingData(self,trainingdata):
        self.trainingdata = trainingdata

    # Copy this function
    def modeldesc(self):
        print("The Science Dogs model - Lin Reg")
    
    # Copy this function 
    def makeForecasts(self):
        forecastsForAllRegions = pd.DataFrame()
        for fip in self.trainingdata.fips.unique():
            if np.isnan(fip):
                continue
            
            sys.stdout.write('\rForecasting FIPS = {:05d}\r'.format(int(fip)))
            sys.stdout.flush()
            
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
        from sklearn.linear_model import LinearRegression

        # Make prediction 1 week into the future
        # Covariates needed for model
        
        target = ['dohweb__numnewpos']

        regiondata['_1behindDOH'] = [np.nan] + list(regiondata.dohweb__numnewpos.iloc[:-1])
        covariates2predictCases = ['_1behindDOH']
        
        # Setup "X" data
        Xvariables = regiondata.loc[:,['modelweek']+covariates2predictCases]
        #Xvariables.loc[:,'modelweekPlusOne'] = Xvariables.modelweek.values+1
        #Xvariables = Xvariables.drop(columns=['modelweek'])

        # extract target variable 
        targetVariable = regiondata.loc[:,['modelweek']+target]

        # build data for model
        #XandTarget = Xvariables.merge(targetVariable, left_on=['modelweekPlusOne'], right_on = ['modelweek'])
        XandTarget = Xvariables.merge(targetVariable, left_on=['modelweek'], right_on = ['modelweek'])

        #build model
        XandTarget = XandTarget.dropna()
        model      = LinearRegression().fit(X=XandTarget[covariates2predictCases] , y= XandTarget[target])
        beta       = float(model.coef_[0])
        intercept  = model.intercept_

        # make probabilistic predictions
        mostRecentDataOnXandTarget = XandTarget.iloc[-1]

        # Linear regression assumes a normal distribution
        def computeSigma(XandTarget,target):
            predictions = model.predict(XandTarget[covariates2predictCases])
            target = XandTarget[target].values

            return sum((target - predictions)**2) / len(predictions)-1

        forecastData = {'numnewcases_leftbin':[],'numnewcases_rightbin':[],'weekahead':[],'prob':[]}
        
        nSamples = 5*10**4
        variance = computeSigma(XandTarget,target)
        std = np.sqrt(variance)
        
        doh1behind = np.array( [XandTarget.iloc[-1]['dohweb__numnewpos']] * nSamples )
        for weekahead in np.arange(1,4+1,1):
            mean    = intercept + beta*doh1behind # i am not accounting for uncertainty in the intercept and beta. Should make overconfident preds
            samples = np.random.normal(loc=mean,scale=std,size=nSamples)
        
            stepsize=5
            for numOfNewCases in np.arange(0,5*10**2,stepsize):
                probability = float( np.mean(samples <= numOfNewCases+stepsize) - np.mean(samples<=numOfNewCases) )

                forecastData['numnewcases_leftbin'].append(numOfNewCases)
                forecastData['numnewcases_rightbin'].append(numOfNewCases+stepsize)
                forecastData['prob'].append(probability)
                forecastData['weekahead'].append(weekahead)
            doh1behind = samples
        forecastData = pd.DataFrame(forecastData)

        # i restricted the range of values to forecast between 0 and 1000.
        def normalize(x):
            x['prob'] = x.prob/x.prob.sum()
            return x
        forecastData = forecastData.groupby(['weekahead']).apply(normalize).reset_index()
        forecastData['modelname'] = 'theScienceDogs'

        return forecastData

if __name__ == "__main__":

    # test packaged model

    # read data
    data = pd.read_csv('../../data/cases/PATrainingDataCases.csv')

    # subset to specific training week and use only epidemic weeks in 2020
    specificTrainingWeek = data[(data.trainingweek==202020) & (data.epiweek>201952)]

    # Startup model
    forecastingModel = sciencedogsmodel( specificTrainingWeek )

    # Forecast for all regions
    forecastsForAllRegions = forecastingModel.makeForecasts()

    # Store forecasts in forecasts folder
    forecastsForAllRegions.to_csv("../../forecasts/theScienceDogs/currentForecast/theScienceDogsForecast.csv")
