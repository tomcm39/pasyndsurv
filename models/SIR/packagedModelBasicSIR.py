# mcandrew,kline,lin

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as eSmooth

class basicSIR(object):
    # Copy this section 
    def __init__(self):
        pass

    def addTrainingData(self,trainingdata):
        self.trainingdata = trainingdata

    # Copy this function
    def modeldesc(self):
        print("The Basic SIR model")
    
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
    def generateMeanEpidemic(self,timesteps=10,sigmas=[1],beta=1,gamma=1,S0=0.99,I0=0.01,R=0.):
        from scipy.integrate import odeint

        def odem(y,t,p):
            # y= [S,I,R,c]
            # p= [beta,gamma,delta]

            dS = -1.*y[0]*y[1]*p[0]
            dI = y[0]*y[1]*p[0] - ( p[1]*y[1] )  
            dR = p[1]*y[1]

            dc = y[0]*y[1]*p[0]

            return [dS,dI,dR,dc]
        
        times = np.arange(0,timesteps)
        y     = odeint(odem,t=times,y0=[S0,I0,R0,0], args=((beta, gamma),) )
        yobs  = np.random.lognormal( np.log(y[:,-1]) ,sigmas)
        
        self.epidemicData = yobs
        self.epidemicMeanData = y
        return y
    
    def inferenceCases(self,S0,I0,R0,observedCumConfirmedCases):
        import pymc3 as pm
        
        import theano
        import theano.tensor as tt # <-
        from theano import printing

        from pymc3.ode import DifferentialEquation
        from scipy.integrate import odeint
        import arviz as az

        N = S0+I0+R0

        def odem(y,t,p):
            # y= [S,I,R,c]
            # p= [beta,gamma,delta]

            dS = -1.*y[0]*y[1]*p[0]
            dI = y[0]*y[1]*p[0] - ( p[1]*y[1] )  # infected
            dR = p[1]*y[1]

            dc = y[0]*y[1]*p[0] # cumulative cases

            return [dS,dI,dR,dc] # last state is cumulative cases

        dta          =  observedCumConfirmedCases.reshape(-1,)
        #dta          = O[:,-1].reshape(-1,)

        timesteps = len(dta)
        times     = np.arange(0,timesteps)

        sir_model = DifferentialEquation(
            func  = odem,
            times = times,
            n_states = 4,
            n_theta  = 2,
            t0       = 0,
        )
        with pm.Model() as model:
            beta  = pm.Normal('beta'  ,2.0,1)
            gamma = pm.Normal('gamma' ,2.0,1)

            tt.printing.Print('beta')(beta)
            tt.printing.Print('gamma')(gamma)
            
            sigmas = pm.Gamma('sigmas', 0.5,0.5, shape=1)

            sir_curves = sir_model(y0=[S0,I0,R0,0], theta=[beta,gamma]) # maybe??
            obs        = pm.Normal("obs", observed = dta , mu = sir_curves[:,-1], sd=sigmas) # maybe?
            
        with model:
            MAP = pm.find_MAP()
            self.MAP = MAP

        self.beta = MAP['beta']
        self.gamma = MAP['gamma']
 
    def makeForecastForOneRegion(self,regiondata):
        regiondata = regiondata.replace(np.nan,0.)

        import scipy

        numOfWeeks = len(regiondata)
        
        newcases  = regiondata[["dohweb__numnewpos"]].values

        N  = int(regiondata.iloc[0].census)

        cuumcases = np.cumsum(newcases)/N # this has to be a fraction? Tariq thinks so too. Hes not sure.
        
        S0 = (N-1)/N
        I0 = 1./N
        R0 = 0.
        
        self.inferenceCases(S0=S0,I0=I0,R0=0.,observedCumConfirmedCases = cuumcases)

        parameterMAPS = self.MAP 
    
        betas  = parameterMAPS['beta']
        gamma  = parameterMAPS['gamma']
        isigma = parameterMAPS['sigmas'][0]

        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N  = N 

        # changes - here
        self.beta  = betas
        self.gamma = gamma

        print(betas)
        print(gamma)
        
        SIRCmeans = self.generateMeanEpidemic( numOfWeeks+4, isigma )# here? 

        IpredictedMeansPcts = SIRCmeans[-4:,1]
        IpredictedMeansNums = IpredictedMeansPcts*self.N

        forecastData = {'numnewcases_leftbin':[],'numnewcases_rightbin':[],'weekahead':[],'prob':[]}
        for weekahead in np.arange(0,4,1):
            
            dist = scipy.stats.norm(loc= self.N*IpredictedMeansPcts[weekahead] , scale = self.N*isigma )
            
            stepsize=5
            for numOfNewCases in np.arange(0,5*10**2,stepsize):
                probability = float(dist.cdf(numOfNewCases+stepsize) - dist.cdf(numOfNewCases) )

                forecastData['numnewcases_leftbin'].append(numOfNewCases)
                forecastData['numnewcases_rightbin'].append(numOfNewCases+stepsize)
                forecastData['prob'].append(probability)
                forecastData['weekahead'].append(weekahead)
        forecastData = pd.DataFrame(forecastData)

        def normalize(x):
            sprob = np.sum(x['prob'])
            x['prob'] = x['prob']/sprob
            return x
        forecastData = forecastData.groupby(['weekahead']).apply( normalize )
        forecastData['modelname'] = 'basicSIR'
        return forecastData

if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()

    mostRecentWeekData = allData[allData.trainingweek == mostrecentweek]

    md = basicSIR()
    md.addTrainingData(mostRecentWeekData)
    md.makeForecasts()
    
   
