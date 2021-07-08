#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SIR(object):
    def __init__(self,S0,I0,R0,beta,gamma):
       
        self.N  = S0+I0+R0
        self.S0 = S0
        self.C0 = 0
        self.I0 = I0
        self.R0 = R0
        
        self.beta  = beta
        self.gamma = gamma
   
    def generateMeanEpidemic(self,timesteps=10,sigmas=[1]):
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
        y     = odeint(odem,t=times,y0=[self.S0,self.I0,self.R0,0], args=((self.beta, self.gamma),) )
        yobs  = np.random.lognormal( np.log(y[:,-1]) ,sigmas)
        
        self.epidemicData = yobs
        self.epidemicMeanData = y
        return y

    def inferenceCases(self,S0,I0,R0,observedCumConfirmedCases):
        import pymc3 as pm
        
        import theano
        import theano.tensor as tt
        from theano import printing

        from pymc3.ode import DifferentialEquation
        from scipy.integrate import odeint
        import arviz as az

        N = S0+I0+R0

        def odem(y,t,p):
            # y= [S,I,R,c]
            # p= [beta,gamma,delta]

            dS = -1.*y[0]*y[1]*p[0]
            dI = y[0]*y[1]*p[0] - ( p[1]*y[1] )  
            dR = p[1]*y[1]

            dc = y[0]*y[1]*p[0]

            return [dS,dI,dR,dc]

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
            beta  = pm.Uniform('beta'  ,0.0,10.0)
            gamma = pm.Uniform('gamma' ,0.0,10.0)
            
            sigmas = pm.Gamma('sigmas', 0.5,0.5, shape=1)

            sir_curves = sir_model(y0=[S0,I0,R0,0], theta=[beta,gamma])
            obs        = pm.Normal("obs", observed = dta , mu = sir_curves[:,-1], sd=sigmas)
            
        with model:
            MAP = pm.find_MAP()
            self.MAP = MAP

        self.beta = MAP['beta']
        self.gamma = MAP['gamma']

    def makeForecastForOneRegion(self,regiondata ):
        import scipy

        numOfWeeks = len(regiondata)
        
        newcases  = regiondata[["dohweb__numnewpos"]].values
        cuumcases = np.cumsum(newcases)
        
        self.inferenceCases(S0=self.S0,I0=self.I0,R0=0.,observedCumConfirmedCases = cuumcases)
        
        parameterMAPS = self.MAP 
    
        betas  = parameterMAPS['beta']
        gamma = parameterMAPS['gamma']

        isigma = parameterMAPS['sigmas'][0]

        self.beta = betas
        self.gamma = gamma
        
        SIRCmeans = self.generateMeanEpidemic( numOfWeeks+4, isigma )

        IpredictedMeansPcts = SIRCmeans[-4:,1]
        IpredictedMeansNums = IpredictedMeansPcts*self.N

        forecastData = {'numnewcases_leftbin':[],'numnewcases_rightbin':[],'weekahead':[],'prob':[]}
        for weekahead in np.arange(0,4,1):
            
            print(IpredictedMeansPcts[weekahead])
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
        self.forecastData = forecastData
            
if __name__ == "__main__":

    bin = np.random.binomial
    
    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[(allData.fips == 42095) & (allData.trainingweek == mostrecentweek)]
    singleTW = allData[(allData.trainingweek == mostrecentweek)]

    # # until a case appears
    firstcase = 0
    for (idx,dta) in singleCounty.loc[:,["modelweek","dohweb__numnewpos"]].iterrows():
        if dta.dohweb__numnewpos==0 or np.isnan(dta.dohweb__numnewpos):
            continue
        firstcase = dta.modelweek
        break

    timeshift = int(firstcase - singleCounty.iloc[0]['modelweek'])
    
    #singleCounty =singleCounty[singleCounty.modelweek>=firstcase]
    
    # Looks like a few NANS show up in the data. I'm not sure why yet, but those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0

    S0 = singleCounty.iloc[0]['census']- 1
    I0 = 1.
    R0 = 0
    N =  S0+1
 
    singleCounty = singleCounty.replace(np.nan, 0.)
    
    epidemic = SIR(S0/N,I0/N,R0/N,1.,1.)
 
    epidemic.makeForecastForOneRegion(singleCounty)
