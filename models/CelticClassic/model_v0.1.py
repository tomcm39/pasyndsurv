#mcandrew, Damon Luk, Martin Magazzolo 

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



    # ---------------------------------------------------------------------------
    # TRAIN MODEL Below
    # Your model should take in data from Single County and output forecasts (predictions) for at least 1 week ahead. 
    # ---------------------------------------------------------------------------


class SIR(object):
    def __init__(self,S0,I0,R0,beta,gamma): #constructor
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N = S0+I0+R0
        
        self.beta = beta
        self.gamma = gamma

    def S(self,Stm1,Itm1): 
        return Stm1-bin(Stm1, self.beta*Itm1/self.N) #susceptible people from last week - random sample from binomial distribution. distr based on people susceptible from last week and prob. they become infected

    def Smean(self,Stm1,Itm1):
        return Stm1-Stm1*self.beta*Itm1/self.N #most likely to happen - average
        
    def I(self,Stm1,Itm1):
        return Itm1 + bin( Stm1, Itm1*self.beta/self.N ) - bin(Itm1,self.gamma) #people that are infected last week plus people that became infected this week (distr) minus people that recovered

    def Imean(self,Stm1,Itm1):
        return Itm1 + Stm1*Itm1*self.beta/self.N - Itm1*self.gamma 

    def R(self,Rtm1,Itm1):
        return Rtm1 + bin(Itm1,self.gamma) #people recovered last week plus new recoveries
    
    def Rmean(self,Rtm1,Itm1):
        return Rtm1 + Itm1*self.gamma

    def generateEpidemic(self,timesteps=10):
        pop = { "S":[self.S0], "I":[self.I0],"R":[self.R0]} #dictionary with initial values

        for t in range(timesteps): 
            Stm1,Itm1,Rtm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1] #takes previous week's value and makes them stm,itm,rtm
        
            St = self.S(Stm1,Itm1) #calls above functions for present week
            It = self.I(Stm1,Itm1)
            Rt = self.R(Rtm1,Itm1)

            pop['S'].append( St ) #adding them to the dictionary
            pop['I'].append( It )
            pop['R'].append( Rt )
        self.epidemicData = pd.DataFrame(pop) #takes dictionary and makes it into a tabular data, kind of like an array

    def generateMeanEpidemic(self,timesteps=10): #same thing as above but with the means
        pop = { "S":[self.S0], "I":[self.I0],"R":[self.R0]}

        for t in range(timesteps):
            Stm1,Itm1,Rtm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1]
        
            St = self.Smean(Stm1,Itm1)
            It = self.Imean(Stm1,Itm1)
            Rt = self.Rmean(Rtm1,Itm1)

            pop['S'].append( St )
            pop['I'].append( It )
            pop['R'].append( Rt )
        self.epidemicMeanData = pd.DataFrame(pop) #dataframe

    def inference(self,observedData): #figure out properties of beta, gamma of our model
        import pymc3 as pm
        import theano.tensor as tt

        def Smean(Stm1,Itm1,beta,N):
            return Stm1-Stm1*beta*Itm1/N
        
        def Imean(Stm1,Itm1,beta,gamma,N):
            return Itm1 + Stm1*Itm1*beta/N - Itm1*gamma

        def Rmean(Rtm1,Itm1,gamma):
            return Rtm1 + Itm1*gamma

        def proposeEpidemic(S0,I0,R,beta,gamma,timesteps=10):
            pop = { "S":[S0], "I":[I0],"R":[R0]}
            N = S0+I0+R0
            
            for t in range(timesteps-1):
                Stm1,Itm1,Rtm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1]
        
                St = Smean(Stm1,Itm1 ,beta,N)
                It = Imean(Stm1,Itm1,beta,gamma,N)
                Rt = Rmean(Rtm1,Itm1,gamma)

                pop['S'].append( St )
                pop['I'].append( It )
                pop['R'].append( Rt )
            return pop['S'], pop['I'], pop['R']
        
        timesteps    = len(observedData)

        S0,I0,R0 = observedData.iloc[0]

        p = proposeEpidemic( S0,I0,R0, 1., 0.5, timesteps )
        self.p = p
        
        with pm.Model() as model: #uses pymc3 to make an object of a model class
            beta  = pm.Normal('beta'  ,mu=1,sigma=0.5) #calculate normal distribution of beta and gamma
            gamma = pm.Normal('gamma' ,mu=1,sigma=0.5)

            sigma = pm.Gamma('sigma', 0.5,0.5 ) 

            meanS, meanI, meanR = proposeEpidemic( S0,I0,R0, beta, gamma, timesteps )

            allObs        = list(observedData.I.values) + list(observedData.R.values)
            meanIandMeanR = meanI + meanR
            
            obs = pm.MvNormal('obs',observed = allObs , mu = meanIandMeanR ,cov=sigma*np.eye(timesteps*2))
            
        with model:
            trace = pm.sample(2*10**3)
            self.trace = trace

if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("/Users/damonluk924/Desktop/pasyndsurv/data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[ (allData.fips == 42095) & (allData.trainingweek==mostrecentweek)  ]

    # Looks like a few NANS show up in the data. I'm not sure why yet, but those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0
    singleCounty = singleCounty.replace(np.nan , 0.)
    
    #for values in singleCounty["covidtracker__numnewpos"]:
        #if np.isnan(values) == True:
            #singleCounty.replace(values, 0)

    bin = np.random.binomial
    startWeek = singleCounty[singleCounty.modelweek == 2617]
    S0 = float(startWeek [ ["census"] ].values) #population from census minus people that are infected
    I0 = float(startWeek [ ["covidtracker__numnewpos"] ].values) #number of positive cases, but from which source?
    #I0 = 2.
    R0 =  0. #census population - deaths - total cases 

    beta = 1.
    gamma = 0.5

    print (S0)
    print (I0)
    print (R0)

    epidemic = SIR(S0,I0,R0,beta,gamma)
    epidemic.generateEpidemic(50)
    epidemic.generateMeanEpidemic(50)

    O = epidemic.epidemicData
    
    #epidemic.inference(O)

    #parameterSamples = epidemic.trace
    #betas  = parameterSamples.get_values('beta')
    #gammas = parameterSamples.get_values('gamma')
    
    #epidemic = SIR(S0,I0,R0,beta = np.mean(betas) ,gamma = np.mean(gammas))
    #epidemic.generateMeanEpidemic(50)
    fig,ax = plt.subplots()

    ax.scatter(O.index,O.S,s=10,alpha=0.4)
    ax.scatter(O.index,O.I,s=10,alpha=0.4)
    ax.scatter(O.index,O.R,s=10,alpha=0.4)
        
    ax.plot(epidemic.epidemicMeanData.S, label='Susc') 
    ax.plot(epidemic.epidemicMeanData.I, label='Infec')
    ax.plot(epidemic.epidemicMeanData.R, label='Recov')
   
    ax.set_xlabel("Epidemic Week")
    ax.set_ylabel("Absolute number")

    ax.legend()
    
    plt.show()


