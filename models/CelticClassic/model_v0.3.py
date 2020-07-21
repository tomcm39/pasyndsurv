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
        self.C0 = 0
        self.I0 = I0
        self.R0 = R0
        self.N = S0+I0+R0
        
        self.beta = beta
        self.gamma = gamma

    def C(self,Stm1,Itm1): #auxiliary variable to track the cumulative number of infections
        return bin( Stm1, Itm1*self.beta/self.N)

    def S(self,Stm1,Ct): 
        return Stm1-Ct #susceptible people from last week - random sample from binomial distribution. distr based on people susceptible from last week and prob. they become infected
 
    def I(self,Itm1,Ct):
        return Itm1 + Ct - bin(Itm1,self.gamma) #people that are infected last week plus people that became infected this week (distr) minus people that recovered

    def R(self,Rtm1,Itm1):
        return Rtm1 + bin(Itm1,self.gamma) #people recovered last week plus new recoveries
    
    def Cmean(self,Stm1, Itm1): 
        return Stm1*Itm1*self.beta/self.N
    
    def Smean(self,Stm1,Ct):
        return Stm1-Ct #most likely to happen - average
     
    def Imean(self,Itm1,Ct):
        return Itm1 + Ct - Itm1*self.gamma 

    def Rmean(self,Rtm1,Itm1):
        return Rtm1 + Itm1*self.gamma

    def generateEpidemic(self,timesteps=10):
        pop = { "S":[self.S0], "I":[self.I0],"R":[self.R0], "C":[0]} #dictionary with initial values

        for t in range(timesteps): 
            Stm1,Itm1,Rtm1,Ctm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1], pop["C"][-1] #takes previous week's value and makes them stm,itm,rtm
        
            Ct = self.C(Stm1,Itm1)
            St = self.S(Stm1,Itm1) #calls above functions for present week
            It = self.I(Stm1,Itm1)
            Rt = self.R(Rtm1,Itm1)

            pop['C'].append (Ct) 
            pop['S'].append( St ) #adding them to the dictionary
            pop['I'].append( It )
            pop['R'].append( Rt )
        self.epidemicData = pd.DataFrame(pop) #takes dictionary and makes it into a tabular data, kind of like an array

    def generateMeanEpidemic(self,timesteps=10): #same thing as above but with the means
        pop = { "S":[self.S0], "I":[self.I0],"R":[self.R0], "C":[0]}

        for t in range(timesteps):
            Stm1,Itm1,Rtm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1], pop["C"][-1]
            
            Ct = self.Cmean(Stm1,Itm1)
            St = self.Smean(Stm1,Itm1)
            It = self.Imean(Stm1,Itm1)
            Rt = self.Rmean(Rtm1,Itm1)

            pop['C'].append( Ct )
            pop['S'].append( St )
            pop['I'].append( It )
            pop['R'].append( Rt )
        self.epidemicMeanData = pd.DataFrame(pop) #dataframe

    def inference(self,observedNewConfirmedCases): #figure out properties of beta, gamma of our model
        import pymc3 as pm
        import theano
        import theano.tensor as tt

        def Cmean(Stm1,Itm1,beta,N):
            return Stm1*Itm1*beta/N
        
        def Smean(Stm1,Ct):
            return Stm1-Ct
        
        def Imean(Itm1,Ct,gamma):
            return Itm1 + Ct - Itm1*gamma

        def Rmean(Rtm1,Itm1,gamma):
            return Rtm1 + Itm1*gamma

        def proposeEpidemic(S0,I0,R0,beta,gamma,timesteps=10):
            pop = { "S":[S0], "I":[I0],"R":[R0], "C":[0]}
            N = S0+I0+R0
            
            for t in range(timesteps-1):
                Stm1,Itm1,Rtm1,Ctm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1], pop["C"][-1]

                Ct = Cmean(Stm1,Itm1,beta,N)
                St = Smean(Stm1,Ct)
                It = Imean(Itm1,Ct,gamma)
                Rt = Rmean(Rtm1,Itm1,gamma)

                pop['C'].append( Ct )
                pop['S'].append( St )
                pop['I'].append( It )
                pop['R'].append( Rt )
            return pop['S'], pop['I'], pop['R'], pop['C']
        
        timesteps    = len(observedNewConfirmedCases)

        S0,I0,R0 = observedNewConfirmedCases.iloc[0]

        p = proposeEpidemic( S0,I0,R0, 1., 0.5, timesteps )
        self.p = p
        
        with pm.Model() as model: #uses pymc3 to make an object of a model class
            beta  = pm.Gamma('beta'  ,1.0,1.0) #calculate normal distribution of beta and gamma
            gamma = pm.Gamma('gamma' ,1.0,1.0)

            sigma = pm.Gamma('sigma', 0.5,0.5 ) 

            popS,popC,popI,popR = proposeEpidemic( S0,I0,R0, beta, gamma, timesteps )
            C = tt.stack( popC, axis=1)

            allObs        = np.array(observedNewConfirmedCases).reshape(-1,)
            
            obs = pm.Normal("obs",observed = allObs , mu = C ,sd=sigma)
            
        with model:
            MAP = pm.find_MAP()
            self.MAP = MAP

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

    print ("S0:" + S0)
    print ("I0" + I0)
    print ("R0" + R0)

    epidemic = SIR(S0,I0,R0,beta,gamma)
    epidemic.generateEpidemic(50)

    O = epidemic.epidemicData
    epidemic.inference(O)

    parameterMAPS = epidemic.MAP

    #parameterSamples = epidemic.trace
    betas  = parameterMAPS['beta']
    gammas = parameterMAPS['gamma']
    isigma,rsigma = parameterMAPS['sigma'][0]

    epidemic = SIR(S0,I0,R0,beta = betas, gamma = gammas)
    epidemic.generateMeanEpidemic(50)
    
    fig,axs = plt.subplots(1,3)

    ax=axs[0]
    ax.scatter(O.index,O.S,s=10,alpha=0.4)
    ax.scatter(O.index,O.I,s=10,alpha=0.4)
    ax.scatter(O.index,O.R,s=10,alpha=0.4)
    ax.scatter(O.index,O.C,s=30,alpha=0.5)
        
    ax.plot(epidemic.epidemicMeanData.S, label='Susc') 
    ax.plot(epidemic.epidemicMeanData.I, label='Infec')
    ax.plot(epidemic.epidemicMeanData.R, label='Recov')
    ax.plot(epidemic.epidemicMeanData.C, label='Estimated')
   
    ax.set_xlabel("Epidemic Week")
    ax.set_ylabel("Absolute number")

    ax.legend()
    
    ax=axs[1]
    ax.scatter(O.index,O.I,s=10,alpha=0.4)
    ax.plot(epidemic.epidemicMeanData.I, label='Infec')
    ax.fill_between(O.index, epidemic.epidemicMeanData.I - 1.96*isigma, epidemic.epidemicMeanData.I + 1.96*isigma, alpha=0.3)

    ax.set_xlabel("Epidemic Week")
    ax.set_ylabel("Number of Infected")

    ax=axs[2]
    ax.scatter(O.index,O.R,s=10,alpha=0.4)
    ax.plot(epidemic.epidemicMeanData.R, label = 'Recovered')
    ax.fill_between(O.index, epidemic.epidemicMeanData.R - 1.96*rsigma, epidemic.epidemicMeanData.R + 1.96*rsigma, alpha = 0.3)
    
    ax.set_xlabel("Epidemic Week")
    ax.set_ylabel("Number of recovered")

    plt.show()

