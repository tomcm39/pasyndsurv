#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SIR(object):
    #construct the variables needed for this model
    def __init__(self,S0,I0,R0,beta,gamma):
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        #N is the total population (we are assuming that the whole population is in either S, I, or R)
        self.N = S0+I0+R0
        #beta is the rate at which a Susceptible becomes Infected (contact rate)
        self.beta = beta
        #gamma is the rate an infected becomes Recovered
        self.gamma = gamma


    #-----------------------------------------------------------------
        #define each class and how it is calculated using differential equations
        #we are now using the binomial distribution (bin) 
        #X~Bin(N,P) : Out of n possible trials, select one of the probability P then sum up all the X’s 
        #for now, we are taking the mean of the binomial distribution for each class 
    #-----------------------------------------------------------------

    def S(self,Stm1,Itm1):
        return Stm1-bin(Stm1, self.beta*Itm1/self.N)
    def Smean(self,Stm1,Itm1):
        return Stm1-Stm1*self.beta*Itm1/self.N
        
    def I(self,Stm1,Itm1):
        return Itm1 + bin( Stm1, Itm1*self.beta/self.N ) - bin(Itm1,self.gamma)

    def Imean(self,Stm1,Itm1):
        return Itm1 + Stm1*Itm1*self.beta/self.N - Itm1*self.gamma

    def R(self,Rtm1,Itm1):
        return Rtm1 + bin(Itm1,self.gamma)
    
    def Rmean(self,Rtm1,Itm1):
        return Rtm1 + Itm1*self.gamma

    #-----------------------------------------------------------------
    #Here we basically put the values for all the categories (S,I,R) in one data frame
    #-----------------------------------------------------------------
    def generateEpidemic(self,timesteps=10):
        pop = { "S":[self.S0], "I":[self.I0],"R":[self.R0]}

        for t in range(timesteps):
            Stm1,Itm1,Rtm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1]
        
            St = self.S(Stm1,Itm1)
            It = self.I(Stm1,Itm1)
            Rt = self.R(Rtm1,Itm1)

            pop['S'].append( St )
            pop['I'].append( It )
            pop['R'].append( Rt )
        self.epidemicData = pd.DataFrame(pop)

    def generateMeanEpidemic(self,timesteps=10):
        pop = { "S":[self.S0], "I":[self.I0],"R":[self.R0]}

        for t in range(timesteps):
            Stm1,Itm1,Rtm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1]
        
            St = self.Smean(Stm1,Itm1)
            It = self.Imean(Stm1,Itm1)
            Rt = self.Rmean(Rtm1,Itm1)

            pop['S'].append( St )
            pop['I'].append( It )
            pop['R'].append( Rt )
        self.epidemicMeanData = pd.DataFrame(pop)

    def inference(self,observedData):
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
        
        with pm.Model() as model:
            beta  = pm.Normal('beta'  ,mu=1,sigma=0.5)
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

    bin = np.random.binomial
    
    S0 = 1000
    I0 = 1
    R0 = 0
    
    beta=1.
    gamma=0.5

    epidemic = SIR(S0,I0,R0,beta,gamma)
    epidemic.generateEpidemic(50)
    
    O = epidemic.epidemicData
    
    epidemic.inference(O)
    #Sample out the probable betas and gammas 
    parameterSamples = epidemic.trace
    betas  = parameterSamples.get_values('beta')
    gammas = parameterSamples.get_values('gamma')
    
    epidemic = SIR(S0,I0,R0,beta = np.mean(betas) ,gamma = np.mean(gammas))
    epidemic.generateMeanEpidemic(50)
    #plot the data 
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

    #TODO:
    # You need to make this a probabilistic distribution, not just the mean prediction
