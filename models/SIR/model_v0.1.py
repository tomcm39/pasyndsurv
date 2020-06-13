#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SIR(object):
    def __init__(self,S0,I0,R0,beta,gamma):
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N = S0+I0+R0

        self.beta = beta
        self.gamma = gamma

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

if __name__ == "__main__":

    bin = np.random.binomial
    
    S0 = 1000
    I0 = 1
    R0 = 0
    
    beta=1.
    gamma=0.5

    epidemic = SIR( S0,I0,R0,beta,gamma)
    epidemic.generateEpidemic(50)
    epidemic.generateMeanEpidemic(50)

    fig,ax = plt.subplots()

    ax.plot(epidemic.epidemicMeanData)

    plt.show()
 


    
    
    # epiData = epidemic.epidemicData

    # def propose(gamma,beta):
    #     while True:
    #         gamma += np.random.normal(0,1) # prior
    #         if 0 < gamma < 1:
    #             break

    #     while True:
    #         beta += np.random.normal(0,1) # prior
    #         if beta >0:
    #             break
    #     return gamma,beta
    
    # gamma,beta = propose(1,1)
    # epidemic = SIR(S0,I0,R0,beta,gamma)
    # epidemic.generateEpidemic(50)
    # epidemic.generateMeanEpidemic(50)

    
    # def evaluateLikelihoodOfParams(epiData):
    #     prevLogLike = -np.inf
    #     logLikelihood = 0.
    #     for time,(s,i,r) in epiData.iloc[1:].iterrows(): # skip the first row
    #         stm1,itm1,rtm1 = epiData.iloc[time-1]    

    #         prob = scipy.stats.binom( itm1,gamma ).pmf( r-rtm1 )
    #         logProb = np.log(prob)
    #         logLikelihood+=logProb
    #     return logLikelihood

    # def hop(prevLogLike,logLikelihood):
    #     if logLikelihood >= prevLogLike:
    #         currentBeta  = beta
    #         currentGamma = gamma
    #     else:
    #         if np.log(np.random.uniform()) < logLikelihood-prevLogLike:
    #             currentBeta  = beta
    #             currentGamma = gamma
    #         else:
    #             currentGamma, currentBeta = propose(gamma,beta)

    
