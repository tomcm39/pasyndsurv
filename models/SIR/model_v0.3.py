#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SIR(object):
    def __init__(self,S0,I0,R0,beta,gamma):
        self.S0 = S0
        self.C0 = 0
        self.I0 = I0
        self.R0 = R0
        self.N = S0+I0+R0
        
        self.beta = beta
        self.gamma = gamma

    def C(self,Stm1,Itm1): # auxiliary variable to track the cumulative number of infections
        return bin( Stm1, Itm1*self.beta/self.N )
 
    def S(self,Stm1,Ct):
        return Stm1-Ct
       
    def I(self,Itm1,Ct):
        return Itm1 + Ct - bin(Itm1,self.gamma)
    
    def R(self,Rtm1,Itm1):
        return Rtm1 + bin(Itm1,self.gamma)

    def Cmean(self,Stm1,Itm1): # auxiliary variable to track the cumulative number of infections
        return Stm1*Itm1*self.beta/self.N
 
    def Smean(self,Stm1,Ct):
        return Stm1-Ct
      
    def Imean(self,Itm1,Ct):
        return Itm1 + Ct - Itm1*self.gamma
   
    def Rmean(self,Rtm1,Itm1):
        return Rtm1 + Itm1*self.gamma

    def generateEpidemic(self,timesteps=10):
        pop = { "S":[self.S0], "I":[self.I0],"R":[self.R0],"C":[0] }

        for t in range(timesteps):
            Stm1,Itm1,Rtm1,Ctm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1], pop["C"][-1]
        
            Ct = self.C(Stm1,Itm1)
            
            St = self.S(Stm1,Ct)
            It = self.I(Itm1,Ct)
            Rt = self.R(Rtm1,Itm1)

            pop['S'].append( St )
            pop['C'].append( Ct )
            pop['I'].append( It )
            pop['R'].append( Rt )
            
        self.epidemicData = pd.DataFrame(pop)

    def generateMeanEpidemic(self,timesteps=10):
        pop = { "S":[self.S0], "I":[self.I0],"R":[self.R0], "C":[0]}

        for t in range(timesteps):
            Stm1,Itm1,Rtm1,Ctm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1], pop["C"][-1]
        
            Ct = self.Cmean(Stm1,Itm1)
            
            St = self.Smean(Stm1,Ct)
            It = self.Imean(Itm1,Ct)
            Rt = self.Rmean(Rtm1,Itm1)

            pop['S'].append( St )
            pop['C'].append( Ct )
            pop['I'].append( It )
            pop['R'].append( Rt )
        self.epidemicMeanData = pd.DataFrame(pop)

    def inference(self,S0,I0,R0,observedNewConfirmedCases):
        import pymc3 as pm
        
        import theano
        import theano.tensor as tt

        def Cmean(Stm1,Itm1,beta,N): # auxiliary variable to track the cumulative number of infections
            return Stm1*Itm1*beta/N
        
        def Smean(Stm1,Ct):
            return Stm1-Ct
        
        def Imean(Itm1,Ct,gamma):
            return Itm1 + Ct - Itm1*gamma

        def Rmean(Rtm1,Itm1,gamma):
            return Rtm1 + Itm1*gamma

        def proposeEpidemic(S0,I0,R0,beta,gamma,timesteps=10):
            pop = { "S":[S0], "I":[I0],"R":[R0],"C":[0]}
            N = S0+I0+R0
            
            for t in range(timesteps-1):
                Stm1,Itm1,Rtm1,Ctm1 = pop["S"][-1], pop["I"][-1], pop["R"][-1], pop["C"][-1]
        
                Ct = Cmean(Stm1,Itm1,beta,N)
                
                St = Smean(Stm1,Ct)
                It = Imean(Itm1,Ct,gamma)
                Rt = Rmean(Rtm1,Itm1,gamma)

                pop['S'].append( St )
                pop['C'].append( Ct )
                pop['I'].append( It )
                pop['R'].append( Rt )
            return pop['S'], pop['C'], pop['I'], pop['R']
        
        timesteps    = len(observedNewConfirmedCases)

        with pm.Model() as model:
            beta  = pm.Gamma('beta'  ,1.0,1.0)
            gamma = pm.Gamma('gamma' ,1.0,1.0)
            
            sigma = pm.Gamma('sigma', 0.5,0.5 )

            popS,popC,popI,popR = proposeEpidemic( S0,I0,R0, beta, gamma, timesteps )
            C =  tt.stack( popC, axis=1 )
            
            allObs        = np.array(observedNewConfirmedCases).reshape(-1,)
            obs = pm.Normal("obs", observed = allObs , mu = C ,sd=sigma)
            
        with model:
            MAP = pm.find_MAP()
            self.MAP = MAP
            
if __name__ == "__main__":

    bin = np.random.binomial
    
    S0 = 1000
    I0 = 1
    R0 = 0
    
    beta, gamma =1., 0.5

    epidemic = SIR(S0,I0,R0,beta,gamma)
    epidemic.generateEpidemic(50)
    
    O = epidemic.epidemicData
    plt.plot(O);plt.show()

    epidemic.inference(S0=10**3,I0=1,R0=0,observedNewConfirmedCases = O[['C']])
    
    parameterMAPS = epidemic.MAP
    
    betas  = parameterMAPS['beta']
    gammas = parameterMAPS['gamma']
    isigma = parameterMAPS['sigma']

    epidemicT = SIR(S0,I0,R0,beta = betas ,gamma = gammas )
    epidemicT.generateMeanEpidemic(50)
    
    fig,ax = plt.subplots()

    ax.plot(epidemicT.epidemicMeanData.C, color='red',label="Estimated")
    ax.scatter(x=O.index, y=O.C, s=30, alpha=0.5, color = 'blue',label="Observed")
  
    ax.set_xlabel("Epidemic Week")
    ax.set_ylabel("Number of new confirmed cases")

    ax.legend()
    plt.savefig('./example.pdf')
    plt.close()
