#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SIRD(object):
    def __init__(self,S0,I0,R0,D0,beta,gamma,delta):
       
        self.N = S0+I0+R0+D0
        self.S0 = S0/self.N
        self.C0 = 0
        self.I0 = I0/self.N
        self.R0 = R0/self.N
        self.D0 = D0/self.N
        
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta # rate of death
   
    def generateMeanEpidemic(self,timesteps=10,sigmas=[1,1]):
        from scipy.integrate import odeint

        def odem(y,t,p):
            # y= [S,I,R,D,c,d,r]
            # p= [beta,gamma,delta]

            dS = -1.*y[0]*y[1]*p[0]
            dI = y[0]*y[1]*p[0] - ( p[1]*y[1] + p[2]*y[1] )  
            dR = p[1]*y[1]
            dD = p[2]*y[1]

            dc = y[0]*y[1]*p[0]

            return [dS,dI,dR,dD,dc]
        
        times = np.arange(0,timesteps)
        y     = odeint(odem,t=times,y0=[self.S0,self.I0,self.R0,self.D0,0], args=((self.beta, self.gamma, self.delta),) )
        yobs  = np.random.lognormal( np.log(y[:,[3,4]]) ,sigmas)
        
        self.epidemicData = yobs
        self.epidemicMeanData = y

    def inferenceCasesAndDeaths(self,S0,I0,R0,D0,observedNewConfirmedCases,observedNewDeaths):
        import pymc3 as pm
        
        import theano
        import theano.tensor as tt
        from theano import printing

        from pymc3.ode import DifferentialEquation
        from scipy.integrate import odeint
        import arviz as az

        def odem(y,t,p):
            # y= [S,I,R,D,c,d,r]
            # p= [beta,gamma,delta]

            dS = -1.*y[0]*y[1]*p[0]
            dI = y[0]*y[1]*p[0] - ( p[1]*y[1] + p[2]*y[1] )  
            dR = p[1]*y[1]
            dD = p[2]*y[1]

            dc = y[0]*y[1]*p[0]

            return [dS,dI,dR,dD,dc]

        dta          =  np.hstack( (observedNewDeaths.reshape(-1,1),observedNewConfirmedCases.reshape(-1,1)) )
        
        timesteps = len(dta)
        times     = np.arange(0,timesteps)

        sir_model = DifferentialEquation(
            func  = odem,
            times = times,
            n_states = 5,
            n_theta  = 3,
            t0       = 0,
        )
        with pm.Model() as model:
            beta  = pm.Uniform('beta'  ,0.0,10.0)
            gamma = pm.Uniform('gamma' ,0.0,10.0)
            delta = pm.Uniform('delta' ,0.0,10.0)
            
            sigmas = pm.Gamma('sigmas', 0.5,0.5, shape=2 )

            sir_curves = sir_model(y0=[S0,I0,R0,D0,0], theta=[beta,gamma,delta])
            obs        = pm.Normal("obs", observed = dta , mu = sir_curves[:, [3,4] ], sd=sigmas)
            
        with model:
            MAP = pm.find_MAP()
            self.MAP = MAP
            
if __name__ == "__main__":

    bin = np.random.binomial
    
    S0 = 0.99
    I0 = 0.01
    R0 = 0
    D0 = 0
    
    beta, gamma, delta =1., 0.5, 0.2

    epidemic = SIRD(S0,I0,R0,D0,beta,gamma,delta)
    epidemic.generateMeanEpidemic(50, [0.1,0.1])
    
    O = epidemic.epidemicData

    epidemic.inferenceCasesAndDeaths(S0=0.99,I0=0.01,R0=0,D0=0
                                     ,observedNewConfirmedCases = O[:,1]
                                     ,observedNewDeaths = O[:,0]
    )
    
    parameterMAPS = epidemic.MAP
    
    betas  = parameterMAPS['beta']
    gammas = parameterMAPS['gamma']
    deltas = parameterMAPS['delta']

    isigma = parameterMAPS['sigmas'][0]
    dsigma = parameterMAPS['sigmas'][1]
   
    epidemic = SIRD(S0,I0,R0,D0,beta = betas ,gamma = gammas,delta = deltas )
    epidemic.generateMeanEpidemic(50)
    
    fig,axs = plt.subplots(1,2)

    ax = axs[0]
    ax.plot(epidemic.epidemicMeanData[:,-1], color='red',label="Estimated")
    ax.scatter(x=np.arange(0,len(O)), y=O[:,-1], s=30, alpha=0.5, color = 'blue',label="Observed")
  
    ax.set_xlabel("Epidemic Week")
    ax.set_ylabel("Number of cumulative confirmed cases")

    ax = axs[1]
    ax.plot(epidemic.epidemicMeanData[:,-2], color='red',label="Estimated")
    ax.scatter(x=np.arange(0,len(O)), y=O[:,0], s=30, alpha=0.5, color = 'blue',label="Observed")
  
    ax.set_xlabel("Epidemic Week")
    ax.set_ylabel("Number of deaths")

    ax.legend()

    fig.set_tight_layout(True)
    
    plt.savefig('./example.pdf')
    plt.close()
    
