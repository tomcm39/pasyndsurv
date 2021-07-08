
# **SIR Model Description**

In this project, our team has worked on creating an SIR model. The SIR model is effective as it tries to determine the rates at which the population moves from the Susceptible(S) category to the Infected(I) and from the Infected(I) to the Recovered(R).  These rates are referred to as Beta and Gamma. Beta is the expected amount of people an infected individual will infect per day. Gamma is the proportion of infected recovering per day. This model creates probabilistic forecasts for the counties of Pennsylvania 1,2,3, and 4 weeks ahead. This model is using data for COVID-19 but it remains applicable to other epidemics i.e influenza-like illnesses, the flu, etc. 

# **Running**

With Python3 installed, in the CelticClassic model directory, run

```
$ python3 model_v0.3.py
```
# **Methods Used**

* Construct the variables needed for this model
    ```
    class SIR(object):
        def __init__(self,S0,I0,R0,beta,gamma):
            self.S0 = S0
            self.C0 = 0
            self.I0 = I0
            self.R0 = R0
    ```
* N is the total population (we are assuming that the whole population is in either S, I, or R)
    ```
            self.N = S0+I0+R0
    ```
* beta is the rate at which a Susceptible becomes Infected (contact rate)
    ```
            self.beta = beta
    ```
* gamma is the rate an infected becomes Recovered
    ```
            self.gamma = gamma
    ``` 
*   define each class and how it is calculated using differential equations. We are now using the binomial distribution (bin) .
* X~Bin(N,P) : Out of n possible trials, select one of the probability P then sum up all the X’s. For now, we are taking the mean of the binomial distribution for each class which will return the equation simply as it is.
    ```
   def S(self,Stm1,Itm1):
        return Stm1-bin(Stm1, self.beta*Itm1/self.N)
    def Smean(self,Stm1,Itm1):
        return Stm1-Stm1*self.beta*Itm1/self.N
    ```

* auxiliary variable to track the cumulative number of infections
    ```
    def C(self,Stm1,Itm1):
    return bin( Stm1, Itm1*self.beta/self.N )
    ```
        
* Here we basically put the values for all the categories (S,I,R) in one data frame in order to generate the epidemic
    ```
    def generateEpidemic(self,timesteps=10):
    .
    .
    .
    self.epidemicData = pd.DataFrame(pop)
    ```

* Here we do the same thing as generateEpidemic but for the mean values
    ```
    def generateMeanEpidemic(self,timesteps=10):
    .
    .
    .
    self.epidemicMeanData = pd.DataFrame(pop)
    ```
* The inference method allows us to create a trace map of the values which could sample out the variables gamma and beta 
    ```
    def inference(self,S0,I0,R0,observedNewConfirmedCases):
    .
    .
    .
      with model:
            MAP = pm.find_MAP()
            self.MAP = MAP
    ```
* The main method is used to call out the methods then create the plot of the model
* First we create an epidemic with the real-time data and plot 
    ```
    epidemic = SIR(S0,I0,R0,beta,gamma)
    epidemic.generateEpidemic(50)
    
    O = epidemic.epidemicData
    plt.plot(O);plt.show()
    ```
* Then we call the inference method to create 10^3 samples to determine our variables using sampling 
    ```
    epidemic.inference(S0=10**3,I0=1,R0=0,observedNewConfirmedCases = O[['C']])
    
    parameterMAPS = epidemic.MAP
    
    betas  = parameterMAPS['beta']
    gammas = parameterMAPS['gamma']
    ```
* Last we generate the mean epidemic based on the sampled variables and plot
    ```
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


# **Usage**

After runnning the program, the model will start off by sampling the beta and gamma values(this can take up to 10 minutes). Then a graph of the model with the number of cases on the y-axis and the epidemic weeks on the x-axis will appear.

**Data Sources Used**
* Centers for Disease Control (CDC) Fluview
* Johns Hopkins University's Center For Systems Science and Engineering COVID-19 Data repository
* The Delphi group at Carnegie Mellon University's COVIDcast

1. **Number of cases** 
* This is to get the information for the most recent week in a single county
    ```
    allData = pd.read_csv('../../data/cases/PATrainingDataCases.csv')
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[ (allData.fips == 42095) & (allData.trainingweek==mostrecentweek)  ]
    singleCounty = singleCounty.replace(np.nan , 0.)
    
    ```
* To get our initial variables, "S" "I" and "R", we started our model at the first week of confirmed cases
   ```
   startWeek = singleCounty[singleCounty.modelweek == 2617]
   S0 = float(startWeek [ ["census"] ].values)
   I0 = float(startWeek [ ["covidtracker__numnewpos"] ].values)
   R0 =  0.

   ```
* Set initial beta and gamma values to start
    ```
    beta = 1.
    gamma = 0.5

    ```

2. **Inference**
*This is used to figure out beta and gamma for our model
    ```
    

# Contact
Damon Luk: chl221@lehigh.edu
Martin Magazzolo: mlm321@lehigh.edu

