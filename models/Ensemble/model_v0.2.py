# luk, donnchie (equal contrib), mcandrew

import sys
import numpy as np
import pandas as pd 

def equalEnsemble(probs):
    """ 
        Equally Weighted ensemble model equation
        Inputs: probs is a list of forecats probabilities 
        Outputs: the EW ensemble prob
    """

    numModels = len(probs)  # number of models
    probs = np.array(probs) # convert to numpy array

    if np.any( np.nan(probs) ):
        print( "A forecast contained a NAN" )
    else:
        ensembleProb = sumProb / numModels
        return ensembleProb

if __name__ == "__main__":

    #goal(week of 9/7): make an equally weighted ensemble for one week ahead, one fip for all the case bins with the most recent training week

    forecastData    = pd.read_csv("../../scores/fulldens.csv") #pull sample forecast data from git repo
    maxTrainingWeek = forecastData.forecastTW.max() # wont need this (looping through al 

    #subset sample forecast data
    forecastDataEz = forecastData[ (forecastData.weekahead == 1) & (forecastData.fips == 42001) & (forecastData.forecastTW == maxTrainingWeek)] #subsets the data to what we need for our goal

    # we need to fill up this dictionary with forecasts
    singleEWForecast = { numnewcases_leftbin   :[]
                         ,numnewcases_rightbin :[]
                         ,numnewcases_midbin   :[]
                         ,fips                 :[]
                         ,weekahead            :[]
                         ,forecastTW           :[]
                         ,prob                 :[]
    }

    # lets start with iterated for loops:
    # 1. forecastTW
    # 2. Fips
    # 3. weekahead
    # in the innermost loop you'll include your below code

    average, midpoint =[], []
    #this for loop gets us the mid point of all the bins
    for (left_bin,right_bin), data in forecastDataEz.groupby(['numnewcases_leftbin','numnewcases_rightbin']):
        numnewcases_mid = (left_bin + right_bin) / 2.0
        midpoint.append(numnewcases_mid)
        averageProb_OneBin = equalEnsemble(data.prob.sum(), 4)
        #print(averageProb_OneBin)
        average.append(averageProb_OneBin)


        # and here is where you'll include data into the dictionary above.
        # Like this
        singleEWForecast["numnewcases_leftbin"].append(left_bin)
    
    # ~~~~~~~~ can delete~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    midbin = pd.DataFrame(midpoint, columns = ['numnewcases_mid'])
    averageProb = pd.DataFrame(average, columns = ['averageProb'])
    ensembleDataEz = pd.merge(midbin, averageProb, left_index = True, right_index = True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # we wont need to merge if we build the above singleEWForecast dict friom scratch
    # we can just use this code
    singleEWForecast = pd.DataFrame(singleEWForecast)
    


    # ----- code for plotting ---------------------------------------------------------------------------
    # let's move this code to a separate python script called visualCheck.py insidet our same folder

    sns.style.use("fivethieryright")

    fig,ax = plt.subplots()

    d = ensembleDataEz
    ax.plot( d.numnewcases_mid, d.prob, "b-", linewidth=2 )
    
    ax.set_xlabel("Number of new cases", fontsize=10)
    ax.set_xlabel("PMF", fontsize=10)

    ax.tick_params(size=2.,direction="in")

    ax.text(0.99,0.99,"Equally Weighted Ensemble"
            ,ha='right',va='top',transform=ax.transAxes)

    plt.savefig("visualCheckProbDist.pdf")
    plt.close()
    

    # sns.lineplot(x = 'numnewcases_mid', y = 'averageProb', data = ensembleDataEz).set_title("Equally Weighted Ensemble for 1 Week ahead in FIP 42001")
    # plt.xlim(0 , 80)
    # plt.show()

    #creates data frame with same format as other models
    #temporarily takes first 100 since we are only look at fips 42001 with weekahead 1
    # ensembleForecast = pd.DataFrame({
    #     'numnewcases_leftbin' : forecastData.numnewcases_leftbin[0:100],
    #     'numnewcases_rightbin' : forecastData.numnewcases_rightbin[0:100],
    #     'weekahead' : forecastData.weekahead[0:100],
    #     'prob' : average,
    #     'fips' : forecastData.fips[0:100]})
    #print(ensembleForecast)
