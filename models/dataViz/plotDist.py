#mcandrew

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

class parseWrangler(object):
    """
    input: forecastFile - A string that points to a CSV file of forecasts. This file MUST contain the following variables: numnewcases_leftbin, numnewcases_rightbin,  fips, TW, and modelname. This file can contain any additional variables needed for analysis. 
           FIPS - On of the 67 FIPS codes corresponding to a county in PA
           TW   - The training week---the week in realtime the data is forecasted from---to plot data for
           model - The name of the model to plot forecasts from  
           toPDF - 1 = produce a pdf and 0 = show a plot after the code is finished.
    """
    
    def __init__(self,forecastFile="",FIPS="",TW="",model="",toPDF=0):
        self.forecastFile = forecastFile

        try:
            self.FIPS = int(FIPS)
        except NotAnInteger("Needs to be a number that can be converted to an integer"):
            pass
        
        try:
            self.TW = int(TW)
        except NotAnInteger("Needs to be a number that can be converted to an integer"):
            pass

        self.model = model # need to add code that outputs all model names as a string in fullDens.csv

        self.toPDF = toPDF

    def NotAnInteger(Exception):
        pass

class plotPmf(object):
    """ input: A "parseWrangler" Object
        output: a visualization of a pmf forecats for 1-4 weeks ahead

    The goal of this code is to interpret a input from a user---that specifies a forecastFile, FIPS, traiing week and model---
    and output a visualization of the probability mass function for 1,2,3, and 4 week ahead new confirmed cases.
    
    Users can specify a 1 in the parseWrangler object to output a pdf of the forecasts in the SAME LOCATION as where the script is run.
    """
    
    def __init__(self,parseWrang):
        self.forecastFile = parseWrang.forecastFile
        self.FIPS         = parseWrang.FIPS
        self.TW           = parseWrang.TW
        self.model        = parseWrang.model
        self.toPDF        = parseWrang.toPDF

        self.loadForecast()
        self.subset()

    def loadForecast(self):
        try:
            self.forecast = pd.read_csv("{:s}".format(self.forecastFile))
        except:
            raise notCSV
        
    class notCSV(Exception):
        pass

    def subset(self):
        d = self.forecast

        if self.model=="":
            self.forecast = d[ (d.fips==self.FIPS) & (d.trainingweek==self.TW) ]
        else:
            self.forecast = d[ (d.fips==self.FIPS) & (d.forecastTW==self.TW) & (d.modelname==self.model) ]

    def whenProbDropsBelow1Pct(self,bins,probs):
        for (b,p) in zip(bins,probs):
            if p<0.01:
                return 1.5*b
        return b
            
    def continuousRVPlot(self):
        plt.style.use("fivethirtyeight")
        fig,ax = plt.subplots()

        forecast = self.forecast

        maxbelow1 = -1
        for wkahead in np.arange(1,4+1):
            singleWeekAhead = forecast[forecast.weekahead==wkahead]
            ax.plot(singleWeekAhead.numnewcases_leftbin, singleWeekAhead.prob, lw=2., alpha=0.60, label="{:d} Week ahead".format(wkahead))
            ax.scatter(singleWeekAhead.numnewcases_leftbin[::10], singleWeekAhead.prob[::10], s=10., alpha=0.60)

            below1 = self.whenProbDropsBelow1Pct(singleWeekAhead.numnewcases_leftbin, singleWeekAhead.prob)
            if below1>maxbelow1:
                maxbelow1=below1

        ax.set_xlabel("Values"       ,fontsize=12.)
        ax.set_ylabel("Probabilities",fontsize=12.)

        ax.set_xlim(0,maxbelow1)

        ax.tick_params(direction="in",size=2.)

        ax.legend(frameon=False)

        def subsetlabels(lbl,info,fs=10,x=0.50,y=0.50): 
            ax.text(x,y,lbl.format(info),fontsize=fs,ha="left",va="center",transform=ax.transAxes)

        subsetlabels("Model={:s}"           ,self.model ,x=0.60,y=0.60)
        subsetlabels("FIPS = {:d}"          ,self.FIPS  ,x=0.60,y=0.50)
        subsetlabels("TW = {:d}"            ,self.TW    ,x=0.60,y=0.40)
        subsetlabels("forecastFile = {:s}"  ,self.forecastFile,fs=8,x=0.60,y=0.30)
        
        fig.set_tight_layout(True)

        if self.toPDF:
            from datetime import datetime
            runtime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
            plt.savefig("{:s}{:d}{:d}__runtime{:s}.pdf".format(self.model,self.FIPS,self.TW,runtime))
        else:
            plt.show()
            
if __name__ == "__main__":
    #--------------------------------
    # Inputs
    #--------------------------------
    # forecastFile:
    # FIPS        :
    # TW          :
    # model       :
    # toPDF       :
    #--------------------------------
    # Outputs
    #--------------------------------
    # Draws a plot. If toPDF is selected then this script will generate a pdf in the folder the code was run.
    

    
    parser = argparse.ArgumentParser(description='Plot the PMF (PDF) of a forecast.')
    
    parser.add_argument('--forecastFile', type=str,help='the filename of a CSV file containing a forecast.')
    parser.add_argument('--FIPS'        , type=str,help='Forecast for this FIPS.')
    parser.add_argument('--TW'          , type=str,help='Plot for this training week.')
    parser.add_argument('--model'       , type=str,help='The model to plot, the forecast file contains multiple models.')

    parser.add_argument('--toPDF'       , type=int,help='Produces a PDF in the same location the script is run')

    args = parser.parse_args()


    # Parser Wrangler
    pw = parseWrangler(args.forecastFile, args.FIPS, args.TW, args.model, args.toPDF)

    dataViz = plotPmf(pw)
    dataViz.continuousRVPlot()
    
    
