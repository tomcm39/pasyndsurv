#mcandrew

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd

from delphi_epidata import Epidata
from epiweeks import Week, Year

from downloadHelper.downloadtools import timestamp

class downloader(object):
    def __init__(self,state='',region=''):
        self.state=state
        self.region=region
        self.signals = ['release_date','region','epiweek','lag','num_patients','num_providers','wili','ili']
        self.addTodaysEpiWeek()
        self.grabDataFromEpicast()

    def addTodaysEpiWeek(self):
        thisweek = Week.thisweek()
        yr,wek = thisweek.year,thisweek.week
        self.todaysEW = int("{:04d}{:02d}".format(yr,wek))
        
    def grabDataFromEpicast(self):
        if self.region=='':
            self.fludata = Epidata.fluview(self.state, [Epidata.range(201840,self.todaysEW)])
        elif self.state=='':
            self.fludata = Epidata.fluview(self.region, [Epidata.range(201840,self.todaysEW)])
        else:
            self.fludata = Epidata.fluview(self.region+self.state, [Epidata.range(201840,self.todaysEW)])
            
        self.fludata_message = self.fludata['message']
        self.fludata_data    = self.fludata['epidata']

    def has_fludata(self):
        return True if self.fludata_message == "success" else False
       
    def downloadILIdata(self):
        def createDataSet():
            iliData = {}
            for signal in self.signals:
                iliData[signal]=[]
            self.iliData = iliData
        createDataSet()
        
        def appendData(d):
            for var in self.signals:
                self.iliData[var].append(d[var])

        if self.has_fludata():
            for data in self.fludata_data:
                appendData(data)
            self.iliData = pd.DataFrame(self.iliData).sort_values(['region','epiweek'])
        return self
            
    def export(self,filename):
        ts = timestamp()    
        self.iliData.to_csv("{:s}_{:s}.csv".format(filename,ts), index=False)
            
if __name__ == "__main__":

    cdcDownloader = downloader(['PA'])
    cdcDownloader.downloadILIdata().export("./ilidata_cdc")
