#mcandrew

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd

from delphi_epidata import Epidata
from epiweeks import Week, Year

from downloadHelper.downloadHelper import timestamp, listPACounties

def todayYMD():
    from datetime import datetime
    time = datetime.now()
    return int("{:04d}{:02d}{:02d}".format(time.year,time.month,time.day))

def fromToday2EpiWeek():
    thisweek = Week.thisweek()
    yr,wek = thisweek.year,thisweek.week
    return int("{:04d}{:02d}".format(yr,wek))

class DS(object):
    def __init__(self,varss,datasource,signal):
        self.varss=varss
        self.datasource = datasource
        self.signal     = signal 
        
        self.data = {}
        for var in varss:
            self.data[var]=[]

    def appendData(self,d):
        """ Take an existing dataset (x) and append a new data set (d) to it.
        """
        for var in self.varss:
            self.data[var].append(d[var])

    def convert2pandasDF(self):
        self.data = pd.DataFrame(self.data)
        self.data = self.data.sort_values(['geo_value','time_value'])
        return self
        
    def exportDF(self):
        ts = timestamp()
        self.data.to_csv("./covidcast_{:s}_{:s}_{:s}.csv".format(self.datasource,self.signal,ts),index=False)

    def has_data(self):
        return 1 if len(self.data)>0 else 0

def fromDataSource2Signal():
    d = {'fb-survey' :['raw_cli', 'raw_ili', 'raw_wcli', 'raw_wili']
     ,'ght'          :['raw_cli','smoothed_cli']
     ,'doctor-visits':['smoothed_cli']
     ,'google-survey':['raw_cli','smoothed_cli']
     ,'quidel'       :['smoothed_pct_negative','smoothed_tests_per_device']}
    return d

if __name__ == "__main__":

    todaysEW = fromToday2EpiWeek()
    todayYMD = todayYMD()
    
    variables = ['geo_value','time_value','value','stderr','sample_size']

    fromDataSource2Signal = fromDataSource2Signal()
    fips2name = listPACounties()
    
    for datasource in ['fb-survey','ght','doctor-visits','google-survey','quidel']:
        for signal in fromDataSource2Signal[datasource]:
            
            dataSet = DS(variables,datasource,signal)
            for county in fips2name:
                sys.stdout.write('\r{:s}--{:s}--{:06d}\r'.format(datasource,signal,county))
                sys.stdout.flush()
                
                dataFromAPI = Epidata.covidcast(datasource,signal,'day','county',Epidata.range(20200101,todayYMD),county)
                if dataFromAPI["message"] == "no results":
                    continue
                
                if dataFromAPI['message'] == "success":
                    for data in dataFromAPI['epidata']:
                        dataSet.appendData(data)
            if dataSet.has_data():
                dataSet.convert2pandasDF().exportDF()
