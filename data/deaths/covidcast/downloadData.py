#mcandrew

import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from delphi_epidata import Epidata
from epiweeks import Week, Year

from downloadHelper.cases import timestamp

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

def listPACounties():
    fips2name = {42001:"Adams",42003:"Allegheny",42005:"Armstrong",42007:"Beaver",42009:"Bedford",42011:"Berks",42013:"Blair",42015:"Bradford",
                 42017:"Bucks",42019:"Butler",42021:"Cambria",42023:"Cameron",42025:"Carbon",42027:"Centre",42029:"Chester",42031:"Clarion",
                 42033:"Clearfield",42035:"Clinton",42037:"Columbia",42039:"Crawford",42041:"Cumberland",42043:"Dauphin",42045:"Delaware",
                 42047:"Elk",42049:"Erie",42051:"Fayette",42053:"Forest",42055:"Franklin",42057:"Fulton",42059:"Greene",42061:"Huntingdon",
                 42063:"Indiana",42065:"Jefferson",42067:"Juniata",42069:"Lackawanna",42071:"Lancaster",42073:"Lawrence",42075:"Lebanon"	,
                 42077:"Lehigh",42079:"Luzerne",42081:"Lycoming",42083:"McKean",42085:"Mercer",42087:"Mifflin",42089:"Monroe",42091:"Montgomery",
                 42093:"Montour",42095:"Northampton",42097:"Northumberland",42099:"Perry",42101:"Philadelphia",42103:"Pike",42105:"Potter",
                 42107:"Schuylkill",42109:"Snyder",42111:"Somerset",42113:"Sullivan",42115:"Susquehanna",42117:"Tioga",42119:"Union",
                 42121:"Venango",42123:"Warren",42125:"Washington",42127:"Wayne",42129:"Westmoreland",42131:"Wyoming",42133:"York"}
    return fips2name

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
