#mcandrew

import sys
sys.path.append('../../')

import numpy as np
import pandas as pd

from APIkeys import apiKeyForPEP

class pepData(object):
    def __init__(self,apikey,statename):
        self.key = apikey
        self.statename = statename
        self.apiRoot = "https://api.census.gov/data/2019/pep/population"
        self.addFIPSCode()

    def addFIPSCode(self):
        import us
        self.statecode = int(us.states.lookup(self.statename).fips)
        
    def grabCountyevelPEPdataForState(self):
        import requests
        data = {"get" : "COUNTY,POP,DENSITY,NAME,STATE"
                ,"for":"county:*"
                ,"in":"state:{:d}".format(self.statecode)
                ,"key":self.key
        }
        response = requests.get(self.apiRoot, params=data)

        headers = response.json()[0]
        data    = response.json()[1:]
        
        df = {x:[] for x in headers}
        for listt in data:
            for d,c in zip(listt,df):
                if c=="NAME":
                    d = d.split(',')[0].strip()
                df[c].append(d)
        df = pd.DataFrame(df)
        df['countyfips'] = [ "{:02d}{:03d}".format(self.statecode,int(x)) for x in df.COUNTY]
        df.drop(columns=['state'],inplace=True)
        self.data = df
        return self

    def export(self,filename):
        self.data.to_csv(filename,index=False)

if __name__ == "__main__":

    apiKey        = apiKeyForPEP()
    
    countyPopInfo = pepData(apiKey, 'PA')
    countyPopInfo.grabCountyevelPEPdataForState().export("./PApopdata.csv")
