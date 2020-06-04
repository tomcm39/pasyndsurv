#mcandrew

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd

from downloadHelper.downloadtools import timestamp
from downloadHelper.downloadtools import listPACounties

if __name__ == "__main__":

    fips2county = listPACounties()
    county2fips = {county:[fip] for (fip,county) in fips2county.items() }
    
    dohWebsiteData = pd.read_csv('https://raw.githubusercontent.com/jeremymack-LU/covid19/master/covid19_pa_counties.csv')

    def AddNumNewPos(d):
        d = d.sort_values('Date')
        d['numnewpos'] = d.Cases.diff()
        return d[['Date','numnewpos']]
    dohWebsiteData = dohWebsiteData.groupby(['County']).apply(AddNumNewPos).reset_index()
    dohWebsiteData = dohWebsiteData.replace(np.nan,0).drop(columns=['level_1'])
    
    countyAndFips = pd.DataFrame(county2fips).melt().rename(columns={'variable':'County','value':'fips'})
    
    dohWebsiteData = dohWebsiteData.merge(countyAndFips, on = ['County'])
    dohWebsiteData = dohWebsiteData.rename(columns={'Date':'date'})

    dohWebsiteData['region']='pa'
    
    dohWebsiteData.to_csv('./dohWebsite_{:s}.csv'.format(timestamp()),index=False)
