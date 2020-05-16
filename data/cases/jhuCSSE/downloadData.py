#mcandrew

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from downloadHelper.cases import timestamp

def collectRawdata():
    jhucsse_USconfirmedcasesUrl = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    return pd.read_csv(jhucsse_USconfirmedcasesUrl)

def subset2PA(d):
    return d[d.Province_State=='Pennsylvania']

if __name__ == "__main__":

    usData = collectRawdata()
    paData = subset2PA(usData)

    dates  = [x for x in paData.columns if "/" in x]
    paData = paData.melt(id_vars = ['FIPS','Province_State'],value_vars=dates)
    paData['FIPS'] = paData.FIPS.astype('int')
    paData = paData.rename(columns={'variable':'date','value':'count'})

    paData['date'] = pd.to_datetime(paData.date) 
    paData = paData.sort_values(['FIPS','date'])
    paData.to_csv('./jhuCSSEdata_{:s}.csv'.format(timestamp()),index=False)
