# klin

import sys
sys.path.append('../..')

import tabula
import pandas as pd
from downloadHelper.downloadtools import timestamp
from downloadHelper.downloadtools import listPACounties

if __name__ == "__main__":
    data_link = '''https://www.health.pa.gov/topics/Documents/Diseases%20and%20
Conditions/COVID-19%20County%20Data/County%20Case%20Counts_6-15-2020.pdf'''

    # formats the return file with timestamp of retrieval
    return_file = './dohCumulative_{:s}.csv'.format(timestamp())
    tabula.convert_into(data_link, return_file, pages="all")

    fips2county = listPACounties()
    #print(fips2county)
    county2fips = [fip for (fip, county) in fips2county.items()]

    doh_DF = pd.read_csv(return_file)

    def cal_tot_tests(data_frame):
        test_col = data_frame['Confirmed'] + data_frame['PersonsWithNegativePCR']
        data_frame['Est_Tot_Tests'] = test_col  # estimated total tests

    def add_fips(df):
        df.insert(1, 'Fips', county2fips)

    cal_tot_tests(doh_DF)
    add_fips(doh_DF)
    doh_DF.to_csv(return_file, index=False)
