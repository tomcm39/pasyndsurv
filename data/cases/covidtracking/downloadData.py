#mcandrew

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd

from downloadHelper.downloadtools import timestamp

if __name__ == "__main__":
 
    covidtracking = pd.read_csv("https://covidtracking.com/api/v1/states/pa/daily.csv") 
    covidtracking.to_csv("./covidtracking_{:s}.csv".format(timestamp()))
