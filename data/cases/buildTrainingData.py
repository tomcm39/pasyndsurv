#mcandrew

import sys
import pickle
import numpy as np
import pandas as pd

def importData():
    from glob import glob
    pacasesdata = sorted(glob("./PAcasesData*"))[-1]
    return pd.read_csv("{:s}".format(pacasesdata))

if __name__ == "__main__":

    paCasesData = importData()
    epiweeks    = sorted(paCasesData.epiweek.unique())

    # turn back time
    trainingData_df   = pd.DataFrame()
    for currentweek in epiweeks:
        sys.stdout.write('\rTraining week = {:06d}\r'.format(currentweek))
        sys.stdout.flush()
        
        dataAsOfCurrentWeek = paCasesData[paCasesData.epiweek<currentweek]
        dataAsOfCurrentWeek['trainingweek'] = currentweek

        trainingData_df = trainingData_df.append(dataAsOfCurrentWeek)
    trainingData_dict = {trainingweek:data for (trainingweek,data) in trainingData_df.groupby(['trainingweek'])}

    trainingData_df.to_csv("./PATrainingDataCases.csv",index=False)
    pickle.dump(trainingData_dict, open("PATrainingDataCases.pkl","wb"))
