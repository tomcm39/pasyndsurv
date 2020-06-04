#mcandrew

import numpy as np
import pandas as pd

class trainModelOverTime(object):
    """ Input:  
           - trainingData: TrainingData from this repository, either cases or deaths, that your model will take as input.
           - model: An object that has a single input, trainingData and contains a method called "makeforecasts"
       Output:
           - When the method trainModelFromFirst2MostRecentEpiWeek is run a dataset called forecastDataFromTraining is attached to this object. This dataset contains 1-4 week forecasts for every trainingweek.
    """
    def __init__(self,trainingData,model):
        self.trainingData=trainingData
        self.model = model
    
    def trainModelFromFirst2MostRecentEpiWeek(self):
        def subset2SpecificTrainingWeek(tw):
            return self.trainingData[self.trainingData.trainingweek==tw]

        forecastData = pd.DataFrame()
        trainingweeks = sorted(self.trainingData.trainingweek.unique())
        for tw in trainingweeks:
            trainingData = subset2SpecificTrainingWeek(tw)

            model = self.model(trainingData)
            forecasts_1to4weeks = model.makeforecasts()
            
            forecastData = forecastData.append(forecastData)
        self.forecastDataFromTraining = forecastData

if __name__ == "__main__":
    pass

    # d = pd.read_csv("../cases/PATrainingDataCases.csv")
    # m = trainModelOverTime(d,d)
    # m.trainModelFromFirst2MostRecentEpiWeek()
    
    
