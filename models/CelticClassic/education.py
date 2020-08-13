#Damon Luk
#Script that graphs number of new cases vs education data in the most recent week

import pandas as pd 
import matplotlib.pyplot as plt 
import math

educationAndConfirmed = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/models/CelticClassic/demo_confirmed/Education_Confirmed_Merged.xlsx")
maxTrainingWeek = povertyAndConfirmed.trainingweek.max()
maxEpiWeek = povertyAndConfirmed.epiweek.max()

subsetEducation = educationAndConfirmed [(educationAndConfirmed.trainingweek == maxTrainingWeek) & (educationAndConfirmed.epiweek == maxEpiweek)]
subsetEducation = subsetEducation[ []]
