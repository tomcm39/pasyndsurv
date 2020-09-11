#Damon Luk
#script that merges together the USDA data and our data with confirmed cases

import pandas as pd

education = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/data/USDA/Education(PA-only).xlsx")
populationEstimate = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/data/USDA/PopulationEstimate(PA-only).xlsx")
povertyEstimate = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/data/USDA/PovertyEstimate(PA-only).xlsx")
unemployment = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/data/USDA/Unemployment(PA-only).xlsx")

confirmedCases = pd.read_csv("/Users/damonluk924/Desktop/pasyndsurv/data/cases/PATrainingDataCases.csv")

educationAndConfirmed = pd.merge(education, confirmedCases, left_on = 'FIPS Code', right_on = 'fips')
educationAndConfirmed.to_excel("Education_Confirmed_Merged.xlsx")

populationEstimateAndConfirmed = pd.merge(populationEstimate, confirmedCases, left_on = 'FIPStxt', right_on = 'fips')
populationEstimateAndConfirmed.to_excel("PopulationEstimate_Confirmed_Merged.xlsx")

povertyAndConfirmed = pd.merge(povertyEstimate, confirmedCases, left_on = 'FIPStxt', right_on = 'fips')
povertyAndConfirmed.to_excel("Poverty_Confirmed_Merged.xlsx")

unemploymentAndConfirmed = pd.merge(unemployment, confirmedCases, left_on = 'FIPStxt', right_on = 'fips')
unemploymentAndConfirmed.to_excel("Unemployment_Confirmed_Merged.xlsx")

