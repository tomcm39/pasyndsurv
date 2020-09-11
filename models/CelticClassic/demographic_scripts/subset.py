#Damon Luk
#a script that subsets the USDA data to PA - only counties


import pandas as pd 

education = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/data/USDA/Education.xls", skiprows = 4)
populationEstimate = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/data/USDA/PopulationEstimates-2.xls", skiprows = 2)
povertyEstimate = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/data/USDA/PovertyEstimates.xls", skiprows = 4)
unemployment = pd.read_excel("/Users/damonluk924/Desktop/pasyndsurv/data/USDA/Unemployment-2.xls", skiprows = 7)


paEducationData = education[education.State == "PA"]
#print (paEducationData)

paPopulationData = populationEstimate[populationEstimate.State == "PA"]
#print (paPopulationData)

paPovertyEstimate = povertyEstimate[povertyEstimate.Stabr == "PA"]
#print (paPovertyEstimate)

paUnemployment = unemployment[unemployment.Stabr == "PA"]
#print (paUnemployment)

paEducationData.to_excel('Education(PA-only).xlsx')
paPopulationData.to_excel('PopulationEstimate(PA-only).xlsx')
paPovertyEstimate.to_excel('PovertyEstimate(PA-only).xlsx')
paUnemployment.to_excel('Unemployment(PA-only).xlsx')
