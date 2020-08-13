#Damon Luk
#analyzing log scores 

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

scoresData = pd.read_csv("/Users/damonluk924/Desktop/pasyndsurv/scores/scores.csv") #read in the scores file
#print (scoresData.head(20))

scoresData = scoresData.replace(np.NINF, -10) #replaces all -inf with a negative 10
#print (soresData.head(50))

#average log score for each model
averageModel_Dict = {"Model Name":[], "Average for each model":[]}
for model, data in scoresData.groupby(['modelname']):
    average = data["score"].mean(skipna = True)
    #print ("Model: {:s}, Avg = {:.2f}".format(model,average))
    averageModel_Dict["Model Name"].append(model)
    averageModel_Dict["Average for each model"].append(average)

averageModel = pd.DataFrame(averageModel_Dict)
averageModel = averageModel[['Model Name', 'Average for each model']]


#average log score for each fip
averageFip_Dict = {"Fips":[], "Average for each fip":[]}
for fip, data in scoresData.groupby(['fips']):
    average = data["score"].mean(skipna = True)
    #print ("FIP: {:5.0f}, Avg = {:.2f}".format(fip,average))
    averageFip_Dict["Fips"].append(fip)
    averageFip_Dict["Average for each fip"].append(average)

averageFip = pd.DataFrame(averageFip_Dict)
averageFip = averageFip[['Fips', 'Average for each fip']]
    
 
#average log score for weeks ahead
averageWeek_Dict = {"Weeks Ahead":[], "Average for each week":[]}
i = 1
for week, data in scoresData.groupby(['weekahead']):
    average = data["score"].mean(skipna = True)
    #print ("week " + str(i) + ": " + str(average))
    averageWeek_Dict["Weeks Ahead"].append(i)
    averageWeek_Dict["Average for each week"].append(average)
    i += 1

averageWeek = pd.DataFrame(averageWeek_Dict)
averageWeek = averageWeek [['Weeks Ahead', 'Average for each week']]

#averageModel.to_excel("Average Score for each Model.xlsx")
#averageFip.to_excel("Average Score for each Fip.xlsx")
#averageWeek.to_excel("Average Score for each Week.xlsx")

scoresData = scoresData[['modelname', 'fips', 'weekahead', 'score']]

sns.set()

sns.boxplot(x = 'modelname', y = 'score', data = scoresData)
plt.show()


sns.boxplot(x = 'weekahead', y = 'score', data = scoresData)
plt.show()

sns.boxplot(x = 'fips', y = 'score', data = scoresData)
plt.show()







