# mcandrew,kline,lin

#import sys
import numpy as np
import pandas as pd
#import matplotlib.pyplot as pltmn
#import seaborn as sns
import tkinter as tk
#import statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing as eSmooth
# X marks the import statement!

fips_library = [42039, 42079, 42077, 42083, 42089, 42027, 42041, 42043, 42045, 42051, 42073, 42081, 42091, 42093, 42095, 42047, 42049, 42075, 42007, 42055, 42057, 42071, 42085, 42059, 42061, 42063, 42065, 42067, 42069, 42087, 42097, 42099, 42101, 42103, 42105, 42107, 42109, 42111, 42113, 42115, 42117, 42119, 42121, 42123, 42125, 42127, 42129, 42131, 42133, 42019, 42021, 42023, 42025, 42029, 42031, 42033, 42035, 42037, 42053, 42001, 42003, 42005, 42009, 42011, 42013, 42015, 42017]
counties = ['Crawford', 'Luzerne', 'Lehigh', 'McKean', 'Monroe', 'Centre', 'Cumberland', 'Dauphin', 'Delaware', 'Fayette', 'Lawrence', 'Lycoming', 'Montgomery', 'Montour', 'Northampton', 'Elk', 'Erie', 'Lebanon', 'Beaver', 'Franklin', 'Fulton', 'Lancaster', 'Mercer', 'Greene', 'Huntingdon', 'Indiana', 'Jefferson', 'Juniata', 'Lackawanna', 'Mifflin', 'Northumberland', 'Perry', 'Philadelphia', 'Pike', 'Potter', 'Schuylkill', 'Snyder', 'Somerset', 'Sullivan', 'Susquehanna', 'Tioga', 'Union', 'Venango', 'Warren', 'Washington', 'Wayne', 'Westmoreland', 'Wyoming', 'York', 'Butler', 'Cambria', 'Cameron', 'Carbon', 'Chester', 'Clarion', 'Clearfield', 'Clinton', 'Columbia', 'Forest', 'Adams', 'Allegheny', 'Armstrong', 'Bedford', 'Berks', 'Blair', 'Bradford', 'Bucks']

r = tk.Tk()
r.title('Covid County Finder')
r.geometry('300x300')
v = tk.StringVar()
v.set("")
tbox = tk.Entry(r,width=10,textvariable=v)    #sets up start screen
s = v.get()
tbox.grid(column=1, row=2)
txt = tk.Label(r, text = "Please enter county below.")
txt.grid(column=0, row=0)


def onClick():
    entry = tbox.get()
    i = counties.index(entry)
    countfips = fips_library[i]
    getData(countfips, counties[i])

button = tk.Button(r, text='PLOT', width=5, bg = 'purple', fg = 'white', command=onClick)
button.grid(column=1, row=3)

allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")


def getData(countfips, county):

    # for now, lets subset our data to the most recent training week and
    # a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[ (allData.fips == countfips) & (allData.trainingweek==mostrecentweek)  ]

    # Looks like a few NANS show up in the data. I'm not sure why yet, but
    # those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0
    singleCounty = singleCounty.replace(np.nan,0.)

    # Documentation on how to use Holt-Winters for Python is here
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    # notice the import state above marked with an X

    trainingData = singleCounty[['modelweek','dohweb__numnewpos']].set_index('modelweek')
    covidCrushersModel = eSmooth(trainingData, trend='add')
    fittedCovidCrushersModel = covidCrushersModel.fit()

    # predictions for the next 4 weeks
    lastWeekOfData = trainingData.index.max()
    forecasts = fittedCovidCrushersModel.predict(start=len(trainingData),
                                                 end=len(trainingData)+4)
    # some issues will arise here when we are in December

    # NOW is the hard part.
    # lets look at a plot of the data and the forecast

    fig,ax = plt.subplots() # setup a plot environment
    ax.plot( trainingData.index, trainingData.dohweb__numnewpos, color='b', alpha=0.50, label = "DOH data" )
    ax.scatter( trainingData.index, trainingData.dohweb__numnewpos, s=30, color='b', alpha=0.50 )

    # build a list of forecasted epiweeks
    forecastedEpiweeks = [ lastWeekOfData+x for x in np.arange(0,4+1)]

    ax.plot(forecastedEpiweeks, forecasts, color='k', label = "prediction")
    # now I'll plot the forecasted weeks and the predictions

    # i should label my axes. No one likes bare axes, no one!
    ax.set_xlabel("Model week")
    ax.set_ylabel("Number of new COVID cases")
    title = str('New Cases in ' + county+' County')
    plt.title(title)

    # lets check out my legend too. Note the two "label" statements above
    ax.legend()

    # and then I'll take a look at what we did!
    plt.show()

    # TODOS:
    # Holt Winter's is a point prediction. How will you turn it into a
    # probabilistic forecast, a probability distribution over future values?
    # Take a look at FIPS = 42045. Anything we need to fix about this model?
    # So many more for Alex and Kenny to list
r.mainloop()
