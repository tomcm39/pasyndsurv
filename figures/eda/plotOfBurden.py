#mcandrew

import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random

import matplotlib.gridspec as gridspec

from scipy.stats import multivariate_normal as mvn

sys.path.append('..')
from plotHelper.plothelp import grabColors
from plotHelper.plothelp import ph
from plotHelper.plothelp import fg

def fixXticks(ax):
    xticklabs = []
    for i,xtick in enumerate(ax.get_xticks()):
        txt = str(int(xtick))
        if i==0:
            txt = str(txt)
            xticklabs.append( txt[:4]+"W"+txt[4:] )
        else:
            xticklabs.append(str(txt)[4:])
    ax.set_xticklabels(xticklabs)


if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleTW = allData[(allData.trainingweek == mostrecentweek)]

    colors = grabColors(10)

    fig,axs = plt.subplots(1,2)
    
    fh = fg(fig)

    prevs,maxnew = [],[]

    ax = axs[0]
    for region,data in singleTW.groupby(["fips"]):
        cs = np.cumsum(data.dohweb__numnewpos)
        ax.plot(    data.epiweek, cs, color = colors[5], lw=2.1,alpha=0.4 )
        ax.scatter( data.epiweek, np.cumsum(data.dohweb__numnewpos), s=10, color = colors[5] )

        prevs.append(cs.iloc[-1])
        maxnew.append(max(data.dohweb__numnewpos.dropna()))
        
    p = ph(ax)
    p.setTicks()
    p.xlabel("Epidemic week")
    p.ylabel("Cumul. number of cases")

    ax.set_yticks([0,2500,5000,7500,10000,12500,15000,17500])
    ax.set_yticklabels(["0","2,500","5,000","7,500","10,000","12,500","15,000","17,500"])

    ax.set_xticks([202010,202015,202020,202024])
    
    fixXticks(ax)

    p.addletter("A.")

    ax = axs[1]
    p = ph(ax)

    x, y = np.meshgrid(np.linspace(min(prevs)  ,max(prevs)*0.75 ,200)
                       ,np.linspace(min(maxnew),max(maxnew)*0.75,200))
    xy = np.column_stack([x.flat, y.flat])

    # density values at the grid points
    meanPrev, meanMaxnew = np.mean(prevs), np.mean(maxnew)
    cov = np.cov( prevs, maxnew )
    Z = mvn.pdf(xy, [meanPrev,meanMaxnew] , cov).reshape(x.shape)

    # arbitrary contour levels
    ax.contourf(x, y, Z, cmap = "Blues", alpha=0.50 )
    ax.scatter( prevs,maxnew, s=10, color='black', edgecolors="white", lw=0.5 )

    ax.set_xlim(0,8000)
    ax.set_ylim(0,1200)

    p.setTicks()
    p.xlabel("Total number of cases by county")
    p.ylabel("Max number of new cases by county")

    ax.set_xticks([0,2000,4000,6000,8000])
    ax.set_xticklabels(["0","2,000","4,000","6,000","8,000"])

    ax.set_yticks([0,200,400,600,800,1000,1200])
    ax.set_yticklabels(["0","200","400","600","800","1,000","1,200"])
    
    p.addletter("B.")
    
    fh.setsize(183,183/2.)
    fh.tl()
    plt.savefig("burdenOnPA.pdf")

    plt.close()
    
