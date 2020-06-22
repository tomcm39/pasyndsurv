#mcandrew

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random

import matplotlib.gridspec as gridspec

sys.path.append('..')
from plotHelper.plothelp import grabColors
from plotHelper.plothelp import ph
from plotHelper.plothelp import fg

def ccdf(x):
    x,L=sorted(x),len(x)
    return x, 1. - np.arange(0.,L)/L


if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleTW = allData[(allData.trainingweek == mostrecentweek)]

    sumOfCases = singleTW.groupby(['epiweek']).apply( lambda x: pd.Series({'sumcases':x.dohweb__numnewpos.sum()}) )

    singleTW = singleTW.replace(np.nan,0.) 
    stds    = singleTW.groupby(["fips"]).apply(  lambda x: pd.Series({'std':np.std(x.dohweb__numnewpos)} ) )
    
    levels  = singleTW.groupby(["fips"]).apply(  lambda x: pd.Series({'mean':np.median(x.dohweb__numnewpos)} ) )


    colors = grabColors(4)
    
    fig,axs = plt.subplots(1,2)

    ax=axs[0]
    x,px = ccdf([float(x) for x in stds.values])
    ax.plot( x,px, lw=2., color=colors[0],alpha=0.50 )
    ax.scatter(x[::3],px[::3],s=10,color=colors[0] )

    i=10
    scatterx,scattery = x[::3],px[::3]
    ax.plot([10**2,scatterx[i]],[0.7,scattery[i]],lw=1,color=colors[0])
    ax.text( 10**2, 0.7, s="Std. dev.", ha="left",va="center",weight="bold",fontsize=10,color=colors[0])

    
    _25,_75 = np.percentile( stds.values, [25,75] )
    xs,ys = zip(*[ (xval,yval) for (xval,yval) in zip(x,px) if _25<=xval<=_75])
    ax.fill_between( x = xs  , y1=[0]*len(xs), y2=ys, color = colors[0],alpha=0.3 )

    print(_25)
    print(_75)
    
    x,px = ccdf([float(x) for x in levels.values])
    x=np.array(x)

    # quick fix for log-scaling
    idx = x>0
    x = x[idx]
    px= px[idx]
    
    x=[0.3]+list(x)
    px = [1.0]+list(px)
    
    ax.plot( x,px, lw=2., color=colors[1],alpha=0.50 )
    ax.scatter( x[::3],px[::3],s=10,color=colors[1] )

    i=10
    scatterx,scattery = x[::3],px[::3]
    ax.plot([10**2,scatterx[i]],[0.4,scattery[i]],lw=1,color=colors[1])
    ax.text( 10**2, 0.4, s="Avg.", ha="left",va="center",weight="bold",fontsize=10,color=colors[1])

    _25,_75 = np.percentile( levels.values, [25,75] )
    print(_25)
    print(_75)
 
    xs,ys = zip(*[ (xval,yval) for (xval,yval) in zip(x,px) if _25<=xval<=_75])
    ax.fill_between( x = xs  , y1=[0]*len(xs), y2=ys, color = colors[1],alpha=0.3 )
    
    ax.set_xscale("log")

    def minandmax(x):
        mn = min(x.dohweb__numnewpos)
        return pd.DataFrame( {"mx":[max(x.dohweb__numnewpos)], "mn": [max(0,mn)]})
    ranges  = singleTW.groupby(["fips"]).apply(minandmax)
    ranges = ranges.sort_values("mx")

    p = ph(ax)
    p.setTicks()
    p.xlabel("Number of new cases")
    p.ylabel("P(X>x)")

    
    ax=axs[1]

    for i,(fip, dta) in enumerate(ranges.iterrows()):
        ax.plot([i]*2, [dta.mn+1, dta.mx], color=colors[3], lw=1.)
        ax.scatter([i]*2, [dta.mn+1, dta.mx], s=10, color=colors[3])

    p = ph(ax)
    ax.set_yscale("log")

    ax.set_xticklabels([""])
    
    p.setTicks()
    p.xlabel("Counties in PA")
    p.ylabel("Range of the number of new cases")


    fh = fg(fig)
    fh.setsize(183,183/2.)
    fh.tl()
    
    plt.savefig("variabilityInTheNumberOfNewCases.pdf")
    plt.close()
