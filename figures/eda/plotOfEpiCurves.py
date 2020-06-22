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

def fixXticks(ax):
    xticklabels = [x.get_text() for x in ax.get_xticklabels()]

    ax.set_xticks(ax.get_xticks()[4::3])
    xticklabels = xticklabels[4::3]
    
    xticklabels[0] = xticklabels[0][:4]+"W"+ xticklabels[0][4:]

    for i,xtick in enumerate(xticklabels):
        if i==0:
            pass
        elif i==1:
            xticklabels[i] = ""
        else:
            xticklabels[i]=xtick[4:]
    ax.set_xticklabels(xticklabels)
 
if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleTW = allData[(allData.trainingweek == mostrecentweek)]

    sumOfCases = singleTW.groupby(['epiweek']).apply( lambda x: pd.Series({'sumcases':x.dohweb__numnewpos.sum()}) )
    
    fips = [42125,42005,42017,42133,42041,42035]

    colors = grabColors(10)
    
    fig = plt.figure(constrained_layout=True)
    gs  = fig.add_gridspec(3, 3, width_ratios=[1.75,1,1])

    fh = fg(fig)

    ax = fig.add_subplot(gs[:,0])
    sns.barplot( x='epiweek',y='sumcases',data=sumOfCases.reset_index(), ax=ax, color=colors[0]  )

    ax.text(0.01,1.,s="Commonwealth of PA"
            ,ha='left',va='top',transform=ax.transAxes
            ,weight='bold',color=colors[0],fontsize=10
    )

    p = ph(ax)
    p.setTicks()
    p.xlabel("Epidemic week")
    p.ylabel("Number of new cases")

    fixXticks(ax)
    ax.set_yticks(ax.get_yticks()[1:])
   
    n=1
    for c in np.arange(1,2+1):
        for r in np.arange(0,3):
            ax = fig.add_subplot(gs[r,c])
            p = ph(ax)

            sns.barplot( x='epiweek',y='dohweb__numnewpos',data=singleTW[singleTW.fips==fips[n-1] ] , ax=ax, color=colors[n]  )
            
            ax.text(0.01,1.05,s="FIP={:d}".format(int(fips[n-1]))
                    ,ha='left',va='top',transform=ax.transAxes
                    ,weight='bold',color=colors[n],fontsize=10
            )
            n+=1

            p.setTicks()

            p.xlabel("")
            fixXticks(ax)
           
            p.ylabel("")
            ax.set_yticks(ax.get_yticks()[1:])

    fh.setsize(183,183/2.)
    #fh.tl()

    plt.savefig("epiCurveForPAandFips.pdf")
    plt.close()
