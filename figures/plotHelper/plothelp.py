#mcandrew

def grabColors(N):
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')

    colors = []
    fig,ax = plt.subplots()
    for x in range(N):
        ax.plot([0],[0])
    for n,x in enumerate(ax.get_children()):
        if n>N-1:
            break
        colors.append(x.get_color())
    plt.close()
    return colors

class ph(object):
    
    def __init__(self,ax):
        self.ax=ax
        
    def setTicks(self):
        self.ax.tick_params(which='both',labelsize=8.)
    
    def ylabel(self,y):
        self.ax.set_ylabel(y,fontsize=10,color="#3d3d3d")
        
    def xlabel(self,x):
        self.ax.set_xlabel(x,fontsize=10,color="#3d3d3d")

    def addletter(self,letter,tl=1):
        if tl:
            self.ax.text(0.0,1.0,"{:s}".format(letter),ha="left",va="center",weight="bold",fontsize=12,color="#2d2d2d",transform=self.ax.transAxes)
        else:
            self.ax.text(1.0,1.0,"{:s}".format(letter),ha="right",va="center",weight="bold",fontsize=12,color="#2d2d2d",transform=self.ax.transAxes)
    
class fg(object):
    def __init__(self,fig):
        self.fig=fig
        
    def mm2inch(self,x):
        return x/25.4

    def setsize(self,width,height):
        m = self.mm2inch
        self.fig.set_size_inches( m(width), m(height) )

    def tl(self):
        self.fig.set_tight_layout(True)


