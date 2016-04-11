import numpy as np
import matplotlib.pyplot as plt
from distributions import *
from resampling import *

##########################################################################
# Helper: Present the path degenerecy
##########################################################################

def plotTrajectories(sm,sys):
    #plt.plot(sys.T-1*np.ones(sm.nPart),sm.p[:,sys.T-1],'k.'); plt.axis((0,sys.T,-2,2))
    
    #plt.hold("on");
    
    # Plot all the particles and their resampled ancestors
    for ii in range(0,sm.nPart):
        att = ii;
        for tt in np.arange(sys.T-2,0,-1):
            at = sm.a[att,tt+1]; at = at.astype(int);
            #plt.plot(tt,sm.p[at,tt],'k.');
            #plt.plot((tt,tt+1),(sm.p[at,tt],sm.p[att,tt+1]),'k');
            att = at; att = att.astype(int);
    
    #plt.hold("off")
