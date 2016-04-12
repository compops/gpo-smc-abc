##############################################################################
##############################################################################
# Routines for
# Particle smoothing
# Version 2015-03-12
#
# Copyright (c) 2015 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy                 as     np
from smc_resampling          import *
from smc_helpers             import *
from smc_additivefunctionals import *


##########################################################################
# Particle smoothing: fixed-lag smoother
##########################################################################
def proto_flPS(classSMC,sys,fixParticles=False):

    #=====================================================================
    # Initalisation
    #=====================================================================

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.smootherType = "fl"
    setSettings(classSMC,"flsmoother");

    # Run initial filter
    if ( fixParticles == False ):
        classSMC.filter(sys);

    # Initalise variables
    xs    = np.zeros((sys.T,1));
    g1    = np.zeros((sys.nParInference,sys.T));
    h1    = np.zeros((sys.nParInference,sys.nParInference));
    h2    = np.zeros((sys.nParInference,sys.nParInference));
    sal   = np.zeros((classSMC.nPart,sys.nParInference,sys.T))
    q2    = np.zeros((sys.nQInference,sys.T));

    #=====================================================================
    # Main loop
    #=====================================================================

    # Construct the sa from the filter smoother to compute the cross-term in Louis identity
    if ( classSMC.calcHessianFlag == "louis" ):
        for tt in range(0, sys.T-1):

            # Get the ancestor indicies for the filter smoother
            bt = classSMC.a[:,tt+1]; bt = bt.astype(int);

            # Estimate the alpha quantity
            # Discussed in Dahlin, Lindsten, Schon (2014): Particle Metropolis-Hastings using gradient and Hessian information

            if ( classSMC.calcGradientFlag ):
                if ( classSMC.filterType == "abcPF" ):
                    sys.epsilon = classSMC.epsilon[tt];
                    sal[:,:,tt+1] = sal[bt,:,tt] + sys.Dparm( classSMC.p[:,tt+1], classSMC.p[bt,tt], classSMC.v1[bt,tt], classSMC.v2[bt,tt], tt );
                else:
                    sal[:,:,tt+1] = sal[bt,:,tt] + sys.Dparm( classSMC.p[:,tt+1], classSMC.p[bt,tt], np.zeros(classSMC.nPart), bt, tt );

    # Run the fixed-lag smoother for the rest
    for tt in range(0, sys.T-1):
        at  = np.arange(0,classSMC.nPart)
        kk  = np.min( (tt+classSMC.fixedLag, sys.T-1) )

        # Reconstruct particle trajectory
        for ii in range(kk,tt,-1):
            att = at.astype(int);
            at  = at.astype(int);
            at  = classSMC.a[at,ii];
            at  = at.astype(int);

        # Estimate state
        xs[tt] = np.nansum( classSMC.p[at,tt] * classSMC.w[:, kk] );

        #=================================================================
        # Estimate score and terms for the Louis identity
        #=================================================================

        if ( classSMC.calcGradientFlag ):
            if ( classSMC.filterType == "abcPF" ):
                sys.epsilon = classSMC.epsilon[tt];

                sa = sys.Dparm  ( classSMC.p[att,tt+1], classSMC.p[at,tt], classSMC.v1[at,tt], classSMC.v2[at,tt], tt);

                if ( classSMC.calcHessianFlag != False ):
                    sb = sys.DDparm ( classSMC.p[att,tt+1], classSMC.p[at,tt], classSMC.v1[at,tt], classSMC.v2[at,tt], tt);
            else:
                sa = sys.Dparm  ( classSMC.p[att,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);

                if ( classSMC.calcHessianFlag != False ):
                    sb = sys.DDparm ( classSMC.p[att,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);

            for nn in range(0,sys.nParInference):
                g1[nn,tt]       = np.nansum( sa[:,nn] * classSMC.w[:,kk] );

                if ( classSMC.calcHessianFlag == "louis" ):
                    for mm in range(0,(nn+1)):
                        h1[nn,mm] += np.nansum( classSMC.w[:,kk] * sb[:,nn,mm] );
                        h2[nn,mm] += np.nansum( classSMC.w[:,kk] * ( sa[:,nn]*sa[:,mm] + sa[:,nn]*sal[at,mm,tt] + sal[at,nn,tt]*sa[:,mm] ) );

                        h1[mm,nn]  = h1[nn,mm];
                        h2[mm,nn]  = h2[nn,mm];

        #=================================================================
        # Estimate of Q-function for EM algorithm
        #=================================================================
        if ( classSMC.calcQFlag  ):
            q1 = sys.Qfunc  ( classSMC.p[att,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);
            for nn in range(0,sys.nQInference):
                q2[nn,tt]       += np.nansum( q1[:,nn] * classSMC.w[:,kk] );

    #=====================================================================
    # Estimate additive functionals and write output
    #=====================================================================
    calcGradient(classSMC,sys,g1);
    calcHessian(classSMC,sys,g1,h1,h2);
    calcQ(classSMC,sys,q2);
    classSMC.g1 = g1;

    # Save the smoothed state estimate
    classSMC.xhats = xs;
    classSMC.q2 = q2;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################