##############################################################################
##############################################################################
# Routines for
# Particle filtering with smooth/continious resampling
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

##########################################################################
# Smooth particle filter
##########################################################################
def proto_sPF(classSMC,sys):

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    setSettings(classSMC,"smoothfilter");
    classSMC.filterType = "sPF"

    # Initalise variables
    p      = np.zeros((classSMC.nPart,sys.T));
    pt     = np.zeros((classSMC.nPart,sys.T));
    v      = np.zeros((classSMC.nPart,sys.T));
    w      = np.zeros((classSMC.nPart,sys.T));
    r      = np.zeros((classSMC.nPart,sys.T));
    us     = np.zeros((classSMC.nPart,sys.T));
    xh     = np.zeros(sys.T);
    ll     = np.zeros(sys.T);

    # Generate or set initial state
    if ( classSMC.genInitialState ):
        p[:,0] = sys.generateInitialState( classSMC.nPart );
        w[:,0] = 1.0 / classSMC.nPart;
    else:
        p[:,0] = classSMC.xo;
        w[:,0] = 1.0 / classSMC.nPart;

    # Fix for continuous PF, to fix the random numbers in the resampling and propagation
    np.random.seed( classSMC.seed );

    for tt in range(0, sys.T):
        if tt != 0:

            #=============================================================
            # Resample particles
            #=============================================================
            ( r[:,tt] , us[:,tt] ) = resampleCont( w[:,tt-1], classSMC.nPart )

            # Calculate the resulting particles
            tmp        = np.where( r[:,tt] == 0 );
            pt[tmp,tt] = p[0,  tt-1];

            tmp        = np.where( r[:,tt] == classSMC.nPart-1 );
            pt[tmp,tt] = p[classSMC.nPart-1,tt-1];

            tmp        = np.where( ( r[:,tt] < classSMC.nPart-1 ) & ( r[:,tt] > 0 ) );
            prjj       = p[ (r[tmp,tt]+1).astype(int), tt-1];
            prj        = p[  r[tmp,tt].astype(int),    tt-1];
            pt[tmp,tt] = ( prjj - prj ) * us[tmp,tt] + prj;

            #=============================================================
            # Propagate particles
            #=============================================================
            if ( classSMC.filterTypeInternal == "bootstrap" ):
                p[:,tt] = sys.generateState   ( pt[:,tt], tt-1);
            elif ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
                p[:,tt] = sys.generateStateFA ( pt[:,tt], tt-1);

            if ( classSMC.sortParticles == True ):
                p[:,tt] = np.sort( p[:,tt] );

        #=============================================================
        # Weight particles
        #=============================================================
        if ( classSMC.filterTypeInternal == "bootstrap" ):
            w[:,tt] = sys.evaluateObservation   ( p[:,tt], tt);
        elif ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            w[:,tt] = sys.evaluateObservationFA ( p[:,tt], tt);

        # Rescale log-weights and recover weights
        wmax    = np.max( w[:,tt] );
        w[:,tt] = np.exp( w[:,tt] - wmax );

        # Estimate log-likelihood
        ll[tt] = wmax + np.log( np.sum( w[:,tt] )) - np.log( classSMC.nPart );
        w[:,tt] /= np.sum(w[:,tt]);

        # Calculate the normalised filter weights (1/N) as it is a FAPF
        if ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            v[:,tt] = w[:,tt];
            w[:,tt] = np.ones(classSMC.nPart) / classSMC.nPart;

        # Estimate the filtered state
        xh[tt]  = np.sum( w[:,tt] * p[:,tt] );

    #=============================================================
    # Compile output
    #=============================================================
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum(ll);
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v     = v;
    classSMC.p     = p;
    classSMC.r     = r;
    classSMC.us    = us;

##########################################################################
# Smooth particle filter with same order as in Pitt (2002)
##########################################################################
def proto_sPFpitt(classSMC,sys):

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    setSettings(classSMC,"smoothfilterPitt");
    classSMC.filterType = "sPF"

    # Initalise variables
    p      = np.zeros((classSMC.nPart2,sys.T));
    pt     = np.zeros((classSMC.nPart,sys.T));
    v      = np.zeros((classSMC.nPart2,sys.T));
    w      = np.zeros((classSMC.nPart2,sys.T));
    r      = np.zeros((classSMC.nPart,sys.T));
    us     = np.zeros((classSMC.nPart,sys.T));
    xh     = np.zeros(sys.T);
    ll     = np.zeros(sys.T);

    # Generate or set initial state
    if ( classSMC.genInitialState ):
        p[:,0] = sys.generateInitialState( classSMC.nPart );
        w[:,0] = 1.0 / classSMC.nPart;
    else:
        p[:,0] = classSMC.xo;
        w[:,0] = 1.0 / classSMC.nPart;

    # Fix for continuous PF, to fix the random numbers in the resampling and propagation
    np.random.seed( classSMC.seed );

    for tt in range(1, sys.T):

        #=============================================================
        # Resample particles (stratified)
        #=============================================================
        nIdx     = resampleSystematic( np.ones(classSMC.nPart).astype(float), classSMC.nPart2 )
        pt[:,tt] = p[nIdx,tt-1]

        #=============================================================
        # Propagate particles
        #=============================================================
        if ( classSMC.filterTypeInternal == "bootstrap" ):
            p[:,tt] = sys.generateState   ( pt[:,tt], tt-1);
        elif ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            p[:,tt] = sys.generateStateFA ( pt[:,tt], tt-1);

        if ( classSMC.sortParticles == True ):
            p[:,tt] = np.sort( p[:,tt] );

        #=============================================================
        # Weight particles
        #=============================================================
        if ( classSMC.filterTypeInternal == "bootstrap" ):
            w[:,tt] = sys.evaluateObservation   ( p[:,tt], tt);
        elif ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            w[:,tt] = sys.evaluateObservationFA ( p[:,tt], tt);

        # Rescale log-weights and recover weights
        wmax    = np.max( w[:,tt] );
        w[:,tt] = np.exp( w[:,tt] - wmax );

        # Estimate log-likelihood
        ll[tt]   = wmax + np.log( np.sum( w[:,tt] ) ) - np.log( classSMC.nPart2 );
        w[:,tt] /= np.sum( w[:,tt] );

        # Calculate the normalised filter weights (1/N) as it is a FAPF
        if ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            v[:,tt] = w[:,tt];
            w[:,tt] = np.ones(classSMC.nPart) / classSMC.nPart;

        # Estimate the filtered state
        xh[tt]  = np.sum( w[:,tt] * p[:,tt] );

        #=============================================================
        # Resample particles (continious)
        #=============================================================
        ( r[:,tt] , us[:,tt] )     = resampleCont( w[:,tt], classSMC.nPart )

        # Calculate the resulting particles
        tmp        = np.where( r[:,tt] == 0 );
        pt[tmp,tt] = p[0,  tt];

        tmp        = np.where( r[:,tt] == classSMC.nPart2-1 );
        pt[tmp,tt] = p[classSMC.nPart2-1,tt];

        tmp        = np.where( ( r[:,tt] < classSMC.nPart2-1 ) & ( r[:,tt] > 0 ) );
        prjj       = p[ (r[tmp,tt]+1).astype(int), tt];
        prj        = p[  r[tmp,tt].astype(int),    tt];
        pt[tmp,tt] = ( prjj - prj ) * us[tmp,tt] + prj;

    #=============================================================
    # Compile output
    #=============================================================
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum(ll);
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v     = v;
    classSMC.p     = p;
    classSMC.pt    = pt;
    classSMC.r     = r;
    classSMC.us    = us;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################