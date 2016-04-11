##############################################################################
##############################################################################
# Routines for
# Particle filtering based on approximate Bayesian computations
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
# Particle filtering with ABC: main routine
##########################################################################

def proto_pf_abc(classSMC,sys):

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.filterType = "abcPF"
    setSettings(classSMC,"abcfilter");

    # Initalise variables
    a   = np.zeros((classSMC.nPart,sys.T));
    p   = np.zeros((classSMC.nPart,sys.T));
    v   = np.zeros((classSMC.nPart,sys.T));
    v1  = np.zeros((classSMC.nPart,sys.T));
    v2  = np.zeros((classSMC.nPart,sys.T));
    w   = np.zeros((classSMC.nPart,sys.T));
    xh  = np.zeros((sys.T,1));
    ess = np.zeros(sys.T);
    ll  = np.zeros(sys.T);

    # Generate or set initial state
    if ( classSMC.genInitialState ):
        p[:,0] = sys.generateInitialState( classSMC.nPart );
    else:
        p[:,0] = classSMC.xo;

    # Set tolerance parameter
    classSMC.epsilon = np.ones(sys.T) * classSMC.tolLevel;

    #=====================================================================
    # Run main loop
    #=====================================================================

    for tt in range(0, sys.T):
        if tt != 0:

            #==================================================================
            # Resample particles
            #==================================================================

            # If resampling is enabled
            if ( classSMC.resamplingInternal == 1 ):

                # Calculate ESS
                ess[tt] = (np.sum(w[:,tt-1]**2))**(-1)

                # Check if ESS if below threshold, then resample
                if ( ess[tt]  < (classSMC.nPart * classSMC.resampFactor)  ):

                    if classSMC.resamplingType == "stratified":
                        nIdx     = resampleStratified(w[:,tt-1]);
                        nIdx     = np.transpose(nIdx.astype(int));

                    elif classSMC.resamplingType == "systematic":
                        nIdx     = resampleSystematic(w[:,tt-1]);
                        nIdx     = np.transpose(nIdx.astype(int));

                    elif classSMC.resamplingType == "multinomial":
                        nIdx     = resampleMultinomial(w[:,tt-1]);

                else:
                    # No resampling
                    nIdx     = np.arange(0,classSMC.nPart);

            a[:,tt]  = nIdx;

            #==================================================================
            # Propagate particles
            #==================================================================
            p[:,tt] = sys.generateState   ( p[nIdx,tt-1], tt-1);

        #======================================================================
        # Weight particles
        #======================================================================
        (v[:,tt], v1[:,tt], v2[:,tt]) = sys.generateObservation(p[:,tt], tt);

        if ( classSMC.weightdist == "boxcar" ):

            # Standard ABC
            w[:,tt] = 1.0 * ( np.abs( v[:,tt] - sys.y[tt] ) < classSMC.tolLevel );

            # Calculate log-likelihood
            ll[tt] =  np.log( np.sum( w[:,tt] ) ) - np.log(classSMC.nPart) - np.log( classSMC.tolLevel );

        elif ( classSMC.weightdist == "gaussian" ):

            # Smooth ABC
            w[:,tt] = loguninormpdf( sys.y[tt], v[:,tt], classSMC.tolLevel );

            # Rescale log-weights and recover weights
            wmax    = np.max( w[:,tt] );
            w[:,tt] = np.exp( w[:,tt] - wmax );

            # Calculate log-likelihood
            ll[tt] = wmax + np.log( np.sum( w[:,tt] ) ) - np.log( classSMC.nPart );

        #======================================================================
        # Calculate state estimate
        #======================================================================
        w[:,tt] /= np.sum( w[:,tt] );
        xh[tt]  =  np.sum( w[:,tt] * p[:,tt] );

    #=====================================================================
    # Create output
    #=====================================================================
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum(ll);
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v     = v;
    classSMC.v1    = v1;
    classSMC.v2    = v2;
    classSMC.a     = a;
    classSMC.p     = p;

##########################################################################
# Particle filtering with ABC: adaptive routine
##########################################################################

def proto_pf_abc_adaptive(classSMC,sys):

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.filterType = "abcPF"
    setSettings(classSMC,"abcfilter");

    # Initalise variables
    a   = np.zeros((classSMC.nPart,sys.T));
    p   = np.zeros((classSMC.nPart,sys.T));
    v   = np.zeros((classSMC.nPart,sys.T));
    v1  = np.zeros((classSMC.nPart,sys.T));
    v2  = np.zeros((classSMC.nPart,sys.T));
    w   = np.zeros((classSMC.nPart,sys.T));
    xh  = np.zeros((sys.T,1));
    ess = np.zeros(sys.T);
    ll  = np.zeros(sys.T);

    # Generate or set initial state
    if ( classSMC.genInitialState ):
        p[:,0] = sys.generateInitialState( classSMC.nPart );

    else:
        p[:,0] = classSMC.xo;

    # Should the tolerance level be adapted
    if classSMC.adaptTolLevel:
        classSMC.epsilon = np.zeros(sys.T);
    else:
        classSMC.epsilon = np.ones(sys.T) * classSMC.tolLevel;

    #=====================================================================
    # Run main loop
    #=====================================================================

    for tt in range(0, sys.T):
        if tt != 0:

            #=============================================================
            # Resample particles
            #=============================================================

            # If resampling is enabled
            if ( classSMC.resamplingInternal == 1 ):

                # Calculate ESS
                ess[tt] = (np.sum(w[:,tt-1]**2))**(-1)

                # Check if ESS if below threshold, then resample
                if ( ess[tt]  < (classSMC.nPart * classSMC.resampFactor)  ):

                    if classSMC.resamplingType == "stratified":
                        nIdx     = resampleStratified(w[:,tt-1]);
                        nIdx     = np.transpose(nIdx.astype(int));

                        # Only resample particles with | u - y | > epsilon
                        if ( (classSMC.weightdist == "gaussian") & classSMC.rejectionSMC ):
                            tmpw       = np.where( w[:,tt-1] < -1.4189385332046727 );
                            tmpi       = np.arange(0,classSMC.nPart);
                            tmpi[tmpw] = nIdx[tmpw];
                            a[:,tt]    = tmpi;
                            nIdx       = tmpi;
                        else:
                            a[:,tt]    = nIdx;
                            nIdx       = nIdx;

                    elif classSMC.resamplingType == "systematic":
                        nIdx     = resampleSystematic(w[:,tt-1]);
                        nIdx     = np.transpose(nIdx.astype(int));

                        # Only resample particles with | u - y | > epsilon
                        if ( (classSMC.weightdist == "gaussian") & classSMC.rejectionSMC ):
                            tmpw       = np.where( w[:,tt-1] < -1.4189385332046727 );
                            tmpi       = np.arange(0,classSMC.nPart);
                            tmpi[tmpw] = nIdx[tmpw];
                            a[:,tt]    = tmpi;
                            nIdx       = tmpi;
                        else:
                            a[:,tt]    = nIdx;
                            nIdx       = nIdx;
                    elif classSMC.resamplingType == "multinomial":
                        nIdx     = resampleMultinomial(w[:,tt-1]);

                        # Only resample particles with | u - y | > epsilon
                        if ( (classSMC.weightdist == "gaussian") & classSMC.rejectionSMC ):
                            tmpw       = np.where( w[:,tt-1] < -1.4189385332046727 );
                            tmpi       = np.arange(0,classSMC.nPart);
                            tmpi[tmpw] = nIdx[tmpw];
                            a[:,tt]    = tmpi;
                            nIdx       = tmpi;
                        else:
                            a[:,tt]    = nIdx;
                            nIdx       = nIdx;
                else:
                    # No resampling
                    nIdx     = np.arange(0,classSMC.nPart);
                    a[:,tt]  = nIdx;

            #=============================================================
            # Propagate particles
            #=============================================================
            p[:,tt] = sys.generateState   ( p[nIdx,tt-1], tt-1);

        #=================================================================
        # Weight particles
        #=================================================================
        (v[:,tt], v1[:,tt], v2[:,tt]) = sys.generateObservation(p[:,tt], tt);

        # Adapt epsilon?
        if classSMC.adaptTolLevel:
            # Calculate distance and sort
            distances = np.abs( v[:,tt] - sys.y[tt] )
            sortedIdx = np.argsort(distances)

            # Set the epsilon to the propAlive*nPart-th distance
            classSMC.epsilon[tt] = distances[ sortedIdx[ np.floor(classSMC.propAlive*classSMC.nPart) ] ];

        if ( classSMC.weightdist == "boxcar" ):
            w[:,tt] = 1.0 * ( np.abs( v[:,tt] - sys.y[tt] ) < classSMC.epsilon[tt] );
        elif ( classSMC.weightdist == "gaussian" ):
            w[:,tt] = loguninormpdf( sys.y[tt], v[:,tt], classSMC.epsilon[tt] );

            # Rescale log-weights and recover weights
            wmax    = np.max( w[:,tt] );
            w[:,tt] = np.exp( w[:,tt] - wmax );

        # Estimate log-likelihood and normalise weights
#        if ( classSMC.weightdist == "boxcar" ):
#            ll[tt] =  np.log(np.nansum(w[:,tt])) - np.log(classSMC.nPart) - np.log( classSMC.epsilon[tt] );
#        elif ( classSMC.weightdist == "gaussian" ):
#            ll[tt] = wmax + np.log(np.nansum(w[:,tt])) - np.log( classSMC.nPart );
#
#        w[:,tt] /= np.nansum(w[:,tt]);
#        xh[tt]  = np.nansum( w[:,tt] * p[:,tt] );

        if ( classSMC.weightdist == "boxcar" ):
            ll[tt] =  np.log(np.sum(w[:,tt])) - np.log(classSMC.nPart) - np.log( classSMC.epsilon[tt] );
        elif ( classSMC.weightdist == "gaussian" ):
            ll[tt] = wmax + np.log(np.sum(w[:,tt])) - np.log( classSMC.nPart );

        w[:,tt] /= np.sum(w[:,tt]);
        xh[tt]  = np.sum( w[:,tt] * p[:,tt] );

    #=====================================================================
    # Create output
    #=====================================================================
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum(ll);
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v     = v;
    classSMC.v1    = v1;
    classSMC.v2    = v2;
    classSMC.a     = a;
    classSMC.p     = p;


##########################################################################
# Alive particle filtering with ABC
##########################################################################

def proto_pf_abc_alive(classSMC,sys):

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.filterType = "abcPF-alive"
    setSettings(classSMC,"abcfilter");

    # Initalise variables
    a   = np.zeros((classSMC.nPart,sys.T));
    p   = np.zeros((classSMC.nPart,sys.T));
    v   = np.zeros((classSMC.nPart,sys.T));
    v1  = np.zeros((classSMC.nPart,sys.T));
    v2  = np.zeros((classSMC.nPart,sys.T));
    w   = np.zeros((classSMC.nPart,sys.T));

    ta  = np.zeros(classSMC.nPart);
    tp  = np.zeros(classSMC.nPart);
    tv  = np.zeros(classSMC.nPart);
    tv1 = np.zeros(classSMC.nPart);
    tv2 = np.zeros(classSMC.nPart);
    tw  = np.zeros(classSMC.nPart);

    xh  = np.zeros((sys.T,1));
    ns  = np.zeros(sys.T);
    ll  = np.zeros(sys.T);

    # Tolerance level is fixed
    classSMC.epsilon = np.ones(sys.T) * classSMC.tolLevel;

    # Generate or set initial state
    if ( classSMC.genInitialState ):
        p[:,0] = sys.generateInitialState( classSMC.nPart );
        w[:,0] = np.ones(classSMC.nPart) / classSMC.nPart;
    else:
        p[:,0] = classSMC.xo;

    #=====================================================================
    # Run main loop
    #=====================================================================

    for tt in range(1, sys.T):

        notEnoughSamples = True;
        nSamples         = 0;
        nSamplesReq      = 0;

        while notEnoughSamples:

            #=============================================================
            # Resample particles
            #=============================================================
            if classSMC.resamplingType == "stratified":
                nIdx     = resampleStratified(w[:,tt-1]);
                nIdx     = np.transpose(nIdx.astype(int));
                ta       = nIdx;

            elif classSMC.resamplingType == "systematic":
                nIdx     = resampleSystematic(w[:,tt-1]);
                nIdx     = np.transpose(nIdx.astype(int));
                ta       = nIdx;

            elif classSMC.resamplingType == "multinomial":
                nIdx     = resampleMultinomial(w[:,tt-1]);
                ta       = nIdx;

            #=============================================================
            # Propagate particles
            #=============================================================
            tp = sys.generateState( p[nIdx,tt-1], tt-1);

            #=================================================================
            # Weight particles
            #=================================================================
            (tv, tv1, tv2) = sys.generateObservation( tp, tt);
            tw = 1.0 * ( np.abs( tv - sys.y[tt] ) < classSMC.epsilon );

            #=================================================================
            # Extract the alive particles
            #=================================================================
            mIdx      = ( np.where ( tw > 0.0 ) )[1];
            nSamples2 = nSamples + len(mIdx)

            if ( nSamples2 > classSMC.nPart ):
                left = classSMC.nPart - nSamples;
                mIdx    = mIdx[range(left)];
                nSamplesReq += classSMC.nPart - np.max(mIdx);

                nSamples2 = classSMC.nPart;
            else:
                nSamplesReq += classSMC.nPart;

            # Compile the results
            if ( len( mIdx ) > 0 ):
                a[nSamples:nSamples2,tt]  = ta[mIdx];
                p[nSamples:nSamples2,tt]  = tp[0,mIdx];
                v[nSamples:nSamples2,tt]  = tv[0,mIdx];
                v1[nSamples:nSamples2,tt] = tv1[0];
                v2[nSamples:nSamples2,tt] = tv2[0];
                w[nSamples:nSamples2,tt]  = tw[0,mIdx];

            nSamples     = nSamples2;

            if nSamples == classSMC.nPart:
                notEnoughSamples = False
                ns[tt]           = nSamplesReq

                p[classSMC.nPart-1,tt] = 0
                w[classSMC.nPart-1,tt] = 0

        # Estimate log-likelihood and normalise weights
        ll[tt] = np.log( (classSMC.nPart - 1.0) / ( ( nSamplesReq - 1.0 ) * classSMC.epsilon ) )
        w[:,tt] /= np.nansum(w[:,tt]);

        xh[tt]  = np.nansum( w[:,tt] * p[:,tt] );

    #=====================================================================
    # Create output
    #=====================================================================
    print("I used a maximum of " + str(np.max(ns)) + " attempts to obtain N: " + str(classSMC.nPart) + " particles.")
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum(ll);
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v1    = v1;
    classSMC.v2    = v2;
    classSMC.a     = a;
    classSMC.p     = p;
    classSMC.ns    = ns;

##########################################################################
# Smooth particle filter with ABC
##########################################################################

def proto_sPF_abc(classSMC,sys):

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    setSettings(classSMC,"smoothabcfilter");
    classSMC.filterType = "sPF_abc"

    # Initalise variables
    p      = np.zeros((classSMC.nPart,sys.T));
    pt     = np.zeros((classSMC.nPart,sys.T));
    v      = np.zeros((classSMC.nPart,sys.T));
    v1     = np.zeros((classSMC.nPart,sys.T));
    v2     = np.zeros((classSMC.nPart,sys.T));
    w      = np.zeros((classSMC.nPart,sys.T));
    r      = np.zeros((classSMC.nPart,sys.T));
    us     = np.zeros((classSMC.nPart,sys.T));
    xh     = np.zeros(sys.T);
    ll     = np.zeros(sys.T);

    # Set initial state
    p[:,0] = classSMC.xo;

    # Should the tolerance level be adapted
    if classSMC.adaptTolLevel:
        classSMC.epsilon = np.zeros(sys.T);
    else:
        classSMC.epsilon = np.ones(sys.T) * classSMC.tolLevel;

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
            p[:,tt] = sys.generateState   ( pt[:,tt], tt-1);
            p[:,tt] = np.sort( p[:,tt] );

        #=============================================================
        # Weight particles
        #=============================================================
        (v[:,tt], v1[:,tt], v2[:,tt]) = sys.generateObservation(p[:,tt], tt);

        # Adapt epsilon?
        if classSMC.adaptTolLevel:
            # Calculate distance and sort
            distances = np.abs( v[:,tt] - sys.y[tt] )
            sortedIdx = np.argsort(distances)

            # Set the epsilon to the propAlive*nPart-th distance
            classSMC.epsilon[tt] = distances[ sortedIdx[ np.floor(classSMC.propAlive*classSMC.nPart) ] ];

        if ( classSMC.weightdist == "boxcar" ):
            w[:,tt] = 1.0 * ( np.abs( v[:,tt] - sys.y[tt] ) / classSMC.epsilon[tt] < 1.0 );
        elif ( classSMC.weightdist == "gaussian" ):
            w[:,tt] = loguninormpdf( sys.y[tt], v[:,tt], classSMC.epsilon[tt] );

            # Rescale log-weights and recover weights
            wmax    = np.max(w[:,tt]);
            w[:,tt] = np.exp(w[:,tt] - wmax);

        # Estimate log-likelihood and normalise weights
        if ( classSMC.weightdist == "boxcar" ):
            ll[tt]   = np.log(np.nansum(w[:,tt])) - np.log(classSMC.nPart);
        elif ( classSMC.weightdist == "gaussian" ):
            ll[tt]   = wmax + np.log(np.nansum(w[:,tt])) - np.log(classSMC.nPart);

        w[:,tt] /= np.nansum(w[:,tt]);

        xh[tt]  = np.nansum( w[:,tt] * p[:,tt] );

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

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################