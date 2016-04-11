##############################################################################
##############################################################################
# Routines for
# Particle filtering (bootstrap, fully adapted and partially adapted)
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
# Bootstrap and fully adapted particle filtering
##########################################################################

def proto_pf(classSMC,sys):

    # Set the filter type and save T
    classSMC.T = sys.T;

    # Check algorithm settings and set to default if needed
    setSettings(classSMC,"filter");

    # Initalise variables
    a   = np.zeros((classSMC.nPart,sys.T));
    ar  = np.zeros((classSMC.nPart,sys.T));
    p   = np.zeros((classSMC.nPart,sys.T));
    pt  = np.zeros((classSMC.nPart,sys.T));
    v   = np.zeros((classSMC.nPart,sys.T));
    w   = np.zeros((classSMC.nPart,sys.T));
    xh  = np.zeros((sys.T,1));
    ll  = np.zeros(sys.T);
    ess = np.zeros(sys.T);

    # Generate or set initial state
    if ( classSMC.genInitialState ):
        p[:,0] = sys.generateInitialState( classSMC.nPart );
        w[:,0] = 1.0 / classSMC.nPart;
    else:
        p[:,0] = classSMC.xo;
        w[:,0] = 1.0 / classSMC.nPart;

    #=====================================================================
    # Run main loop
    #=====================================================================

    for tt in range(1, sys.T):
        if tt != 0:

            #=============================================================
            # Resample particles
            #=============================================================

            # If resampling is enabled
            if ( classSMC.resamplingInternal == 1 ):

                # Calculate ESS
                ess[tt] = ( np.sum( w[:,tt-1]**2 ) )**(-1)

                # Check if ESS if below threshold, then resample
                if ( ess[tt] < ( classSMC.nPart * classSMC.resampFactor )  ):

                    if classSMC.resamplingType == "stratified":
                        nIdx            = resampleStratified(w[:,tt-1]);
                        nIdx            = np.transpose(nIdx.astype(int));
                        pt[:,tt]        = p[nIdx,tt-1];
                        ar[:,0:(tt-1)]  = ar[nIdx,0:(tt-1)];
                        ar[:,tt]        = nIdx;
                        a[:,tt]         = nIdx;
                    elif classSMC.resamplingType == "systematic":
                        nIdx            = resampleSystematic(w[:,tt-1]);
                        nIdx            = np.transpose(nIdx.astype(int));
                        pt[:,tt]        = p[nIdx,tt-1];
                        ar[:,0:(tt-1)]  = ar[nIdx,0:(tt-1)];
                        ar[:,tt]        = nIdx;
                        a[:,tt]         = nIdx;
                    elif classSMC.resamplingType == "multinomial":
                        nIdx            = resampleMultinomial(w[:,tt-1]);
                        nIdx            = np.transpose(nIdx.astype(int));
                        pt[:,tt]        = p[nIdx,tt-1];
                        ar[:,0:(tt-1)]  = ar[nIdx,0:(tt-1)];
                        ar[:,tt]        = nIdx;
                        a[:,tt]         = nIdx;
                else:
                    # No resampling
                    nIdx                = np.arange(0,classSMC.nPart);
                    nIdx                = np.transpose(nIdx.astype(int));
                    pt[:,tt]            = p[nIdx,tt-1];
                    a[:,tt]             = nIdx;
                    ar[:,tt]            = nIdx;
            else:
                pt[:,tt]            = p[:,tt-1];

            #=============================================================
            # Propagate particles
            #=============================================================
            if ( classSMC.filterTypeInternal == "bootstrap" ):
                p[:,tt] = sys.generateState   ( pt[:,tt], tt-1);
            elif ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
                p[:,tt] = sys.generateStateFA ( pt[:,tt], tt-1);

            # Conditioning (for the conditional particle filter)
            if ( classSMC.condFilterInternal == 1 ):
                p[classSMC.nPart-1, tt] = classSMC.condPath[0,tt];
                a[classSMC.nPart-1, tt] = classSMC.nPart-1;

            # Conditioning
            if ( classSMC.condFilterInternal == 1 ):
                p[classSMC.nPart-1, tt] = classSMC.condPath[0,tt];

                if ( classSMC.ancestorSamplingInternal == 1 ):
                    # Ancestor sampling, compute weights
                    if ( classSMC.filterTypeInternal == "bootstrap" ):
                        tmp  = sys.evaluateState   ( classSMC.condPath[0,tt], p[:,tt], tt );
                    elif ( classSMC.filterTypeInternal == "fullyadapted" ):
                        tmp  = sys.evaluateStateFA   ( classSMC.condPath[0,tt], p[:,tt], tt );

                    tmax = np.max(tmp);
                    tmp  = np.exp(tmp - tmax);
                    tmp  = tmp * w[:,tt-1];

                    # Ancestor sampling, normalize weights and sample ancestor index
                    tmp = tmp / np.nansum( tmp );
                    classSMC.tmp = tmp;
                    a[classSMC.nPart-1, tt] = np.random.choice(classSMC.nPart, 1, p=tmp);

                else:
                    a[classSMC.nPart-1, tt] = classSMC.nPart-1;

        #=================================================================
        # Weight particles
        #=================================================================
        if ( classSMC.filterTypeInternal == "bootstrap" ):
            w[:,tt] = sys.evaluateObservation   ( p[:,tt], tt);
        elif ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            w[:,tt] = sys.evaluateObservationFA ( p[:,tt], tt);

        # Rescale log-weights and recover weights
        wmax    = np.max( w[:,tt] );
        w[:,tt] = np.exp( w[:,tt] - wmax );

        # Estimate log-likelihood
        ll[tt]   = wmax + np.log( np.sum( w[:,tt] ) ) - np.log(classSMC.nPart);
        w[:,tt] /= np.sum( w[:,tt] );

        # Calculate the normalised filter weights (1/N) as it is a FAPF
        if ( ( classSMC.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
            v[:,tt] = w[:,tt];
            w[:,tt] = np.ones(classSMC.nPart) / classSMC.nPart;

        # Estimate the filtered state
        xh[tt]  = np.sum( w[:,tt] * p[:,tt] );

    #=====================================================================
    # Create output
    #=====================================================================

    # Sample a trajectory
    idx            = np.random.choice( classSMC.nPart, 1, p=w[:,sys.T-1] )
    idx            = ar[idx,sys.T-1].astype(int);
    classSMC.xtraj = p[idx,:]

    # Compile the rest of the output
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum( ll );
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v     = v;
    classSMC.a     = a;
    classSMC.ar    = ar;
    classSMC.p     = p;
    classSMC.pt    = pt;
    classSMC.ess   = ess;

##########################################################################
# Partially adapted particle filtering
##########################################################################

def proto_papf(classSMC,sys):

    # Set the filter type and save T
    classSMC.T = sys.T;

    # Check algorithm settings and set to default if needed
    setSettings(classSMC,"filter");

    # Initalise variables
    a   = np.zeros((classSMC.nPart,sys.T));
    p   = np.zeros((classSMC.nPart,sys.T));
    pam = np.zeros((classSMC.nPart,sys.T));
    pas = np.zeros((classSMC.nPart,sys.T));
    pt  = np.zeros((classSMC.nPart,sys.T));
    v   = np.zeros((classSMC.nPart,sys.T));
    w   = np.zeros((classSMC.nPart,sys.T));
    xh  = np.zeros((sys.T,1));
    ll  = np.zeros(sys.T);
    ess = np.zeros(sys.T);

    # Generate or set initial state
    if ( classSMC.genInitialState ):
        p[:,0] = sys.generateInitialState( classSMC.nPart );
        w[:,0] = 1.0 / classSMC.nPart;
    else:
        p[:,0] = classSMC.xo;
        w[:,0] = 1.0 / classSMC.nPart;

    #=====================================================================
    # Run main loop
    #=====================================================================

    for tt in range(1, sys.T):

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
                    pt[:,tt] = p[nIdx,tt-1];
                    a[:,tt]  = nIdx;
                elif classSMC.resamplingType == "systematic":
                    nIdx     = resampleSystematic(w[:,tt-1]);
                    nIdx     = np.transpose(nIdx.astype(int));
                    pt[:,tt] = p[nIdx,tt-1];
                    a[:,tt]  = nIdx;
                elif classSMC.resamplingType == "multinomial":
                    nIdx     = resampleMultinomial(w[:,tt-1]);
                    nIdx     = np.transpose(nIdx.astype(int));
                    pt[:,tt] = p[nIdx,tt-1]
                    a[:,tt]  = nIdx;
            else:
                # No resampling
                nIdx     = np.arange(0,classSMC.nPart);
                pt[:,tt] = p[nIdx,tt-1];
                a[:,tt]  = nIdx;
        else:
            pt[:,tt] = p[:,tt-1];

        #=============================================================
        # Propagate particles
        #=============================================================

        # Run quasi-Newton to obtain mode and estimate of the Hessian
        # Basically a Laplace approximation of the optimal proposal
        if (classSMC.doPartialOptForAll == True ):
            for kk in range(classSMC.nPart):
                res = optimize.minimize(sys.paOpt, pam[kk,tt-1], method='BFGS', jac=sys.paOptGradient, options={'maxiter': 5}, args=((pt[kk,tt],tt)) )
                pam[kk,tt] = res.x
                pas[kk,tt] = np.sqrt( res.hess_inv )
        else:
            # Do the optimisation for only a single random particle
            kk = np.random.choice( classSMC.nPart, 1, p = w[:,tt-1] )
            res = optimize.minimize(sys.paOpt, pam[kk,tt-1], method='BFGS', jac=sys.paOptGradient, options={'maxiter': 5}, args=((pt[kk,tt],tt)) )
            pam[:,tt] = res.x * np.ones(classSMC.nPart)
            pas[:,tt] = np.sqrt( res.hess_inv ) * np.ones(classSMC.nPart)

        p[:,tt] = pam[:,tt] + pas[:,tt] * np.random.normal(size=classSMC.nPart);

        # Conditioning (for the conditional particle filter)
        if ( classSMC.condFilterInternal == 1 ):
            p[classSMC.nPart-1, tt] = classSMC.condPath[0,tt];
            a[classSMC.nPart-1, tt] = classSMC.nPart-1;

        # Conditioning
        if ( classSMC.condFilterInternal == 1 ):
            p[classSMC.nPart-1, tt] = classSMC.condPath[0,tt];

            if ( classSMC.ancestorSamplingInternal == 1 ):
                # Ancestor sampling, compute weights
                tmp  = sys.evaluateState   ( classSMC.condPath[0,tt], p[:,tt], tt );

                tmax = np.max(tmp);
                tmp  = np.exp(tmp - tmax);
                tmp  = tmp * w[:,tt-1];

                # Ancestor sampling, normalize weights and sample ancestor index
                tmp = tmp / np.nansum( tmp );
                classSMC.tmp = tmp;
                a[classSMC.nPart-1, tt] = np.random.choice(classSMC.nPart, 1, p=tmp);

            else:
                a[classSMC.nPart-1, tt] = classSMC.nPart-1;

        #=================================================================
        # Weight particles
        #=================================================================
        w[:,tt] = -1.0 * sys.paOpt( pam[:,tt], pt[:,tt], tt ) + pas[:,tt];

        # Rescale log-weights and recover weights
        wmax    = np.max( w[:,tt] );
        w[:,tt] = np.exp( w[:,tt] - wmax );

        # Estimate log-likelihood
        ll[tt]   = wmax + np.log(np.sum(w[:,tt])) - np.log(classSMC.nPart);
        w[:,tt] /= np.sum(w[:,tt]);

        # Estimate the filtered state
        if ( (tt+1) < sys.T ):
            xh[tt+1]  = np.sum( w[:,tt] * p[:,tt] );

    #=====================================================================
    # Create output
    #=====================================================================
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum(ll);
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v     = v;
    classSMC.a     = a;
    classSMC.p     = p;
    classSMC.pt    = pt;
    classSMC.pam   = pam;
    classSMC.pas   = pas;
    classSMC.ess   = ess;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################