##############################################################################
##############################################################################
# Subroutines for calculating gradients and Hessians using
# Particle filtering and smoothing
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy as np

##########################################################################
# Helper: Hessian estimators (Newey-West)
##########################################################################
def NeweyWestInfoEstimator(sm,score,sys):

    if ( sm.NeweyWestLag == None ):
        print("NeweyWest: lag missing, defaulting to using 5.")
        sm.NeweyWestLag = 5;

    infom = np.zeros((sys.nParInference,sys.nParInference));

    for jj in range(0,sm.NeweyWestLag):
        omega = np.zeros((sys.nParInference,sys.nParInference))

        for tt in range(0,sys.T-jj):
            omega += np.mat( score[:,tt] ).T * np.mat( score[:,tt+jj] );

        w      = 1.0 - jj / ( sm.NeweyWestLag + 1.0);
        infom += w * ( omega + omega.T )

    return infom;

##########################################################################
# Helper: Hessian estimators (Segal-Weinstein)
##########################################################################
def SegelWeinsteinInfoEstimator(sm,score,sys):
    s = np.sum(score, axis=1);
    infom = np.dot(np.mat(score),np.mat(score).transpose()) - np.dot(np.mat(s).transpose(),np.mat(s)) / sys.T;
    return infom;

##########################################################################
# Helper: calculate the gradient of the log-target
##########################################################################
def calcGradient(sm,sys,term1):

    if (sm.calcGradientFlag):

        # Check dimensions of the input
        if ( len(term1.shape) == 2):
            # Sum up the contributions from each time step
            gradient = np.nansum(term1,axis=1);
        else:
            gradient = term1;

        # Add the gradient of the log-prior
        for nn in range(0,sys.nParInference):
            gradient[nn]     = sys.dprior1(nn) + gradient[nn];

        # Write output
        sm.gradient  = gradient;
        sm.gradient0 = term1;

##########################################################################
# Helper: calculate the Hessian of the log-target
##########################################################################
def calcHessian(sm,sys,term1,term2=0,term3=0):

    # Select method of calculation
    if ( sm.calcHessianFlag == "louis" ):
        sm.infom = np.mat(np.sum(term1,axis=1)).T * np.mat(np.sum(term1,axis=1)) - ( term2 + term3 );
    elif ( sm.calcHessianFlag == "neweywest" ):
        sm.hessian = NeweyWestInfoEstimator(sm,term1,sys)
    elif ( sm.calcHessianFlag == "segelweinstein" ):
        sm.hessian = SegelWeinsteinInfoEstimator(sm,term1,sys)
    elif ( sm.calcHessianFlag == False ):
        sm.hessian = np.zeros((sys.nParInference,sys.nParInference));
    else:
        raise NameError("calcHessian: unknown method selected.")

    # Add the Hessian of the log-prior
    if ( sm.calcHessianFlag != False ):
        for nn in range(0,sys.nParInference):
            for mm in range(0,sys.nParInference):
                sm.hessian[nn,mm] = sys.ddprior1(nn,mm) + sm.hessian[nn,mm];

    # Write output
    sm.gradient0 = term1
    sm.hessian0  = term2
    sm.hessian1  = term3

##########################################################################
# Helper: calculate the Q-function for the EM-algorithm of FS
##########################################################################
def calcQfs(sm,sys,term1):

    if ( sm.calcQFlag  ):
        q0 = np.zeros(sys.nQInference );

        for nn in range(0,sys.nQInference):
            if ( sm.filterTypeInternal == "bootstrap" ):
                q0[nn]   = np.sum( term1[:,nn] * sm.w[:,sys.T-1] );
            elif ( sm.filterTypeInternal == "fullyadapted" ):
                # TODO: T-2 or T-1??
                q0[nn]   = np.sum( term1[:,nn] * sm.w[:,sys.T-2] );

        sm.qfunc  = q0;

##########################################################################
# Helper: calculate the Q-function for the EM-algorithm
##########################################################################
def calcQ(sm,sys,term1):
    if ( sm.calcQFlag  ):
        if ( len(term1.shape) == 2):
            sm.qfunc  = np.sum(term1,axis=1);
        else:
            sm.qfunc  = term1;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################