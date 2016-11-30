##############################################################################
##############################################################################
# Subroutines for calculating gradients and Hessians using
# Particle filtering and smoothing
#
# Copyright (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

import numpy as np

##########################################################################
# Helper: calculate the gradient of the log-target
##########################################################################


def calcGradient(sm, sys, term1):

    if (sm.calcGradientFlag):

        # Check dimensions of the input
        if (len(term1.shape) == 2):
            # Sum up the contributions from each time step
            gradient = np.nansum(term1, axis=1)
        else:
            gradient = term1

        # Add the gradient of the log-prior
        for nn in range(0, sys.nParInference):
            gradient[nn] = sys.dprior1(nn) + gradient[nn]

        # Write output
        sm.gradient = gradient
        sm.gradient0 = term1

##########################################################################
# Helper: calculate the Hessian of the log-target
##########################################################################


def calcHessian(sm, sys, term1, term2=0, term3=0):

    # Write output
    # sm.gradient0 = 0.0
    sm.hessian0 = 0.0
    sm.hessian1 = 0.0

##########################################################################
# Helper: calculate the Q-function for the EM-algorithm
##########################################################################


def calcQ(sm, sys, term1):
    sm.qfunc = 0.0

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
