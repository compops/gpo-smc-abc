##############################################################################
##############################################################################
# Model specification
# Generalised Pareto distribution
# Version 2016-05-26
#
# Copyright (c) 2015 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy                 as     np
from   scipy.stats           import norm

class dist( object ):

    #=========================================================================
    # Define model settings
    #=========================================================================

    nPar          = 2;
    par           = np.zeros(nPar);
    modelName     = "pareto distribution"
    filePrefix    = "pareto";
    nParInference = None;
    nQInference   = None;

    #=========================================================================
    # Define the model
    #=========================================================================
    def loglikelihood( self, x, eta=None, beta=None ):

        if ( ( eta == None ) | ( beta == None ) ):
            eta  = self.par[0];
            beta = self.par[1];

        n    = len( x );

        ll = -n*np.log(beta) - ( 1.0 + eta**(-1.0) ) * np.sum( np.log( 1.0 + eta * x / beta ) );
        return ll

    def loglikelihoodBFGS( self, th ):
        eta  = th[0];
        beta = th[1];
        n    = len( self.thres );
        ll = -n*np.log(beta) - ( 1.0 + eta**(-1.0) ) * np.sum( np.log( 1.0 + eta * self.thres / beta ) );

        return -ll

    def loglikelihoodHessian( self, th ):
        eta  = th[0];
        beta = th[1];
        n    = len( self.thres );
        ll = -n*np.log(beta) - ( 1.0 + eta**(-1.0) ) * np.sum( np.log( 1.0 + eta * self.thres / beta ) );

        return ll

    def cdf( self, x, eta=None, beta=None ):

        if ( ( eta == None ) | ( beta == None ) ):
            eta  = self.par[0];
            beta = self.par[1];

        out = np.zeros( x.shape[0] );

        if ( eta >= 0.0 ):
            # x needs to be positive, otherwise cdf is zero
            idx = np.where( x > 0.0 )
            out[idx] = 1.0 - ( 1.0 + eta * x[idx] / beta )**(-eta**(-1));

        if ( eta < 0.0 ):
            # x needs to be positive and less than -beta/eta
            idx1 = np.where(  ( x > 0.0 ) & ( x < -beta/eta ) );
            idx2 = np.where( x > -beta/eta );

            out[idx1] = 1.0 - ( 1.0 + eta * x[idx1] / beta )**(-eta**(-1));
            out[idx2] =  x.shape[0] / ( x.shape[0] + 1.0 );

        return out;

    def pdf( self, x, eta=None, beta=None ):

        if ( ( eta == None ) | ( beta == None ) ):
            eta  = self.par[0];
            beta = self.par[1];

        return beta**(-1) * ( 1.0 + eta * x / beta )**( -(eta**(-1) + 1.0 ) )

    def logpdf( self, x, eta=None, beta=None ):

        if ( ( eta == None ) | ( beta == None ) ):
            eta  = self.par[0];
            beta = self.par[1];

        return - np.log( beta ) - (eta**(-1) + 1.0 ) * np.log( 1.0 + eta * x / beta )