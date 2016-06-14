##############################################################################
##############################################################################
# Model specification
# Gaussian copula
# Version 2016-05-26
#
# Copyright (c) 2015 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy                 as     np
from   scipy.stats           import multivariate_normal
from   scipy.stats           import norm
from   models.models_helpers import *
from   models.models_dists   import *

class copula(object):

    #=========================================================================
    # Define model settings
    #=========================================================================

    nPar          = 3;
    par           = np.zeros(nPar);
    modelName     = "gaussian-copula"
    filePrefix    = "gcopula";
    nParInference = None;
    nQInference   = None;

    #=========================================================================
    # Define the model
    #=========================================================================
#    def constructCorrelationMatrix( self, p ):
#        # Construct the correlation matrix
#        Sigma = np.ones((p,p))
#
#        if ( self.covarianceType == "unstructured" ):
#            kk    = 0;
#            for ii in range(p):
#                for jj in range(ii+1,p):
#                    Sigma[ii,jj] = self.par[kk];
#                    Sigma[jj,ii] = self.par[kk];
#                    kk += 1;
#
#        elif ( self.covarianceType == "exchangable" ):
#            for ii in range(p):
#                for jj in range(ii+1,p):
#                    Sigma[ii,jj] = self.par[0];
#                    Sigma[jj,ii] = self.par[0];
#
#        elif ( self.covarianceType == "ar1" ):
#            for ii in range(p):
#                for jj in range(ii+1,p):
#                    Sigma[ii,jj] = self.par[0]**(jj-ii);
#                    Sigma[jj,ii] = self.par[0]**(jj-ii);
#
#        elif ( self.covarianceType == "toeplitz" ):
#            for ii in range(p):
#                for jj in range(ii+1,p):
#                    Sigma[ii,jj] = self.par[(jj-ii)];
#                    Sigma[jj,ii] = self.par[(jj-ii)];
#
#        self.Sigma = Sigma;

    def constructCorrelationMatrix( self, p ):
        # Construct the correlation matrix
        A = np.ones((p,p))

        kk    = 0;
        for ii in range(p):
            A[ii,ii] = 1.0;
            for jj in range(ii+1,p):
                A[jj,ii] = self.par[kk];
                kk += 1;

        sigma  = np.dot(A,A.transpose());
        delta  = np.diag( 1.0 / np.sqrt( np.diag( sigma ) ) )
        self.P = np.dot( np.dot( delta, sigma ), delta )

    def evaluateLogLikelihood( self, sys ):

        # Extract the degrees of freedom and the dimension
        p     = (self.uhat).shape[1];
        n     = (self.uhat).shape[0];

        self.constructCorrelationMatrix(p);

        # Compute the percentile function on univariate Gaussian
        norm_uhat  = norm.ppf(self.uhat, 0, 1);

        # Calculate the log-likelihood
        out = 0;

        for ii in range( n ):
            out += multivariate_normal.logpdf( norm_uhat[ii,:], np.zeros(p), self.P, allow_singular = True )

        self.ll    = out


    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================
    def priorUniform(self):
        out = 1.0;

        for ii in range( self.nPar ):
            if ( np.abs( self.par[ii] ) > 1.0 ):
                out = 0.0;

        return( out );

    # Standard operations on struct
    copyData                = template_copyData;
    storeParameters         = template_storeParameters;
    returnParameters        = template_returnParameters

    # No tranformations available
    transform               = empty_transform;
    invTransform            = empty_invTransform;
    Jacobian                = empty_Jacobian;