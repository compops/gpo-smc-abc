##############################################################################
##############################################################################
# Model specification
# T copula
# Version 2016-05-26
#
# Copyright (c) 2015 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy                 as     np
from   scipy.stats           import t
from   scipy.stats           import gamma
from   scipy.stats           import beta
from   models.models_helpers import *
from   models.models_dists   import *

class copula(object):

    #=========================================================================
    # Define model settings
    #=========================================================================

    nPar          = 3;
    par           = np.zeros(nPar);
    modelName     = "t-copula"
    filePrefix    = "tcopula";
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
#            kk    = 1;
#            for ii in range(p):
#                for jj in range(ii+1,p):
#                    Sigma[ii,jj] = self.par[kk];
#                    Sigma[jj,ii] = self.par[kk];
#                    kk += 1;
#
#        elif ( self.covarianceType == "exchangable" ):
#            for ii in range(p):
#                for jj in range(ii+1,p):
#                    Sigma[ii,jj] = self.par[1];
#                    Sigma[jj,ii] = self.par[1];
#
#        elif ( self.covarianceType == "ar1" ):
#            for ii in range(p):
#                for jj in range(ii+1,p):
#                    Sigma[ii,jj] = self.par[1]**(jj-ii);
#                    Sigma[jj,ii] = self.par[1]**(jj-ii);
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

        kk    = 1;
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
        df    = self.par[0];
        p     = (self.uhat).shape[1];
        n     = (self.uhat).shape[0];

        self.constructCorrelationMatrix(p);

        # Compute the percentile function on univariate t
        tppf_uhat  = t.ppf( self.uhat, df );

        # Calculate the first part of the log-likelihood
        part1 = 0;

        for ii in range( n ):

            part1 += multiTLogPDF( tppf_uhat[ii,:], np.zeros(p), self.P, df, p )

        # Calculate the second part of the log-likelihood
        part2      = np.sum( t.logpdf( tppf_uhat, df ) );

        self.ll        = part1 - part2;

    def evaluateLogLikelihoodBFGS( self, df ):

        # Extract the degrees of freedom and the dimension
        p     = (self.uhat).shape[1];
        n     = (self.uhat).shape[0];

        self.constructCorrelationMatrix( p );

        # Compute the percentile function on univariate t
        tppf_uhat  = t.ppf( self.uhat, df );

        # Calculate the first part of the log-likelihood
        part1 = 0;

        for ii in range( n ):

            part1 += multiTLogPDF( tppf_uhat[ii,:], np.zeros(p), self.P, df, p )

        # Calculate the second part of the log-likelihood
        part2      = np.sum( t.logpdf( tppf_uhat, df ) );

        return     part2 - part1;

    ######################################################################
    # Estimate the log-posterior
    ######################################################################

    def evaluateLogPosteriorBFGS( self, par ):

        if hasattr(par, "__len__"):

            self.par[0] = par[0]

            if ( len(par) >= 2 ):
                self.par[1] = par[1]
            if ( len(par) >= 3 ):
                self.par[2] = par[2]
            if ( len(par) >= 4 ):
                self.par[3] = par[3]
        else:
            self.par[0] = par;

        # Extract the degrees of freedom and the dimension
        p     = (self.uhat).shape[1];
        n     = (self.uhat).shape[0];

        self.constructCorrelationMatrix( p );

        # Compute the percentile function on univariate t
        tppf_uhat  = t.ppf( self.uhat, self.par[0] );

        # Calculate the first part of the log-likelihood
        part1 = 0;

        for ii in range( n ):
            part1 += multiTLogPDF( tppf_uhat[ii,:], np.zeros(p), self.P, self.par[0], p )

        # Calculate the second part of the log-likelihood
        part2      = np.sum( t.logpdf( tppf_uhat, self.par[0] ) );

        out = part2 - part1 - self.logPrior();

        print((par,out));
        return     out

    def evaluateLogLikelihoodHessian( self, par ):

        print( par )

        df          = par[0];
        self.par[1] = par[1];
        self.par[2] = par[2];
        self.par[3] = par[3];

        # Extract the degrees of freedom and the dimension
        p     = (self.uhat).shape[1];
        n     = (self.uhat).shape[0];

        self.constructCorrelationMatrix( p );

        # Compute the percentile function on univariate t
        tppf_uhat  = t.ppf( self.uhat, df );

        # Calculate the first part of the log-likelihood
        part1 = 0;

        for ii in range( n ):

            part1 += multiTLogPDF( tppf_uhat[ii,:], np.zeros(p), self.P, df, p )

        # Calculate the second part of the log-likelihood
        part2      = np.sum( t.logpdf( tppf_uhat, df ) );

        return     part1 - part2;

    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================
    def priorUniform(self):
        out = 1.0;

        if ( self.par[0] < 0.0 ):
            out = 0.0;

        for ii in range(self.nPar-1):
            if ( np.abs( self.par[ii+1] ) > 1.0 ):
                out = 0.0;

        return( out );

    def logPrior(self):
        prior  = 0.0;
        prior += gamma.logpdf(self.par[0],2.0,scale=6.0)
        prior += beta.logpdf(0.5*(self.par[1]+1.0),2.0,2.0)
        prior += beta.logpdf(0.5*(self.par[2]+1.0),2.0,2.0)
        prior += beta.logpdf(0.5*(self.par[3]+1.0),2.0,2.0)

        return(prior)

    # Standard operations on struct
    copyData                = template_copyData;
    storeParameters         = template_storeParameters;
    returnParameters        = template_returnParameters

    # No tranformations available
    transform               = empty_transform;
    invTransform            = empty_invTransform;
    Jacobian                = empty_Jacobian;