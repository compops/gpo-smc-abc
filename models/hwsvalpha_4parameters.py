##############################################################################
##############################################################################
# Model specification
# Stochastic volatility model with alpha-stable returns
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

#=============================================================================
# Model structure
#=============================================================================
# xtt = par[0] + par[1] * ( xt  - par[0] ) + par[2] * vt,
# yt  = exp( 0.5 * xt )* et,
#
# vt  ~ N(0,1)
# e(t) ~ A(par[3],0,1,0)

import numpy          as np
from   models_astable import rstable1
from   models_helpers import *
from   models_dists   import *
from   scipy.stats    import norm

class ssm(object):

    #=========================================================================
    # Define model settings
    #=========================================================================
    nPar         = 4;
    par          = np.zeros(nPar);
    modelName    = "Hull-White Stochastic Volatility model with alpha-stable returns"
    filePrefix   = "hwsvalpha";
    supportsFA   = False;
    nParInfernce = None;
    nQInference  = None;
    transformY   = "none"
    version      = "standard"

    #=========================================================================
    # Define Jacobian and parameter transforms
    #=========================================================================
    def Jacobian( self ):
        if (self.version == "tanhexp"):
            if ( self.nParInference >= 2 ):
                return np.log( 1.0 - self.par[1]**2 );

            if ( self.nParInference >= 3 ):
                return np.log( 1.0 - self.par[1]**2 ) + np.log( self.par[2] );

            if ( self.nParInference >= 4 ):
                return np.log( 1.0 - self.par[1]**2 ) + np.log( self.par[2] ) + np.log( 2.0 * self.par[3] - self.par[3]**2 );
        else:
            return 0.0;

    def transform(self):
        if (self.version == "tanhexp"):
            if ( self.nParInference >= 2 ):
                self.par[1] = np.tanh ( self.par[1] );

            if ( self.nParInference >= 3 ):
                self.par[2] = np.exp  ( self.par[2] );

            if ( self.nParInference >= 4 ):
                self.par[3] = np.tanh ( self.par[3] ) + 1.0;
        return None;

    def invTransform(self):
        if (self.version == "tanhexp"):
            if ( self.nParInference >= 2 ):
                self.par[1] = np.arctanh( self.par[1] );

            if ( self.nParInference >= 3 ):
                self.par[2] = np.log    ( self.par[2] );

            if ( self.nParInference >= 4 ):
                self.par[3] = np.arctanh( self.par[3] - 1.0 );
        return None;

    #=========================================================================
    # Define the model
    #=========================================================================
    def generateInitialState( self, nPart ):
        return self.par[0] + np.random.normal(size=(1,nPart)) * self.par[2] / np.sqrt( 1 - self.par[1]**2 );

    def evaluateState(self, xtt, xt, tt):
        return norm.pdf( xtt, self.par[0] + self.par[1] * ( xt - self.par[0] ), self.par[2] );

    def generateState(self, xt, tt):
        return self.par[0] + self.par[1] * ( xt - self.par[0] ) + self.par[2] * np.random.randn(1,len(xt));

    def generateObservation(self, xt, tt):
        (w, v1, v2) = np.exp(0.5*xt) * rstable1( self.par[3], 0.0, 1.0, 0, (1,len(xt)) );

        if ( self.transformY == "arctan" ):
            w = np.arctan( w );
        elif ( self.transformY == "none" ):
            None;
        else:
            raise NameError("makeNoisy: unknown transformation of data.")

        out = (w, v1, v2)
        return out;

    #=========================================================================
    # Define gradients of logarithm of complete data-likelihood
    #=========================================================================
    def Dparm(self, xtt, xt, pu, pw, tt):

        nOut = len(xtt);
        gradient = np.zeros(( nOut, self.nParInference ));
        Q1 = self.par[2]**(-1);
        Q2 = self.par[2]**(-2);
        Q3 = self.par[2]**(-3);
        px = xtt - self.par[0] - self.par[1] * ( xt - self.par[0] );
        sigma = np.exp( 0.5 * xt )

        if ( self.version == "standard" ):
            for v1 in range(0,self.nParInference):
                if v1 == 0:
                    gradient[:,v1] = ( 1.0 - self.par[1] ) * Q2 * px;
                elif v1 == 1:
                    gradient[:,v1] = ( xt - self.par[0] ) * Q2 * px;
                elif v1 == 2:
                    gradient[:,v1] = Q3 * px**2 - Q1
                elif v1 == 3:
                    # With Atan-transform
                    gradient[:,v1] =1.0*sigma*(np.cos(pu*(self.par[3] - 1))/pw)**(-1 + 1/self.par[3])*(self.y[tt] - np.arctan(sigma*(np.cos(pu*(self.par[3] - 1))/pw)**(-1 + 1/self.par[3])*np.sin(self.par[3]*pu)*np.cos(pu)**(-1/self.par[3])))*(self.par[3]**2*pu*np.cos(pu) - self.par[3]*pu*np.sin(self.par[3]*pu)*np.sin(pu*(self.par[3] - 1)) - np.log(np.cos(pu*(self.par[3] - 1))/pw)*np.sin(self.par[3]*pu)*np.cos(pu*(self.par[3] - 1)) + np.log(np.cos(pu))*np.sin(self.par[3]*pu)*np.cos(pu*(self.par[3] - 1)))*np.cos(pu)**(1/self.par[3])/(self.par[3]**2*self.epsilon**2*(sigma**2*(np.cos(pu*(self.par[3] - 1))/pw)**(-2 + 2/self.par[3])*np.sin(self.par[3]*pu)**2 + np.cos(pu)**(2/self.par[3]))*np.cos(pu*(self.par[3] - 1)))
                    #sigma*(np.cos(pu*(alpha - 1))/pw)**(-1 + 1/alpha)*(self.y[tt] - np.arctan(sigma*(np.cos(pu*(alpha - 1))/pw)**(-1 + 1/alpha)*np.sin(alpha*pu)*np.cos(pu)**(-1/alpha)))*(alpha**2*pu*np.cos(pu) - alpha*pu*np.sin(alpha*pu)*np.sin(pu*(alpha - 1)) - np.log(np.cos(pu*(alpha - 1))/pw)*np.sin(alpha*pu)*np.cos(pu*(alpha - 1)) + np.log(np.cos(pu))*np.sin(alpha*pu)*np.cos(pu*(alpha - 1)))*np.cos(pu)**(1/alpha)/(alpha**2*self.epsilon**2*(sigma**2*(np.cos(pu*(alpha - 1))/pw)**(-2 + 2/alpha)*np.sin(alpha*pu)**2 + np.cos(pu)**(2/alpha))*np.cos(pu*(alpha - 1)))
                else:
                    gradient[:,v1] = 0.0;

        if ( self.version == "tanhexp" ):
            for v1 in range(0,self.nParInference):
                if v1 == 0:
                    gradient[:,v1] = ( 1.0 - self.par[1] ) * Q2 * px;
                elif v1 == 1:
                    gradient[:,v1] = ( xt - self.par[0] ) * Q2 * px * ( 1.0 - self.par[1]**2 );
                elif v1 == 2:
                    gradient[:,v1] = Q2 * px**2 - 1.0
                elif v1 == 3:
                    yt  = self.y[tt];
                    eps = self.epsilon
                    th3 = np.arctanh( self.par[3] );
                    gradient[:,v1] = -1.0*sigma*(np.cos(pu*th3)/pw)**(th3/(th3 + 1.0))*(yt - np.arctan(sigma*(np.cos(pu*th3)/pw)**(-th3/(th3 + 1.0))*np.sin(pu*(th3 + 1.0))*np.cos(pu)**(-1/(th3 + 1.0))))*(pu*(th3 + 1.0)**2*(th3**2 - 1)*np.cos(pu*(th3 + 1.0))*np.cos(pu*th3) + (pu*(th3 + 1.0)*(th3**2 - 1)*np.sin(pu*th3)*th3 - (1.0*th3**2 - 1.0)*np.log(np.cos(pu*th3)/pw)*np.cos(pu*th3))*np.sin(pu*(th3 + 1.0)) + (th3**2 - 1)*np.log(np.cos(pu))*np.sin(pu*(th3 + 1.0))*np.cos(pu*th3))*np.cos(pu)**(1/(th3 + 1.0))/(eps**2*(sigma**2*np.sin(pu*(th3 + 1.0))**2 + (np.cos(pu*th3)/pw)**(2*th3/(th3 + 1.0))*np.cos(pu)**(2/(th3 + 1.0)))*(th3 + 1.0)**2*np.cos(pu*th3));
                else:
                    gradient[:,v1] = 0.0;

        return(gradient);

    #=========================================================================
    # Define Hessians of logarithm of complete data-likelihood
    #=========================================================================
    def DDparm(self, xtt, xt, pu, pw, tt):

        nOut = len(xtt);
        hessian = np.zeros( (nOut, self.nParInference,self.nParInference) );
        return(hessian);

    #=============================================================================
    # Generate or load data from file
    #=============================================================================
    def generateData(self,u=None,fileName=None,order=None):

        # Set input to zero if not given
        if ( u==None ):
            u = np.zeros(self.T);

        x    = np.zeros((self.T+1,1));
        y    = np.zeros((self.T,1));
        e1   = np.zeros(self.T);
        e2   = np.zeros(self.T);
        x[0] = self.xo;

        if (fileName == None):
            for tt in range(0, self.T):
                (y[tt], e1[tt], e2[tt])   = self.generateObservation( x[tt], tt);
                x[tt+1]                   = self.generateState(x[tt], tt);

            self.x  = x[0:self.T]
            self.y  = y;
            self.ynoiseless = np.copy( self.y );
            self.u  = u;
            self.e1 = e1;
            self.e2 = e2;
        else:
            # Try to import data
            tmp   = np.loadtxt(fileName,delimiter=",")

            if ( order == None ):
                self.y          = np.array(tmp, copy=True).reshape((self.T,1));
                self.ynoiseless = np.copy( self.y );
                self.u          = u;
            elif ( order == "y" ):
                self.y          = np.array(tmp, copy=True).reshape((self.T,1));
                self.ynoiseless = np.copy( self.y );
                self.u          = u;
            elif ( order == "xy" ):
                self.x          = np.array(tmp[:,0], copy=True).reshape((self.T,1));
                self.y          = np.array(tmp[:,1], copy=True).reshape((self.T,1));
                self.ynoiseless = np.copy( self.y );
                self.u          = u;
            elif ( order == "xuy" ):
                self.x          = np.array(tmp[:,0], copy=True).reshape((self.T,1));
                self.u          = np.array(tmp[:,1], copy=True).reshape((self.T,1));
                self.y          = np.array(tmp[:,2], copy=True).reshape((self.T,1));
                self.ynoiseless = np.copy( self.y );
            else:
                raise NameError("generateData, import data: cannot import that order.");

    #=============================================================================
    # Make data noisy for the ABC procedure
    #=============================================================================
    def makeNoisy(self, sm ):
        if ( self.transformY == "arctan" ):
            if sm.weightdist == "boxcar":
                self.y = np.arctan(self.ynoiseless) + np.random.uniform(-sm.tolLevel,sm.tolLevel,(self.T,1));
            elif sm.weightdist == "gaussian":
                self.y = np.arctan(self.ynoiseless) + sm.tolLevel * np.random.randn(self.T,1);
        elif ( self.transformY == "none" ):
            if sm.weightdist == "boxcar":
                self.y = self.ynoiseless + np.random.uniform(-sm.tolLevel,sm.tolLevel,(self.T,1));
            elif sm.weightdist == "gaussian":
                self.y = self.ynoiseless + sm.tolLevel * np.random.randn(self.T,1);
        else:
            raise NameError("makeNoisy: unknown transformation of data.")

    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================
    def priorUniform(self):
        out = 1.0;

        if ( self.version == "standard" ):
            if( np.abs( self.par[1] ) > 1.0 ):
                out = 0.0;

            if( self.par[2] < 0.0 ):
                out = 0.0;

            if( self.par[3] > 2.0 ):
                out = 0.0;

            if( self.par[3] < 0.0 ):
                out = 0.0;

        return( out );

    #=========================================================================
    # Define log-priors for the PMH sampler
    #=========================================================================
    def prior(self):
        out = 0.0;

        # Normal prior for mu
        if ( self.nParInference >= 1 ):
            out += normalLogPDF( self.par[0], 0, 0.2 );

        # Truncated normal prior for phi (truncation by hard prior)
        if ( self.nParInference >= 2 ):
            out += normalLogPDF( self.par[1], 0.9, 0.05 );

        # Gamma prior for sigma
        if ( self.nParInference >= 3 ):
            out += gammaLogPDF( self.par[2], a=2.0, b=1.0/20.0 );

        # Beta prior for alpha/2
        if ( self.nParInference >= 4 ):
            #out += betaLogPDF( self.par[3]/2.0, a=6.0, b=2.0 )
            out += betaLogPDF( self.par[3]/2.0, a=20.0, b=2.0 )

        return out;

    #=========================================================================
    # Define gradients of log-priors for the PMH sampler
    #=========================================================================
    def dprior1(self,v1):

        if ( v1 == 0 ):
            # Normal prior for mu
            return normalLogPDFgradient( self.par[0], 0, 0.2 );
        elif ( v1 == 1):
            # Truncated normal prior for phi (truncation by hard prior)
            return normalLogPDFgradient( self.par[1], 0.9, 0.05 );
        elif ( v1 == 2):
            # Gamma prior for sigma
            return gammaLogPDFgradient( self.par[2], a=2.0, b=1.0/20.0 );
        elif ( v1 == 3):
            # Beta prior for alpha/2
            #return betaLogPDFgradient( self.par[3]/2.0, a=6.0, b=2.0 )
            return betaLogPDFgradient( self.par[3]/2.0, a=20.0, b=2.0 )
        else:
            return 0.0;

    #=========================================================================
    # Define hessians of log-priors for the PMH sampler
    #=========================================================================
    def ddprior1(self,v1,v2):

        if ( ( v1 == 0 ) & ( v1 == 0 ) ):
            # Normal prior for mu
            return normalLogPDFhessian( self.par[0], 0, 0.2 );
        elif ( ( v1 == 1 ) & ( v1 == 1 ) ):
            # Truncated normal prior for phi (truncation by hard prior)
            return normalLogPDFhessian( self.par[1], 0.9, 0.05 );
        elif ( ( v1 == 2 ) & ( v1 == 2 ) ):
            # Gamma prior for sigma
            return gammaLogPDFhessian( self.par[2], a=2.0, b=1.0/20.0 );
        elif ( ( v1 == 3 ) & ( v1 == 3 ) ):
            # Beta prior for alpha/2
            #return betaLogPDFhessian( self.par[3]/2.0, a=6.0, b=2.0 )
            return betaLogPDFhessian( self.par[3]/2.0, a=20.0, b=2.0 )
        else:
            return 0.0;

    #=========================================================================
    # Sample from the prior
    #=========================================================================
    def samplePrior(self):

        out = np.zeros( self.nParInference )

        # Normal prior for mu
        if ( self.nParInference >= 1 ):
            out[0] = np.random.normal( 0.0, 0.2 );

        # Truncated normal prior for phi (truncation by hard prior)
        if ( self.nParInference >= 2 ):
            uu = 1.2;
            while (uu > 1.0):
                uu = np.random.normal( 0.9, 0.05 );

            out[1] = uu;

        # Gamma prior for sigma
        if ( self.nParInference >= 3 ):
            out[2] = np.random.gamma( shape=2.0, scale=1.0/20.0 );

        # Beta prior for alpha
        if ( self.nParInference >= 4 ):
            #out[3] = 2.0 * np.random.beta( a=6.0, b=2.0 );
            out[3] = 2.0 * np.random.beta( a=20.0, b=2.0 );

        return out;

    #=============================================================================
    # Copy data from an instance of this struct to another
    #=============================================================================
    def copyData(self,sys):
        self.T             = np.copy( sys.T );
        self.y             = np.copy( sys.y );
        self.u             = np.copy( sys.u );
        self.ynoiseless    = np.copy( sys.ynoiseless );
        self.nPar          = sys.nPar
        self.filePrefix    = sys.filePrefix
        self.transformY    = sys.transformY;

        # Check if nQInference and nParInference are set and use default otherwise
        if ( self.nQInference == None ):
            self.nQInference = np.int( 0 );
            print("model: assuming that Q-function should not be estimated for this model.");

        if ( self.nParInference == None ):
            self.nParInference = np.int( 2 );
            print("model: assuming that " + str(self.nParInference) + " parameters should be inferred.");

        # Copy parameters
        self.par = np.zeros(sys.nPar);
        for kk in range(0,sys.nPar):
            self.par[kk] = np.array(sys.par[kk], copy=True)

    #=========================================================================
    # Define standard methods for the model struct
    #=========================================================================

    # Standard operations on struct
    storeParameters         = template_storeParameters;
    returnParameters        = template_returnParameters

    # No faPF available for this model
    generateStateFA         = empty_generateStateFA;
    evaluateObservationFA   = empty_evaluateObservationFA;
    generateObservationFA   = empty_generateObservationFA;

    # No EM algorithm available for this model
    Qfunc                   = empty_Qfunc;
    Mstep                   = empty_Mstep;

