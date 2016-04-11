##############################################################################
##############################################################################
# Routines for
# Kalman filtering and smoothing
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy as np

##############################################################################
# Main class
##############################################################################

class kalmanMethods(object):
    ##########################################################################
    # Initalisation
    ##########################################################################

    # Identifier
    typeSampler      = 'kalman';

    # Should the gradient and Hessian ( of log-target ) be calculated
    calcGradientFlag = True;
    calcHessianFlag  = True;
    calcQFlag        = False;

    # Initial state for the particles
    Po               = None;
    xo               = None;

    ##########################################################################
    # Kalman filter
    ##########################################################################
    def kf(self,sys):

        #=====================================================================
        # Initialisation
        #=====================================================================

        # Check settings and apply defaults otherwise
        if ( self.xo == None ):
            self.xo = 0.0;
            print("kf: no initial state given assuming x0: " + str(self.xo) + ".");

        if ( self.Po == None ):
            self.Po = 1e-5;
            print("kf: no initial covariance given assuming P0: " + str(self.Po) + ".");

        self.filterType = "kf";

        # Initialise variables for the filter
        S       = np.zeros((sys.T,1));
        K       = np.zeros((sys.T,));
        xhatp   = np.zeros((sys.T+1,1));
        xhatf   = np.zeros((sys.T,1));
        yhatp   = np.zeros((sys.T,1));
        Pf      = np.zeros((sys.T,1));
        Pp      = np.zeros((sys.T+1,1));
        ll      = np.zeros(sys.T);

        # Check if the mean should be in the model or not
        if (sys.nPar == 4):
            # With mean (4 variables)
            self.m  = sys.par[0];
            self.A  = sys.par[1];
            self.C  = 1.0;
            self.Q  = sys.par[2]**2;
            self.q  = sys.par[2];
            self.R  = sys.par[3]**2;
            self.r  = sys.par[3];
        elif (sys.nPar == 3):
            # No mean (3 variables)
            self.m  = 0.0;
            self.A  = sys.par[0];
            self.C  = 1.0;
            self.Q  = sys.par[1]**2;
            self.q  = sys.par[1];
            self.R  = sys.par[2]**2;
            self.r  = sys.par[2];
        else:
            raise NameError("Kalman: unknown model used, check parametrisation.");

        # Set initial covariance and state
        Pp[0]     = self.Po;
        xhatp[0]  = self.xo;

        #=====================================================================
        # Run main loop
        #=====================================================================

        for tt in range(0, sys.T):

            # Calculate the Kalman Gain
            S[tt] = self.C * Pp[tt] * self.C + self.R;
            K[tt] = Pp[tt] * self.C / S[tt];

            # Compute the state estimate
            yhatp[tt]   = self.C * xhatp[tt];
            xhatf[tt]   = xhatp[tt] + K[tt] * ( sys.y[tt] - yhatp[tt] );
            xhatp[tt+1] = self.A * xhatf[tt] + self.m * ( 1.0 - self.A ) + sys.u[tt];

            # Update covariance
            Pf[tt]      = Pp[tt] - K[tt] * S[tt] * K[tt];
            Pp[tt+1]    = self.A * Pf[tt] * self.A + self.Q;

            # Estimate loglikelihood
            ll[tt]      = -0.5 * np.log(2.0 * np.pi * S[tt]) - 0.5 * ( sys.y[tt] - yhatp[tt] ) * ( sys.y[tt] - yhatp[tt] ) / S[tt];

        #=====================================================================
        # Compile output
        #=====================================================================

        self.ll    = np.sum(ll);
        self.llt   = ll;
        self.xhatf = xhatf;
        self.xtraj = xhatf;
        self.xhatp = xhatp;
        self.K     = K;
        self.Pp    = Pp;
        self.Pf    = Pf;

    ##########################################################################
    # RTS smoother
    ##########################################################################

    def rts(self,sys):

        #=====================================================================
        # Initialisation
        #=====================================================================
        self.smootherType    = "rts"

        # Run the preliminary Kalman filter
        self.kf(sys);

        # Initalise variables
        J       = np.zeros((sys.T,1));
        M       = np.zeros((sys.T,1));
        xhats   = np.zeros((sys.T,1));
        Ps      = np.zeros((sys.T,1));

        # Set last smoothing covariance and state estimate to the filter solutions
        Ps[sys.T-1]     = self.Pf[sys.T-1];
        xhats[sys.T-1]  = self.xhatf[sys.T-1];

        #=====================================================================
        # Run main loop
        #=====================================================================

        for tt in range((sys.T-2),0,-1):
            J[tt]       = self.Pf[tt] * self.A / self.Pp[tt+1]
            xhats[tt]   = self.xhatf[tt] + J[tt] * ( xhats[tt+1] - self.xhatp[tt+1] )
            Ps[tt]      = self.Pf[tt] + J[tt] * ( Ps[tt+1] - self.Pp[tt+1] ) * J[tt];

        #=====================================================================
        # Calculate the M-matrix (Smoothing covariance between states at t and t+1)
        #=====================================================================

        M[sys.T-1]  = ( 1 - self.K[sys.T-1] ) * self.A * self.Pf[sys.T-1];
        for tt in range((sys.T-2),0,-1):
            M[tt]   = self.Pf[tt] * J[tt-1] + J[tt-1] * ( M[tt+1] - self.A * self.Pf[tt] ) * J[tt-1];

        #=====================================================================
        # Gradient and Hessian estimation
        #=====================================================================

        if (self.calcGradientFlag != False ):
            gradient = np.zeros((4,sys.T));

            for tt in range(1,sys.T):
                kappa = xhats[tt]   * sys.y[tt];
                eta   = xhats[tt]   * xhats[tt]   + Ps[tt];
                eta1  = xhats[tt-1] * xhats[tt-1] + Ps[tt-1];
                psi   = xhats[tt-1] * xhats[tt]   + M[tt];

                px = xhats[tt] - self.m - self.A * ( xhats[tt-1] - self.m ) + sys.u[tt];
                Q1 = self.q**(-1)
                Q2 = self.q**(-2)
                Q3 = self.q**(-3)

                gradient[0,tt] = Q2 * px * ( 1.0 - self.A );
                gradient[1,tt] = Q2 * ( psi - self.m * xhats[tt-1] * ( 1.0 - self.A ) - self.A * eta1 ) - Q2 * self.m * px;
                gradient[2,tt] = Q3 * ( eta - 2.0 * self.A * psi + self.A**2 * eta1 - 2.0*(xhats[tt]-self.A*xhats[tt-1])*self.m*(1.0-self.A) + self.m**2 * (1.0-self.A)**2 ) - Q1;
                gradient[3,tt] = self.r**(-3) * ( sys.y[tt]**2 - 2 * kappa + eta ) - self.r**(-1);

            # Remove the gradient for mu if the 3 parameter model is used
            if (sys.nPar == 3):
                gradient = gradient[1:4,:];

            # Estimate the gradient
            gradient0 = np.sum(gradient[0:sys.nParInference,:], axis=1);

        if ( self.calcHessianFlag != False ):
            # Estimate the information matrix using the Segal and Weinstein estimator
            hessian = np.dot(np.mat(gradient[0:sys.nParInference,:]),np.mat(gradient[0:sys.nParInference,:]).transpose()) - np.dot(np.mat(gradient0).transpose(),np.mat(gradient0)) / sys.T;

        # Add the log-prior derivatives
        for nn in range(0,sys.nParInference):
            if (self.calcGradientFlag):
                gradient0[nn]     = sys.dprior1(nn) + gradient0[nn];
            for mm in range(0,sys.nParInference):
                if (self.calcHessianFlag):
                    hessian[nn,mm] = sys.ddprior1(nn,mm) + hessian[nn,mm];

        #=====================================================================
        # Q-function
        #=====================================================================

        if ( self.calcQFlag & (sys.nPar == 3) ):
            qfunc = np.zeros((4,sys.T));

            for tt in range(1,sys.T):
                neta  = xhats[tt]   * sys.u[tt-1];
                neta1 = xhats[tt-1] * sys.u[tt-1];
                eta   = xhats[tt]   * xhats[tt]   + Ps[tt];
                eta1  = xhats[tt-1] * xhats[tt-1] + Ps[tt-1];
                psi   = xhats[tt-1] * xhats[tt]   + M[tt];

                Q1 = self.q**(-1)
                Q2 = self.q**(-2)
                Q3 = self.q**(-3)

                qfunc[0,tt] = psi;
                qfunc[1,tt] = neta1;
                qfunc[2,tt] = eta1;
                qfunc[3,tt] = self.A**2*eta1 + 2*self.A*neta1 - 2*self.A*psi + sys.u[tt-1]**2 - 2.0*neta + eta

            # Estimate the q-function
            qfunc0 = np.sum(qfunc, axis=1);

        #=====================================================================
        # Compile output
        #=====================================================================

        self.Ps        = Ps;
        self.xhats     = xhats;

        if (self.calcGradientFlag):
            self.gradient  = gradient0[0:sys.nParInference];
            self.gradient1 = gradient[0:sys.nParInference,:];

        if (self.calcHessianFlag):
            self.hessian   = hessian;

        if ( self.calcQFlag & (sys.nPar == 3) ):
            self.qfunc  = qfunc0;
            self.qfunc0 = qfunc;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
