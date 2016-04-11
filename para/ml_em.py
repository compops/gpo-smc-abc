##############################################################################
##############################################################################
# Routines for 
# Maximum-likelihood inference using optimisation
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy      as     np
from   ml_helpers import *

##############################################################################
# Main class
##############################################################################

class stMLEMopt(object):

    ##########################################################################
    # Initalisation
    ##########################################################################
    
    # Should the algorithm write out its status during each iteration
    verbose                         = None;
    
    # Variables for constructing the file name when writing the output to file
    filePrefix                      = None;
    dataset                         = None;
    
    # Exit conditions (the diff in norm of theta or maximum no iterations)
    tolLevel                        = None;
    noisyTolLevel                   = None;
    maxIter                         = None;
    
    # Adapt step sizes as kk**(-\alpha)
    stochasticApproximation         = None;
    stochasticApproximationFrom     = None;
    alpha                           = None;
    
    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval  = None;
    writeOutProgressToFile          = None;
    fileOutName                     = None;
        
    # Give inital parameters and particle trajectory
    initPar                         = None;
    initCondPath                    = None;
    
    ##########################################################################
    # Wrapper for standard EM
    ##########################################################################
    def em(self,sm,sys,thSys):
        self.optMethod  = "em"
        self.em_base(sm,sys,thSys);
    
    ##########################################################################
    # Wrapper for stochastic-approximation EM
    ##########################################################################
    def psaem(self,sm,sys,thSys):
        self.optMethod = "psaem"
        self.em_base(sm,sys,thSys);
        
    
    ##########################################################################
    # Wrapper for write to file
    ##########################################################################
    def writeToFile(self,sm,fileOutName=None):
        writeToFile_helper(self,sm,fileOutName);
        
    ##########################################################################
    # Main routine for direct optimisation
    ##########################################################################
    def em_base(self,sm,sys,thSys):
        
        #=====================================================================
        # Initalisation
        #=====================================================================
        sm.calcQFlag        = True;
        sm.calcGradientFlag = False;
        sm.calcHessianFlag  = False;

        self.nPars      = thSys.nParInference;  
        self.filePrefix = thSys.filePrefix;
        self.T          = thSys.T;
        runNextIter     = True;

        # Check algorithm settings and set to default if needed
        setSettings(self,"em");
        
        # Allocate vectors
        self.ll             = np.zeros((self.maxIter,1))
        self.th             = np.zeros((self.maxIter,thSys.nParInference))
        self.llDiff         = np.zeros((self.maxIter,1))    
        self.thDiff         = np.zeros((self.maxIter,1))    
        self.step           = np.zeros((self.maxIter,1))
        self.iter      = 1;

        # Store the initial parameters
        thSys.storeParameters( self.initPar, sys );
        self.th[0,:]  = thSys.returnParameters();
        thSys.transform();
        
        # Compute the log-likelihood and conditional path
        if ( self.optMethod == "psaem" ):
            sm.condPath = self.initCondPath
        
        sm.filter(thSys);
        self.ll[0]            = sm.ll;
        
        if ( self.optMethod == "psaem" ):
            sm.reconstructTrajectories(sys);
            
            if ( sm.filterType == "bPF" ):
                nIdx        = np.random.choice(sm.nPart, 1, p=sm.w[:,sys.T-1] );
                sm.condPath = sm.x[nIdx,:];
            elif ( sm.filterType == "faPF" ):
                nIdx        = np.random.choice(sm.nPart, 1, p=sm.w[:,sys.T-2] );
                sm.condPath = sm.x[nIdx,:];
        
        #############################################################################################################################
        # Main loop
        #############################################################################################################################              
        while ( runNextIter ):

            if ( self.optMethod == "psaem" ):
                # Adapt step size
                if ( ( self.stochasticApproximation ) & ( self.iter > self.stochasticApproximationFrom ) ):
                    self.step[self.iter] = ( self.iter - self.stochasticApproximationFrom )**(-self.alpha);
                else:
                    self.step[self.iter] = 0.0;
                
                if ( self.stochasticApproximation == False ):
                    self.step[self.iter] = 0.0;           
                
                # Copy current Q-function estimate to the sufficient statistics vector
                if ( ( self.stochasticApproximation ) & ( self.iter == self.stochasticApproximationFrom+1 ) ):                    
                    thSys.suff = np.array(sm.qfunc, copy=True);
            else:
                self.step[self.iter] = 0.0;
            
            # E-step: Compute the particle system and compute the components of the Q-function
            sm.smoother(thSys);
            self.ll[ self.iter ] = sm.ll;
            
            # M-step: Update the estimates of the parameters
            self.th[self.iter,:]     = thSys.Mstep( sm, self.step[self.iter] );
            self.thDiff[ self.iter ] = np.linalg.norm( self.th[self.iter,:] - self.th[self.iter-1,:], 2)
            
            # Store the new parameters            
            thSys.storeParameters( self.th[self.iter,:], sys );
            thSys.transform();
            
            # Sample a trajectory if using conditional PF
            if ( self.optMethod == "psaem" ):
                sm.reconstructTrajectories(sys);
                
                if ( sm.filterType == "bPF" ):
                    nIdx        = np.random.choice(sm.nPart, 1, p=sm.w[:,sys.T-1] );
                    sm.condPath = sm.x[nIdx,:];
                elif ( sm.filterType == "faPF" ):
                    nIdx        = np.random.choice(sm.nPart, 1, p=sm.w[:,sys.T-2] );
                    sm.condPath = sm.x[nIdx,:];

            # Calculate the difference in log-likelihood and check exit condition
            self.llDiff[self.iter] = np.abs( self.ll[self.iter] - self.ll[self.iter-1] );
            
            if (self.noisyTolLevel == False):
                if ( self.llDiff[self.iter] < self.tolLevel ):
                    runNextIter = False;
                if ( self.thDiff[self.iter] < self.tolLevel ):
                    runNextIter = False;                
            else:
                if ( self.iter > self.noisyTolLevel ):
                    # Check if the condition has been fulfilled during the last iterations
                    if (np.sum( self.llDiff[range( int(self.iter-(self.noisyTolLevel-1)),self.iter+1)] < self.tolLevel ) == self.noisyTolLevel):
                        runNextIter = False;
                    
                    if (np.sum( self.thDiff[range( int(self.iter-(self.noisyTolLevel-1)),self.iter+1)] < self.tolLevel ) == self.noisyTolLevel):
                        runNextIter = False;            
                
            # Update iteration number and check exit condition
            self.iter += 1;
              
            if ( self.iter == self.maxIter ):
                runNextIter = False;
            
            # Write out progress at some intervals
            if ( self.writeOutProgressToFile ):
                if np.remainder(self.iter,self.writeOutProgressToFileInterval) == 0:
                    self.writeToFile(sm);
            
            # Print output to console
            if (self.verbose ):
                parm = ["%.4f" % v for v in self.th[self.iter-1]];
                print("Iteration: " + str(self.iter) + " with current parameters: " + str(parm) + " and lldiff: " + str(self.llDiff[self.iter-1]) ) 
        
        #=====================================================================        
        # Compile output
        #=====================================================================        
        tmp         = range(0,self.iter-1);
        self.th     = self.th[tmp,:];
        self.step   = self.step[tmp]
        self.llDiff = self.llDiff[tmp];
        self.ll     = self.ll[tmp]    

    
##############################################################################
##############################################################################
# End of file
##############################################################################   
##############################################################################
        