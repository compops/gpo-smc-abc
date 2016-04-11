##############################################################################
##############################################################################
# Routines for 
# Input design
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy        as     np

##############################################################################
# Main class
##############################################################################

class stInputDesign(object):

    ##########################################################################
    # Initalisation
    ##########################################################################
    
    # Input design
    inputDesignNoDataSets           = None;
    hessianMethod                   = None;

    ##########################################################################
    # Compute expected information matrix for psuedo-binary input
    ##########################################################################
    def pseduoBinary(self,sm,sys,thSys,parm):  
        
        if ( self.hessianMethod == "sampleCovariance" ):
            sm.calcGradientFlag = True;
            sm.calcHessianFlag  = False;
        elif ( self.hessianMethod == "segelweinstein" ):
            sm.calcGradientFlag = True;
            sm.calcHessianFlag  = "segelweinstein";
            self.inputDesignNoDataSets = 1;
        
        gradient = np.zeros( ( self.inputDesignNoDataSets, thSys.nParInference ) )
        hessian  = np.zeros( ( self.inputDesignNoDataSets, thSys.nParInference, thSys.nParInference ) )
        
        # Compute an input using the current parameters
        inputdata = 1 * ( parm[0] < np.random.uniform(size=sys.T) ) * np.sign( np.random.uniform(size=sys.T) - parm[1] )
        
        # For each data realisation
        for ii in range(self.inputDesignNoDataSets):
                
            # Create a data realisation
            thSys.generateData( inputdata )
            
            # Compute the gradient vector
            sm.smoother( thSys )
            
            if ( self.hessianMethod == "sampleCovariance" ):
                gradient[ii,:] = sm.gradient;
            elif ( self.hessianMethod == "segelweinstein" ):
                hessian[ii,:,:] = sm.hessian;
            
        # Estimate the expected information matrix using the sample covariance
        if ( self.hessianMethod == "sampleCovariance" ):
            out = np.log( np.linalg.det( np.cov(gradient.transpose()) ) )
        elif ( self.hessianMethod == "segelweinstein" ):
            out = np.log( np.linalg.det( sm.hessian ) );
        
        return out;

