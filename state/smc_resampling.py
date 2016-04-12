##############################################################################
##############################################################################
# Routines for
# Resampling
# Version 2015-03-12
#
# Copyright (c) 2015 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy as np
import scipy.weave as weave

##############################################################################
# Resampling for SMC sampler: Multinomial
##############################################################################
def resampleMultinomial(w, N=0, u=None ):
    code = \
    """ py::list ret;
	for(int kk = 0; kk < N; kk++)  // For each particle
        {
            int jj = 0;

            while( ww(jj) < u(kk) && jj < H - 1)
            {
                jj++;
            }
            ret.append(jj);
        }
	return_val = ret;
    """
    H = len(w);
    if N==0:
        N = H;

    if ( u == None ):
        u  = np.random.uniform(0.0,1.0,N);
    else:
        u = float(u);

    ww = ( np.cumsum(w) / np.sum(w) ).astype(float);
    idx = weave.inline(code,['u','H','ww','N'], type_converters=weave.converters.blitz)
    return np.array( idx ).astype(int);

##############################################################################
# Resampling for SMC sampler: Stratified
##############################################################################
def resampleStratified( w, N=0, u=None ):
    code = \
    """ py::list ret;
	int jj = 0;
        for(int kk = 0; kk < N; kk++)
        {
            double uu  = ( u(kk) + kk ) / N;

            while( ww(jj) < uu && jj < H - 1)
            {
                jj++;
            }
            ret.append(jj);
        }
	return_val = ret;
    """
    H = len(w);
    if N==0:
        N = H;

    if ( u == None ):
        u   = ( np.random.uniform(0.0,1.0,(N,1)) ).astype(float);
    else:
        u = float(u);

    ww  = ( np.cumsum(w) / np.sum(w) ).astype(float);
    idx = weave.inline(code,['u','H','ww','N'], type_converters=weave.converters.blitz)
    return np.array( idx ).astype(int);

##############################################################################
# Resampling for SMC sampler: Systematic
##############################################################################
def resampleSystematic( w, N=0, u=None ):
    code = \
    """ py::list ret;
	int jj = 0;
        for(int kk = 0; kk < N; kk++)
        {
            double uu  = ( u + kk ) / N;

            while( ww(jj) < uu && jj < H - 1)
            {
                jj++;
            }
            ret.append(jj);
        }
	return_val = ret;
    """
    H = len(w);
    if N==0:
        N = H;

    if ( u == None ):
        u   = float( np.random.uniform() );
    else:
        u = float(u);

    ww  = ( np.cumsum(w) / np.sum(w) ).astype(float);
    idx = weave.inline(code,['u','H','ww','N'], type_converters=weave.converters.blitz )
    return np.array( idx ).astype(int);

########################################################################
# End of file
########################################################################