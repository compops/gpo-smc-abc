import numpy as np
import scipy.weave as weave

#############################################################################################################################
# Resampling for SMC sampler: Continuous
#############################################################################################################################

# Algorithm from Pitt (2002)
def resampleCont( w , R=0, u=None):
    code = \
    """
    double ss = 0; int jj = 0;

    // Loop over the different regions
    for(int ii = 0; ii < (N-1); ii++)
    {
        // Update the cumulative weights
        if ( ii == 0 )
        {
            ws(ii)  = w(0) + w(1) * 0.5;
        }
        else if ( ii == N-2 )
        {
            ws(ii)  = 0.5 * w(N-2) + w(N-1);
        }
        else
        {
            ws(ii)  = 0.5 * ( w(ii) + w(ii+1) );
        }

        ss += ws(ii);

        //printf("%f\\n",ss);

        // Loop over particles such that the stratified rv is leq than the limit of the region
        while( u(jj) <= ss && jj < R )
        {
            // The particle in within that region so set new particle index
            r(jj)   = ii;

            // Update uniform to find where it crosses the region
            us(jj)  = ( u(jj) - ss + ws(ii) ) / ws(ii);

            // Add one to particle index and repeat
            jj++;
        }
    }
    """

    # Set the number of particles and draw a rv from U[0,1]
    N = int( len(w) );
    if R==0:
        R = N;

    if ( u == None ):
        u = np.random.uniform();
    else:
        u = float(u);

    u = ( ( np.arange(0,R,1) + float( u ) ) / float(R) ).astype(float);

    # Declear variables to hold the new particle indicies and the new uniforms
    r  = np.empty(R, dtype='int')
    us = np.empty(R, dtype='double')
    ws = np.empty(R, dtype='double')

    # Execute the C-code
    weave.inline(code,['u','N','w','r','us','ws','R'], type_converters=weave.converters.blitz)

    # Return new indicies and the new random numbers
    return (r,us);

# Algorithm from Malik and Pitt (2011)
#def resampleCont(w):
#    code = \
#    """
#    double s = 0; int jj = 0;
#
#    // Loop over the different regions
#    for( int ii = 0; ii < N; ii++ )
#    {
#        // Update the cumulative weights
#        s += w(ii);
#
#        //printf("%f\\n",s);
#
#        // Loop over particles such that the stratified rv is leq than the limit of the region
#        while( u(jj) <= s && jj <= N )
#        {
#            // The particle in within that region so set new particle index
#            r(jj)   = ii;
#
#            // Update uniform to find where it crosses the region
#            us(jj)  = ( u(jj) - s + w(ii) ) / w(ii);
#
#            // Add one to particle index and repeat
#            jj++;
#        }
#    }
#    """
#
#    # Set the number of particles and draw a rv from U[0,1]
#    N = int( len(w) );
#    u = ( ( np.arange(0,N+1,1) + float( np.random.uniform() ) ) / float(N) ).astype(float);
#
#    # Declear variables to hold the new particle indicies and the new uniforms
#    r  = np.empty(N, dtype='int')
#    us = np.empty(N, dtype='double')
#
#    # Execute the C-code
#    weave.inline(code,['u','N','w','r','us'], type_converters=weave.converters.blitz)
#
#    # Return new indicies and the new random numbers
#    return (r,us);
#

#############################################################################################################################
# Resampling for SMC sampler: Systematic
#############################################################################################################################
def resampleSystematicnwq( w, N=0 ):
    code = \
    """
    int jj = 0;

    for(int ii = 0; ii < R; ii++)  // For each input particle
        {
            while( u(jj) <= ww(ii) && jj < M )
            {
                //printf("%f\\n",u(jj));
                //printf("%f\\n",ww(ii));
                Ind(jj) = ii;
                jj++;
            }
        }
    """

    # Number of input particles
    R = len(w);

    # Number  of output particles
    if M==0:
        M = R;

    u  = ( ( np.arange(0,M,1) + np.random.uniform() ) / M ).astype(float);
    ww = ( np.cumsum(w) / np.sum(w) ).astype(float);

    Ind = np.empty(M, dtype='int')
    weave.inline(code,['u','R','M','Ind','ww'], type_converters=weave.converters.blitz)
    return Ind;

#############################################################################################################################
# Resampling for SMC sampler: Multinomial
#############################################################################################################################
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

#############################################################################################################################
# Resampling for SMC sampler: Multinomial
#############################################################################################################################
def resampleStratifiednwq(w, M=0 ):
    code = \
    """
    int jj = 0;

    for(int ii = 0; ii < R; ii++)  // For each input particle
        {
            while( u(jj) <= ww(ii) && jj < M )
            {
                //printf("%f\\n",u(jj));
                //printf("%f\\n",ww(ii));
                Ind(jj) = ii;
                jj++;
            }
        }
    """

    # Number of input particles
    R = len(w);

    # Number  of output particles
    if M==0:
        M = R;

    u  = ( ( np.arange(0,M,1) + np.random.uniform(0.0,1.0,(M,1)) ) / M ).astype(float);
    ww = ( np.cumsum(w) / np.sum(w) ).astype(float);

    Ind = np.empty(M, dtype='int')
    weave.inline(code,['u','R','M','Ind','ww'], type_converters=weave.converters.blitz)
    return Ind;

#############################################################################################################################
# Resampling for SMC sampler: Stratified
#############################################################################################################################
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

#############################################################################################################################
# Resampling for SMC sampler: Systematic
#############################################################################################################################
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
