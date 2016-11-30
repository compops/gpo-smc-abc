import numpy as np
import scipy.interpolate as interpolate

def pobs( data ):
    return (1.0 + np.argsort( data ).astype(float) ) / float( len( data ) + 2.0)


def inv_pobs( data, grid, nBins ):
    # Code adapted from:
    # http://www.nehalemlabs.net/prototype/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
    
    # Construct the pdf
    hist, bin_edges  = np.histogram( data, bins=nBins, density=True )
    cum_values       = np.zeros( bin_edges.shape )
    cum_values[1:]   = np.cumsum( hist * np.diff( bin_edges ) )
    
    # Fix boundaries
    bin_edges[0]     = bin_edges[0]  - 0.01
    bin_edges[nBins] = bin_edges[40] + 0.01
    
    # Interpolation
    inv_cdf          = interpolate.interp1d( cum_values, bin_edges )
    
    return inv_cdf( grid )

