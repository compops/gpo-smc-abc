import numpy          as np
import numdifftools   as nd

from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.kernel_regression      import KernelReg
from scipy                                            import optimize
from scipy.stats                                      import genpareto

def evtECDF( quantiles, gpa, data, proportion=0.10 ):

    # Extract the proportion of data in the upper and lower tails and compute excess values
    nSamples   = np.int( np.floor( len(data) * proportion ) )
    idx        = np.argsort( np.array(data) );
    ehat1l     = data[idx[0:nSamples]] - data[idx[nSamples]]
    ehat1u     = data[idx[-nSamples:]] - data[idx[-nSamples]]
    lowerlimit = data[idx[nSamples]]
    upperlimit = data[idx[-nSamples]]

    ### Estimate the parameters and their uncertainty for the lower tail
    gpa.thres = -ehat1l;
    resLower = optimize.fmin_l_bfgs_b(gpa.loglikelihoodBFGS, np.array([0.5, 0.2]), approx_grad=1, bounds=((-1.0, 1.0), (0.0, 5.0)) )
    thLower = resLower[0]

    # Create the numdiff object and estimate the inverse negative Hessian
    hes  = nd.Hessian( gpa.loglikelihoodHessian, delta=.00001 )
    foo  = np.linalg.inv( - hes( resLower[0] ) )
    sdLower = np.sqrt( ( foo[0,0], foo[1,1] ) )

    ### Estimate the parameters and their uncertainty for the lower tail
    gpa.thres = ehat1u;
    resUpper = optimize.fmin_l_bfgs_b(gpa.loglikelihoodBFGS, np.array([0.5, 0.2]), approx_grad=1, bounds=((-1.0, 1.0), (0.0, 5.0)) )
    thUpper = resUpper[0]

    # Create the numdiff object and estimate the inverse negative Hessian
    hes  = nd.Hessian( gpa.loglikelihoodHessian, delta=.00001 )
    foo  = np.linalg.inv( - hes( resUpper[0] ) )
    sdUpper = np.sqrt( ( foo[0,0], foo[1,1] ) )

    ### Find the limits for the GPD tail models
#    if ( thLower[0] < 0.0 ):
#        grid3  = np.arange( -lowerlimit, -thLower[1]/thLower[0] + 0.01, 0.01)
#    else:
    grid3  = np.arange( -lowerlimit, -np.min(data) + 0.01, 0.01)

#    if ( thUpper[0] < 0.0 ):
#        grid2  = np.arange(  upperlimit, thUpper[1]/thUpper[0] + 0.01, 0.01)
#    else:
    grid2  = np.arange(  upperlimit,  np.max(data) + 0.01, 0.01)

    ### Compute the DF

    # Empricial approximation in the middle
    ecdf1 = ECDF( data[nSamples:-nSamples] )
    grid1  = np.arange( lowerlimit + 0.01, upperlimit - 0.01, 0.01)
    #plot(grid1,ecdf1(grid1))

    # Compute the CDF of the lower tail GDP
    outPDFLower = gpa.cdf(grid3,thLower[0],thLower[1])
    #plot(-grid3, 1.0-outPDFLower )

    # Compute the CDF of the upper tail GDP
    outPDFUpper = gpa.cdf(grid2,thUpper[0],thUpper[1])
    #plot(grid2, outPDFUpper )

    # Interpolate between the different CDFS
    from scipy.interpolate import interp1d
    x = np.hstack((np.fliplr([-grid3])[0],grid1,grid2))
    y = np.hstack((np.fliplr([1.0-outPDFLower])[0],ecdf1(grid1),outPDFUpper))
    f = interp1d(x, y, bounds_error = False, fill_value = 0.0 )

    #print((np.min(x),np.max(x),np.min(quantiles),np.max(quantiles)))
    return( f( quantiles ), thUpper, thLower, sdUpper, sdLower )

def evtECDF_kernel( quantiles, gpa, data, proportion=0.10 ):

    # Extract the proportion of data in the upper and lower tails and compute excess values
    nSamples   = np.int( np.floor( len(data) * proportion ) )
    idx        = np.argsort( np.array(data) );
    ehat1l     = data[ idx[0:nSamples] ] - data[ idx[ nSamples] ]
    ehat1u     = data[ idx[-nSamples:] ] - data[ idx[-nSamples] ]
    lowerlimit = data[ idx[nSamples]   ]
    upperlimit = data[ idx[-nSamples]  ]

    ### Estimate the parameters for the GPD
    thLower  = genpareto.fit( -ehat1l )
    thUpper  = genpareto.fit( ehat1u  )

    ## Wrappers for Hessian estimation
    def evalLLLower ( par ):
        return( np.sum( genpareto.logpdf( -ehat1l[:-1], par[0], par[1], par[2] ) ) )

    def evalLLUpper ( par ):
        return( np.sum( genpareto.logpdf( ehat1u[2:], par[0], par[1], par[2] ) ) )

    ## Estimate the standard deviations
    hes  = nd.Hessian( evalLLLower, delta=.00001 )
    foo  = np.linalg.inv( - hes( thLower ) )
    varLower = np.diag( foo )

    hes  = nd.Hessian( evalLLUpper, delta=.00001 )
    foo  = np.linalg.inv( - hes( thUpper ) )
    varUpper = np.diag( foo )

    ### Find the limits for the GPD tail models
    grid1  = np.arange( np.min(data), np.max(data), 0.01)
    grid2  = np.arange( -lowerlimit, -np.min(data) + 0.01, 0.01)
    grid3  = np.arange(  upperlimit,  np.max(data) + 0.01, 0.01)

    ### Compute the three parts of the CDF
    ecdf1       = ECDF( data )
    outPDFLower = genpareto.cdf( grid2, thLower[0], thLower[1], thLower[2] )
    outPDFUpper = genpareto.cdf( grid3, thUpper[0], thUpper[1], thUpper[2] )

    # Use kernel regression to stich together the three parts of the CDF
    x = np.hstack((np.fliplr([-grid2])[0],grid1,grid3))
    y = np.hstack((np.fliplr([1.0-outPDFLower])[0],ecdf1(grid1),outPDFUpper))

    f1  = KernelReg(y,x,'c')
    f2  = KernelReg(x,y,'c')
    print(f1)
    out = f1.fit( quantiles )[0]

    return( out, thUpper, thLower, varUpper, varLower, f1, f2 )