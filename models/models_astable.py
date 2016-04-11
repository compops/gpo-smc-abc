from scipy.interpolate import griddata
import numpy as np

def rstable0(alpha, beta, gamma, delta, shape=()):
        
    r1 = np.random.random(shape)
    r2 = np.random.random(shape)
    
    # Uniform random variable in [-pi/2,pi/2]
    u = np.pi * ( r1 - 0.5 );
    
    # Exponential random variable with rate 1
    w = - np.log( r2 );
    
    # Precompute S
    S = beta * np.tan(np.pi * alpha / 2.0 );
    S = 1 + S**2;
    S = S ** ( -1 / (2 * alpha) );
    
    # Precompute B
    B = beta * np.tan ( np.pi * alpha / 2.0 );
    B = ( 1 / alpha ) * np.arctan( B );
    
    if alpha==1.0:
        t1 = ( ( np.pi / 2.0 ) + beta * u ) * np.tan( u );
        t2 = beta * np.log ( (( np.pi / 2.0 ) * w * np.cos( u )) / ( ( np.pi / 2.0 ) + beta * u ) );
        t  = ( 2.0 / np.pi ) * (t1 - t2);
        return gamma * t + delta
    else:
        t1 = np.cos(u)**( alpha / 2.0 );
        t1 = np.sin( alpha * ( u + B ) ) / t1;
        t2 = np.cos( u - alpha * ( u + B ) ) / w;
        t2 = t2**( ( 1.0 - alpha ) / alpha )
        t = S * t1 * t2;
        return gamma * (t - beta * np.tan( np.pi * alpha / 2.0 ) ) + delta

def rstable1(alpha, beta, gamma, delta, shape=()):
        
    r1 = np.random.random(shape)
    r2 = np.random.random(shape)
    
    # Uniform random variable in [-pi/2,pi/2]
    u = np.pi * ( r1 - 0.5 );
    
    # Exponential random variable with rate 1
    w = - np.log( r2 );
    
    # Precompute T = th0
    T = beta * np.tan( np.pi * alpha / 2.0 );
    T = np.arctan(T) / alpha;
    
    if alpha==1.0:
        t1 = ( ( np.pi / 2.0 ) + beta * u ) * np.tan( u );
        t2 = beta * np.log ( (( np.pi / 2.0 ) * w * np.cos( u )) / ( ( np.pi / 2.0 ) + beta * u ) );
        t  = ( 2.0 / np.pi ) * (t1 - t2);
        return gamma * t + ( delta + beta * 2.0 / np.pi * gamma * np.log(gamma) )
    else:
        t1 = ( np.cos(alpha * T) * np.cos(u) )**( 1.0 / alpha );
        t1 = np.sin( alpha * ( u + T ) ) / t1;
        t2 = np.cos( alpha * T + (alpha - 1.0) * u ) / w;
        t2 = t2**( ( 1.0 - alpha ) / alpha )
        t = t1 * t2;
        
        out = (gamma * t + delta, u, w)
        return out
     

def mcculloch(data):
    q95 = np.percentile(data,95);
    q05 = np.percentile(data,5);
    q75 = np.percentile(data,75);
    q25 = np.percentile(data,25);
    q50 = np.percentile(data,50);
    
    va  = ( q95 - q05 ) / ( q75 - q25 );
    vb  = ( q95 + q05 - 2 * q50 ) / ( q95 - q05 );
    vg  = q75 - q25;
    
    xi_1 = (2,     2,     2,     2,     2,     2,     2, 1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924, 1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829, 1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745, 1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676, 1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547, 1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438, 1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318, 1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150, 1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973, 1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874, 0.896, 0.892, 0.884, 0.883, 0.855, 0.823, 0.769, 0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691, 0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.597, 0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513);
    xi_2 = (0, 2.160, 1,     1,     1,     1,     1, 0, 1.592, 3.390, 1,     1,     1,     1, 0, 0.759, 1.800, 1,     1,     1,     1, 0, 0.482, 1.048, 1.694, 1,     1,     1, 0, 0.360, 0.760, 1.232, 2.229, 1,     1, 0, 0.253, 0.518, 0.823, 1.575, 1,     1, 0, 0.203, 0.410, 0.632, 1.244, 1.906, 1, 0, 0.165, 0.332, 0.499, 0.943, 1.560, 1, 0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195, 0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917, 0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759, 0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596, 0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482, 0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362, 0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274);
    xi_3 = (1.908, 1.908, 1.908, 1.908, 1.908, 1.914, 1.915, 1.916, 1.918, 1.921, 1.921, 1.922, 1.927, 1.936, 1.947, 1.927, 1.930, 1.943, 1.961, 1.987, 1.933, 1.940, 1.962, 1.997, 2.043, 1.939, 1.952, 1.988, 2.045, 2.116, 1.946, 1.967, 2.022, 2.106, 2.211, 1.955, 1.984, 2.067, 2.188, 2.333, 1.965, 2.007, 2.125, 2.294, 2.491, 1.980, 2.040, 2.205, 2.435, 2.696, 2,     2.085, 2.311, 2.624, 2.973, 2.040, 2.149, 2.461, 2.886, 3.356, 2.098, 2.244, 2.676, 3.265, 3.912, 2.189, 2.392, 3.004, 3.844, 4.775, 2.337, 2.634, 3.542, 4.808, 6.247, 2.588, 3.073, 4.534, 6.636, 9.144);
    xi_4 = (0,  0,      0,      0,      0, 0, -0.017, -0.032, -0.049, -0.064, 0, -0.030, -0.061, -0.092, -0.123, 0, -0.043, -0.088, -0.132, -0.179, 0, -0.056, -0.111, -0.170, -0.232, 0, -0.066, -0.134, -0.206, -0.283, 0, -0.075, -0.154, -0.241, -0.335, 0, -0.084, -0.173, -0.276, -0.390, 0, -0.090, -0.192, -0.310, -0.447, 0, -0.095, -0.208, -0.346, -0.508, 0, -0.098, -0.223, -0.380, -0.576, 0, -0.099, -0.237, -0.424, -0.652, 0, -0.096, -0.250, -0.469, -0.742, 0, -0.089, -0.262, -0.520, -0.853, 0, -0.078, -0.272, -0.581, -0.997, 0, -0.061, -0.279, -0.659, -1.198);
    
    tv_1 = [2.4, 2.5, 2.6, 2.7, 2.8, 3,   3.2, 3.5, 4, 5, 6, 8, 10, 15, 25];
    tv_2 = [0,   0.1, 0.2, 0.3, 0.5, 0.7, 1];
    
    t_1 = [2, 1.9,  1.8, 1.7,  1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5];
    t_2 = [0, 0.25, 0.5, 0.75, 1];       
    
    kk = 0;
    tv = np.zeros((len(tv_1)*len(tv_2),2))
    for ii in range(0,len(tv_1)):
        for jj in range(0,len(tv_2)):
            tv[kk,:] = (tv_1[ii],tv_2[jj]);
            kk += 1;
        
    
    kk = 0;
    tt = np.zeros((len(t_1)*len(t_2),2))
    for ii in range(0,len(t_1)):
        for jj in range(0,len(t_2)):
            tt[kk,:] = (t_1[ii],t_2[jj]);
            kk += 1;
        
    
    ahat = griddata(tv, xi_1, (va, vb), method='linear')
    bhat = griddata(tv, xi_2, (va, vb), method='linear')
    chat = griddata(tt, xi_3, (ahat, bhat), method='linear')
    dhat = griddata(tt, xi_4, (ahat, bhat), method='linear')
    
    chat = vg / chat
    dhat = q50 - chat * dhat;
    dhat = dhat - bhat * chat * np.tan( np.pi * ahat / 2.0 )
    
    return (va,vb,vg),(ahat,bhat,chat,dhat)


#dd = rstable0(1.1,0.5,1,0,(10000,1))
#dd = rstable1(1.8,1,1,0,(1000,1))
#uu,that = mcculloch(dd)