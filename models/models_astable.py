##############################################################################
##############################################################################
# Helpers for alpha-stable distribution
#
#
# Copyright (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

import numpy as np

##############################################################################
# Simulate alpha-stable random variable with parameterisation 0
##############################################################################


def rstable0(alpha, beta, gamma, delta, shape=()):

    r1 = np.random.random(shape)
    r2 = np.random.random(shape)

    # Uniform random variable in [-pi/2,pi/2]
    u = np.pi * (r1 - 0.5)

    # Exponential random variable with rate 1
    w = - np.log(r2)

    # Precompute S
    S = beta * np.tan(np.pi * alpha / 2.0)
    S = 1 + S**2
    S = S ** (-1 / (2 * alpha))

    # Precompute B
    B = beta * np.tan(np.pi * alpha / 2.0)
    B = (1 / alpha) * np.arctan(B)

    if alpha == 1.0:
        t1 = ((np.pi / 2.0) + beta * u) * np.tan(u)
        t2 = beta * np.log(((np.pi / 2.0) * w * np.cos(u)) /
                           ((np.pi / 2.0) + beta * u))
        t = (2.0 / np.pi) * (t1 - t2)
        return gamma * t + delta
    else:
        t1 = np.cos(u)**(alpha / 2.0)
        t1 = np.sin(alpha * (u + B)) / t1
        t2 = np.cos(u - alpha * (u + B)) / w
        t2 = t2**((1.0 - alpha) / alpha)
        t = S * t1 * t2
        return gamma * (t - beta * np.tan(np.pi * alpha / 2.0)) + delta


##############################################################################
# Simulate alpha-stable random variable with parameterisation 1
##############################################################################

def rstable1(alpha, beta, gamma, delta, shape=()):

    r1 = np.random.random(shape)
    r2 = np.random.random(shape)

    # Uniform random variable in [-pi/2,pi/2]
    u = np.pi * (r1 - 0.5)

    # Exponential random variable with rate 1
    w = - np.log(r2)

    # Precompute T = th0
    T = beta * np.tan(np.pi * alpha / 2.0)
    T = np.arctan(T) / alpha

    if alpha == 1.0:
        t1 = ((np.pi / 2.0) + beta * u) * np.tan(u)
        t2 = beta * np.log(((np.pi / 2.0) * w * np.cos(u)) /
                           ((np.pi / 2.0) + beta * u))
        t = (2.0 / np.pi) * (t1 - t2)
        return gamma * t + (delta + beta * 2.0 / np.pi * gamma * np.log(gamma))
    else:
        t1 = (np.cos(alpha * T) * np.cos(u))**(1.0 / alpha)
        t1 = np.sin(alpha * (u + T)) / t1
        t2 = np.cos(alpha * T + (alpha - 1.0) * u) / w
        t2 = t2**((1.0 - alpha) / alpha)
        t = t1 * t2

        out = (gamma * t + delta, u, w)
        return out


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
