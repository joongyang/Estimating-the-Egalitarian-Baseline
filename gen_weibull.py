import numpy as np

def generate_3p_weibull(n, gamma, scale, shape):
    """
    Generates data from a 3-parameter Weibull distribution.
    Formula: y = gamma + scale * (Weibull(shape))^(1/shape)
    
    Parameters:
    n : int - number of samples
    gamma : float - location parameter (minimum threshold)
    scale : float - scale parameter (lambda)
    shape : float - shape parameter (alpha)
    """
    # numpy.random.weibull returns values from a Weibull distribution with shape=a, scale=1
    # We transform it to the 3-parameter version
    base_weibull = np.random.weibull(shape, n)
    return gamma + (scale * base_weibull)
