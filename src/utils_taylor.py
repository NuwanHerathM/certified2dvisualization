import numpy as np
import scipy.fftpack as fp
from math import cos, pi

def polys2cheb_dct(polys):
    """
    Vectorized polynomial conversion from canonical basis to Chebyshev basis.

    Parameters
    ----------
    polys: 2D array
           Array of polynomials (arrays of coefficients)
    
    Returns
    -------
    dct_eval: 2D array
              Array of polynomials in the Chebyshev basis
    """
    (n, d) = polys.shape
    nodes_power = np.empty((d, d))
    nodes = np.array([cos((2 * i + 1) * pi / (2 * d)) for i in range(d)])
    for i in range(d):
        nodes_power[:,i] = nodes[i]**np.arange(d) #if nodes[i] > 10e-12 else np.zeros(d) # arbitrary decision to avoid underflow
    node_eval = np.asmatrix(polys) * np.asmatrix(nodes_power)
    dct_eval = np.empty((n, d))
    for i in range(n):
        dct_eval[i] = fp.dct(node_eval[i]) 
    dct_eval /= (d)
    dct_eval[:,0] /= 2

    return dct_eval

def corrected_idct(poly, n):
    """
    Correction of scipy.fftpack.idct.

    Parameters
    ----------
    poly: array
          Polynomial or array of polynomials
    n: int
       Number of points
    
    Returns
    -------
    idct: array
          IDCT or array of IDCTs
    
    Notes
    -----
    The output is the transpose of what one could expect. It was more convenient for me this way.
    """
    return (fp.idct(poly, n=n).T + poly[...,0]) / 2