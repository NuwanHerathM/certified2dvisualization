import numpy as np
import scipy.fftpack as fp
from math import cos, pi

def polys2cheb_dct(polys):
    """
    Vectorized polynomial conversion from canonical basis to Chebyshev basis

    Parameters
    ----------
    polys: 2D array
           Array of polynomials (arrays of coefficients)
    """
    (n, d) = polys.shape
    nodes_power = np.empty((d, d))
    nodes = np.array([cos((2 * i + 1) * pi / (2 * d)) for i in range(d)])
    for i in range(d):
        nodes_power[:,i] = nodes[i]**np.arange(d)
    node_eval = np.asmatrix(polys) * np.asmatrix(nodes_power)
    dct_eval = np.empty((n, d))
    for i in range(n):
        dct_eval[i] = fp.dct(node_eval[i]) 
    dct_eval /= (d)
    dct_eval[:,0] /= 2

    return dct_eval

def corrected_idct(poly, n):
    """
    Correction of scipy.fftpack.idct

    Parameters
    ----------
    poly: array
          Polynomial or array of polynomials
    n: int
       Number of points
    """
    return (fp.idct(poly, n=n).T + poly[...,0]) / 2