from re import T
import numpy as np
import scipy.fftpack as fp
from math import cos, log, pi, floor, sqrt
import sys
import flint as ft
from scipy.fft import idct

from inspect import currentframe, getframeinfo

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
        try:
            nodes_power[:,i] = nodes[i]**np.arange(d)
        except FloatingPointError:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
            print("An underflow issue has been handled manually.")
            val = nodes[i]
            max_d = floor(log(sys.float_info.min) / log(abs(val)))
            head = val**np.arange(max_d)
            tail = np.repeat(sys.float_info.min, d - max_d)
            if val < 0:
                tail[::2] *= (-1)**max_d
                tail[1::2] *= (-1)**(max_d+1)
            nodes_power[:,i] = np.append(head, tail)
    node_eval = np.asmatrix(polys) * np.asmatrix(nodes_power)
    dct_eval = np.empty((n, d))
    for i in range(n):
        dct_eval[i] = fp.dct(node_eval[i]) 
    dct_eval /= d
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

def interval_idct(poly, n=None):
    if n is None:
        n = len(poly)
    x = np.zeros(n+1)
    x[:len(poly)] = poly
    v = np.zeros(n, dtype=object)
    N = ft.acb(n)
    for i in range(n):
        w = ft.acb.exp_pi_i(ft.acb(i) / (2 * N))
        v[i] = w / 2 * ft.acb(x[i], -x[n-i])
    V = ft.acb.dft(v, True)
    X = np.zeros(n, dtype=object)
    X[::2] = V[:floor((n+1)/2)]
    X[1::2] = V[-1:floor((n-1)/2):-1]

    for i in range(n):
        X[i] = X[i].real

    return X

def error_idct(poly, n=None):
    if n is None:
        n = len(poly)
    x_max = max(poly)
    u = np.finfo(float).eps
    rho = 2 * u # if FMA
    g = u / sqrt(2) + rho * (1 + u / sqrt(2))
    factor = sqrt(2) * (sqrt(2) * (1 + u)**2 * (1 + g)**2 * ((1 + u)**n * (1 + g)**(n-2) - 1) + (1 + u)**2 * (1 + g)**2 - 1)
    X = idct(poly, n=n)
    res = np.empty(n, dtype=object)
    for i in range(n):
        res[i] = ft.arb(X[i], x_max * factor)
    return res