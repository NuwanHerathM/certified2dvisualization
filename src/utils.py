# distutils: language=c++
from __future__ import print_function, with_statement
from re import T
import numpy as np
from math import cos, log, pi, floor, sqrt, log2
from scipy.special import comb
import sys
import flint as ft
import scipy.fft
from interval import fpu

from inspect import currentframe, getframeinfo

class UpwardRounding:
    """
    Context manager for upward rounding.
    """
    saved = None

    def __enter__(self):
        self.saved = fpu._fegetround()
        fpu._fesetround(fpu._fe_upward)
    
    def __exit__(self, *args, **kwargs):
        fpu._fesetround(self.saved)

def polys2cheb_dct(polys):
    """
    Vectorized polynomial conversion from canonical basis to Chebyshev basis.

    Parameters
    ----------
    polys: 2D array
           Array of polynomials (arrays of coefficients)
    
    Returns
    -------
    chebs: 2D array
              Array of polynomials in the Chebyshev basis

    Notes
    -----
    For coefficients which are close to zero in the Chebyshev basis, there is no guarantee on the relative error.
    For these values, we guarantee nonetheless an absolute error of 1e-14 at worse. So, the sign may not be correct.
    """
    dim_1 = (polys.ndim == 1) # check if it is a polynomial instead of an array of polynomials
    if dim_1:
        polys.shape = (1, len(polys))
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
    node_eval = polys @ nodes_power
    dct_eval = np.empty((n, d))
    for i in range(n):
        dct_eval[i] = scipy.fft.dct(node_eval[i]) 
    dct_eval /= d
    dct_eval[:,0] /= 2

    res = dct_eval.reshape((-1,)) if dct_eval.shape[0] == 1 else dct_eval

    return res

def interval_polys2cheb_dct(polys):
    """
    Interval arithmetic conversion from canonical basis to Chebyshev basis.

    Parameters
    ----------
    polys: array
          Polynomial or array of polynomials
    
    Returns
    -------
    chebs: 2D array
              Array of polynomials in the Chebyshev basis
    """
    dim_1 = (polys.ndim == 1) # check if it is a polynomial instead of an array of polynomials
    if dim_1:
        polys.shape = (1, len(polys))
    (n, d) = polys.shape
    nodes_power = np.empty((d, d), dtype=object)
    D = ft.acb(d)
    nodes = np.array([ft.acb.cos_pi((2 * ft.acb(i) + 1) / (2 * D)).real for i in range(d)])
    for i in range(d):
        for j in range(d):
            nodes_power[j,i] = nodes[i]**j
    node_eval = polys @ nodes_power
    dct_eval = np.empty((n, d), dtype=object)
    for i in range(n):
        dct_eval[i] = interval_dct(node_eval[i])

    dct_eval /= d
    dct_eval[:,0] /= 2

    res = dct_eval.reshape((-1,)) if dct_eval.shape[0] == 1 else dct_eval

    return res * 2

def error_polys2cheb_dct(polys):
    """
    Conversion from canonical basis to Chebyshev basis with error bound.

    Parameters
    ----------
    polys: array
          Polynomial or array of polynomials
    
    Returns
    -------
    chebs: 2D array
              Array of polynomials in the Chebyshev basis
    """
    dim_1 = (polys.ndim == 1) # check if it is a polynomial instead of an array of polynomials
    if dim_1:
        polys.shape = (1, len(polys))
    (n, d) = polys.shape
    nodes_power = np.empty((d, d), dtype=object)
    # D = ft.acb(d)
    # nodes = np.array([ft.acb.cos_pi((2 * ft.acb(i) + 1) / (2 * D)).real for i in range(d)])
    D = ft.arb(d)
    nodes = ft.arb_mat([[ft.arb.cos_pi((2 * ft.arb(i) + 1) / (2 * D))] for i in range(d)])
    for i in range(d):
        for j in range(d):
            nodes_power[j,i] = nodes[i,0]**j
    # node_eval = polys @ nodes_power
    arb_polys = ft.arb_mat(polys.tolist())
    arb_nodes_power = ft.arb_mat(nodes_power.tolist())
    node_eval = arb_polys * arb_nodes_power
    # print(arb_polys)
    # print(arb_nodes_power)
    # print(len(node_eval))
    dct_eval = np.empty((n, d), dtype=object)
    node_table = node_eval.table()
    for i in range(n):
        tmp = np.empty(d, dtype=object)
        for j in range(d):
            tmp[j] = node_eval[i,j]
        dct_eval[i] = error_dct(tmp)
        # dct_eval[i] = error_dct(node_eval[i])

    dct_eval /= d
    dct_eval[:,0] /= 2

    res = dct_eval.reshape((-1,)) if dct_eval.shape[0] == 1 else dct_eval

    return res * 2

def altered_idct(poly, n):
    """
    Alteration of scipy.fft.idct.

    .. math:: \forall k \in \left[0,n-1\right], X_k = \frac{1}{2} x_0 + \sum_{i = 1}^{n-1} x_i ~\cos\left[\frac{\pi i (2k+1)}{2n}\right]

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
    """
    return scipy.fft.idct(poly, n=n) * n

def idct_eval(poly, n):
    """
    Multipoint evaluation using the IDCT of a polynomial in the Chebyshev basis on Chebyshev nodes.

    Parameters
    ----------
    poly: array
          Polynomial or array of polynomials in Chebyshev basis
    n: int
       Number of points
    
    Returns
    -------
    eval: array
          Evaluation of the polynomial(s) on the roots of the nth Chebyshev polynomial
    """
    return (scipy.fft.idct(poly, n=n).T * n + poly[...,0] / 2).T

def interval_idct_eval(poly, n):
    """
    Interval multipoint evaluation using the IDCT of a polynomial in the Chebyshev basis on Chebyshev nodes.

    Parameters
    ----------
    poly: array
          Polynomial in Chebyshev basis
    n: int
       Number of points
    
    Returns
    -------
    eval: array
          Evaluation of the polynomial on the roots of the nth Chebyshev polynomial
    """
    assert poly.dtype is np.dtype('O'), "The polynomial should be an array of flint.arb."
    return (interval_idct(poly, n) + poly[0] / 2)

def loop_interval_idct_eval(polys, n):
    """
    Looped version of interval_idct_eval, for multiple polynomials.
    """
    l = polys.shape[0]
    res = np.empty((l, n), dtype=object)
    for i in range(l):
        res[i] = interval_idct_eval(polys[i,:], n)
    return res

def error_idct_eval(poly, n):
    assert poly.dtype is np.dtype('O'), "The polynomial should be an array of flint.arb."
    return (error_idct(poly, n).T + poly[...,0] / 2).T

def altered_dct(poly, n=None):
    """
    Alteration of scpiy.fft.dct.

    .. math:: \forall k \in \left[0,n-1\right], X_k = \sum_{i = 0}^{n-1} x_i ~\cos\left[\frac{\pi (2i+1) k}{2n}\right]

    Parameters
    ----------
    poly: array
          Polynomial or array of polynomials
    n: int
       Number of points
    
    Returns
    -------
    idct: array
          DCT or array of DCTs
    """
    if n is None:
        n = poly.shape[-1]
    return scipy.fft.dct(poly, n=n) / 2

from codetiming import Timer
root_dict = {}
halfroot_dict = {}
def interval_idct(poly, n=None):
    """
    Interval arithmetic fast IDCT.

    Parameters
    ----------
    poly: array
          Polynomial
    n: int
       Number of points
    
    Returns
    -------
    idct: array
          IDCT
    """
    if n is None:
        n = len(poly)
    X = np.zeros(n+1)
    X[:len(poly)] = poly # zero padding
    V = np.zeros(n, dtype=object)
    N = ft.acb(n)
    if n in root_dict:
        w = root_dict[n]
    else:
        with Timer("w", logger=None):
            w = np.empty(n, dtype=object)
            for i in range(n):
                w[i] = ft.acb.exp_pi_i(ft.acb(i) / (2 * N))
        root_dict[n] = w
    with Timer("V", logger=None):
        V = w / 2 * (X[:-1] - ft.acb(0,1) * X[-1:0:-1])
    if n % 2 == 1:
        v = ft.acb.dft(V, True)
        for i in range(n):
            v[i] = v[i].real
    else:
        with Timer("T", logger=None):
            T = np.empty(int(n/2) + 1, dtype=object)
            V_conj = np.empty(floor(n/4)+1, dtype=object)
            for i in range(floor(n/4)+1):
                V_conj[i] = ft_acb_conj(V[int(n/2)-i])
            if n in halfroot_dict:
                half_w = halfroot_dict[n]
            else:
                half_w = np.empty(floor(n/4)+1, dtype=object)
                for i in range(floor(n/4)+1):
                    half_w[i] = ft.acb.exp_pi_i(2 * ft.acb(i) / N)
                halfroot_dict[n] = half_w
            V_sum = V[:floor(n/4)+1] + V_conj
            V_diff = V[:floor(n/4)+1] - V_conj
            V_diff_prod = ft.acb(0,1) * half_w * V_diff
            T[:floor(n/4)+1] = 1 / 2 * (V_sum + V_diff_prod)
            T_tmp = 1 / 2 * (V_sum - V_diff_prod)
            for i in range(floor(n/4)+1):
                T[int(n/2)-i] = ft_acb_conj(T_tmp[i])
        t = ft.acb.dft(T[:-1], True)
        v = np.empty(n, dtype=object)
        for i in range(int(n/2)):
            v[2*i] = t[i].real
            v[2*i+1] = t[i].imag
    x = np.zeros(n, dtype=object)
    x[::2] = v[:floor((n+1)/2)] 
    x[1::2] = v[-1:floor((n-1)/2):-1] 

    return x * n

neghalfroot_dict = {}
def interval_dct(poly, n=None):
    """
    Interval arithmetic fast DCT.

    Parameters
    ----------
    poly: array
          Polynomial
    n: int
       Number of points
    
    Returns
    -------
    dct: array
          DCT
    """
    if n is None:
        n = len(poly)
    x = np.zeros(n, dtype=object)
    x[:len(poly)] = poly # zero padding
    v = np.empty(n, dtype=object)
    v[:floor((n+1)/2)] = x[::2]
    v[-1:floor((n+1)/2)-1:-1] = x[1::2]
    N = ft.acb(n)
    if n % 2 == 1:
        V = ft.acb.dft(v)
    else:
        with Timer("TT", logger=None):
            t = np.empty(int(n/2), dtype=object)
            for i in range(int(n/2)):
                t[i] = ft.acb(v[2*i],v[2*i+1])
            T = ft.acb.dft(t)
            V = np.empty(n, dtype=object)
            V[0] = T[0].real + T[0].imag
            V[int(n/2)] = T[0].real - T[0].imag
            T_conj = np.empty(floor(n/4), dtype=object)
            for i in range(1,floor(n/4)+1):
                T_conj[i-1] = ft_acb_conj(T[int(n/2)-i])
            if n in halfroot_dict:
                neghalf_w = neghalfroot_dict[n]
            else:
                neghalf_w = np.empty(floor(n/4), dtype=object)
                for i in range(floor(n/4)):
                    neghalf_w[i] = ft.acb.exp_pi_i(-2 * ft.acb(i+1) / N)
                neghalfroot_dict[n] = neghalf_w
            T_sum = T[1:floor(n/4)+1] + T_conj
            T_diff = T[1:floor(n/4)+1] - T_conj
            T_diff_prod = ft.acb(0,1) * neghalf_w * T_diff
            V[1:floor(n/4)+1] = 1 / 2 * (T_sum - T_diff_prod)
            V_tmp = 1 / 2 * (T_sum + T_diff_prod)
            for i in range(1,floor(n/4)+1):
                V[int(n/2)-i] = ft_acb_conj(V_tmp[i-1])
            for i in range(1,int(n/2)):
                V[n-i] = ft_acb_conj(V[i])
    X = np.empty(n+1, dtype=object)
    for i in range(floor(n/2)+1):
        V[i] *= 2 * ft.acb.exp_pi_i(-ft.acb(i) / (2 * N))
        X[i] = V[i].real
        X[n-i] = -V[i].imag

    return X[:-1] / 2

def factor_dct(n):
    log2_n = log2(n)
    u = np.finfo(float).eps
    rho = sqrt(5) * u # if naive multiplication (no FMA)
    g = u / sqrt(2) + rho * (1 + u / sqrt(2))
    factor = sqrt(2) * n * (((1 + u)**(log2_n-1) * (1 + g)**(log2_n-3) - 1) * (1 + (1 + u)**3 * (1 +g)**2) + sqrt(2) * ((1 + u)**3 * (1 + g)**2 - 1))
    return factor

def error_dct(polys, n=None):
    """
    DCT with its error bound.

    Parameters
    ----------
    polys: array
          Polynomial or array of polynomials
    n: int
       Number of points
    
    Returns
    -------
    dct: array
          DCT
    """
    dim_1 = (polys.ndim == 1) # check if it is a polynomial instead of an array of polynomials
    if dim_1:
        polys.shape = (1, len(polys))
    if n is None:
        n = polys.shape[-1]
    X = altered_dct(polys, n=n)

    m = polys.shape[0]
    x_max = np.amax(polys,axis=1)
    with UpwardRounding(): # force upward floating point rounding (twoward infinity if positive and -infinity if negative)
        factor = factor_dct(n)
        res = np.empty((m,n), dtype=object)
        for i in range(m):
            bound = x_max[i] * factor
            for j in range(n):
                res[i,j] = ft.arb(X[i,j], bound)

    if dim_1:
        polys.shape = (polys.shape[-1],)
        res = res.reshape((-1,))

    return res

def ft_acb_conj(z):
    """
    Compute the conjugate for the class flint.acb.
    Such a function exists though for the class flint.acb_mat (flint.acb_mat.conjugate).
    """
    return ft.acb(z.real,-z.imag)

def factor_idct(n):
    log2_n = log2(n)
    u = np.finfo(float).eps
    rho = sqrt(5) * u # if naive multiplication (no FMA)
    g = u / sqrt(2) + rho * (1 + u / sqrt(2))
    factor = sqrt(2) * (sqrt(2) * (1 + u)**3 * (1 + g)**2 * ((1 + u)**(log2_n-1) * (1 + g)**(log2_n-3) - 1) + (1 + u)**3 * (1 + g)**2 - 1)
    return factor

def error_idct(polys, n=None):
    """
    IDCT with its error bound.

    Parameters
    ----------
    polys: array
          Polynomial or array of polynomials
    n: int
       Number of points
    
    Returns
    -------
    idct: array
          IDCT
    """
    dim_1 = (polys.ndim == 1) # check if it is a polynomial instead of an array of polynomials
    if dim_1:
        polys.shape = (1, len(polys))
    if n is None:
        n = polys.shape[-1]
    X = altered_idct(polys, n=n)

    m = polys.shape[0]
    x_max = np.amax(polys,axis=1)
    with UpwardRounding(): # force upward floating point rounding (twoward infinity if positive and -infinity if negative)
        factor = factor_idct(n)
        res = np.empty((m,n), dtype=object)
        for i in range(m):
            bound = x_max[i] * factor
            for j in range(n):
                res[i,j] = ft.arb(X[i,j], bound)

    if dim_1:
        polys.shape = (polys.shape[-1],)
        res = res.reshape((-1,))

    return res

def comb2D(n, m):
    """
    Compute the 2D binomial coefficients.

    For 0<=i<n and 0<=j<m,
    .. math:: C(i,j,n,m) = (n+m)! / i!j!(n+m-i-j)!.

    Parameters
    ----------
    n : int
        Number of lines of the output
    m : int
        Number of columns of the output
    """
    k1 = np.empty((n,m), dtype=float)
    for i in range(0, m):
        k1[:,i] = comb(n + m - i, range(0,n))
    k2 = comb(n + m, range(0,m))
    out = np.multiply(k1,k2)
    
    return out

def factorial2D(n,m):
    """
    Compute the 2D factorials.
    Using interval arithmetic to avoid overflow issues.

    For 0<=i<n and 0<=j<m,
    .. math:: fact(i,j) = i!j!.

    Parameters
    ----------
    n : int
        Number of lines of the output
    m : int
        Number of columns of the output
    """
    x = np.empty((1,n), dtype=object)
    for i in range(n):
        x[0,i] = ft.arb(i).fac()
    y = np.empty((1,m), dtype=object)
    for j in range(m):
        y[0,j] = ft.arb(j).fac()
    return x.T @ y

def subdivide(val, p):
    def aux(low, up):
        mini = val[low]
        maxi = val[up]
        mid = int(low + (up - low) / 2)
        median = (mini + maxi) / 2
        radius = (maxi - mini) / 2

        ball = ft.arb(median,radius)
        a = p(ball)
        if 0 in a:
            if (up - low == 1 ):
                return [(low, up, up - low == 1)]
            return aux(low, mid) + aux(mid, up)
        else:
            return []
    res = aux(0, len(val)-1)
    return res