from numpy.polynomial.chebyshev import poly2cheb
import numpy as np
from scipy.fftpack import idct
from utils import error_idct, interval_idct, polys2cheb_dct, interval_polys2cheb_dct

# Uncomment the line corresponing to the function you want to use

functions = {
    'idct' : [
        # idct,
        interval_idct,
        # error_idct
    ],
    'poly2cheb' : [
        # np.polynomial.chebyshev.poly2cheb,
        # polys2cheb_dct,
        interval_polys2cheb_dct
    ]
}
