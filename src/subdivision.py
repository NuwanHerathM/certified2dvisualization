
import numpy as np

import flint as ft

from math import cos, pi
from scipy.special import comb
import itertools

from codetiming import Timer

import logging

from idcthandler import IDCTHandler

# Obsolete chunk of code
# # Takes care of the decorators set for the profiling
# try:
#     profile  # throws an exception when profile isn't defined
# except NameError:
#     profile = lambda x: x   # if it's not defined simply ignore the decorator.

# Logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('subdivision.log')
handler_format = logging.Formatter('%(message)s')
logger.addHandler(handler)


class Subdivision:

    def __init__(self, xs, ys, deg_x, deg_y, poly_file):
        self.xs = xs
        self.ys = ys
        self.deg_x = deg_x
        self.deg_y = deg_y
        self.poly_file = poly_file
        self.grid = None

    # Functions

    # @profile
    @staticmethod
    def subdivide(val, low, up, poly):
        p = ft.arb_poly(poly)

        def aux(low, up):
            min = val[low]
            max = val[up]
            mid = int(low + (up - low) / 2)
            median = (min + max) / 2
            radius = (max - min) / 2

            a = p(ft.arb(median,radius))
            if 0 in a:
                if up - low == 1:
                    return [(low, up)]
                return aux(low, mid) + aux(mid, up)
            else:
                return []
        
        return aux(low, up)

    # @profile
    def isolateIntervals(self, poly, n, switch):
        partial_poly = np.empty((n, self.deg_y + 1), dtype=object)
        rad = (self.ys[-1] - self.ys[0]) / (2 * (n - 1))
        rad = 0
        a = self.xs[0]
        b = self.xs[-1]
        alpha = (self.xs[-1] - self.xs[0]) / 2
        shift = self.xs[-1] - alpha
        deg_can = 0
        deg_conv = 0
        deg_ch = 0
        deg_q = 0
        deg_s = 0
        deg_tmp = 0
        for j in range(self.deg_y + 1):
            p = np.trim_zeros(poly[j], 'b')
            if (len(p) == 0):
                p = [0]
            deg_can += len(p) - 1
            if (switch == 1):
                # if we want to use Clenshaw with the Chebyshev basis
                with Timer("conversion", logger=None):
                    tmp = np.polynomial.chebyshev.poly2cheb(p)
                deg_conv += len(np.trim_zeros(tmp, 'b')) - 1
                # TO DO: sqrt of the degree
                tmp[np.abs(tmp) < 1e-15] = 0
                tmp = np.trim_zeros(tmp, 'b')
                if (len(tmp) == 0):
                    tmp = [0]
                deg_q += len(tmp) - 1
                for i in range(n):
                    with Timer("evaluation", logger=None):
                        partial_poly[i,j] = np.polynomial.chebyshev.chebval(self.xs[i], tmp)
            elif (switch == 0):
                # if we do not use the Chebyshev basis
                tmp = p
                for i in range(n):
                    with Timer("evaluation", logger=None):
                        partial_poly[i,j] = np.polynomial.polynomial.polyval(self.xs[i], tmp)
            else:
                #if we want to use the IDCT with the Chebyshev basis
                with Timer("change", logger=None):
                    s = [sum(x) for x in itertools.zip_longest(*[[comb(i, k) * alpha ** k * shift ** (i - k) * c for k in range(i + 1)] for i, c in enumerate(p)], fillvalue=0)]
                # s = sum([np.polynomial.Polynomial([shift, alpha])**i * c for i, c in enumerate(poly[j])])
                deg_s += len(s) - 1
                deg_ch += len(np.trim_zeros(s, 'b')) - 1
                with Timer("conversion", logger=None):
                    tmp = np.polynomial.chebyshev.poly2cheb(s)
                deg_tmp += len(tmp) - 1
                deg_conv += len(np.trim_zeros(tmp, 'b')) - 1
                # tmp = np.zeros(self.deg_y + 1)
                # tmp[:len(c)] = c
                with Timer("evaluation", logger=None):
                    idct = IDCTHandler(p, n, self.ys[0], self.ys[-1])
                    partial_poly[:,j] = idct.getResult()
                    self.grid = idct.getGrid()
        intervals = np.empty(n, dtype="object")
        for i in range(n):
            with Timer("subdivision", logger=None):
                intervals[i] = Subdivision.subdivide(self.ys, 0, n - 1, partial_poly[i].tolist())
        
        logger.info(self.poly_file)
        logger.info("="*len(self.poly_file))
        if (switch == 1):
            logger.info('Clenshaw polynomial degree')
            logger.info(f"Before Chebyshev:\t{self.deg_x} -> {deg_can / (self.deg_y + 1)}")
            logger.info(f"After conversion:\t{deg_conv / (self.deg_y + 1)}")
            logger.info(f"After Clenshaw:\t{deg_q / (self.deg_y + 1)}")
        elif (switch == 0):
            logger.info('Classical polynomial degree')
            logger.info(f"Actual polynomial:\t{self.deg_x} -> {deg_can / (self.deg_y + 1)}")
        else:
            logger.info('IDCT polynomial degree')
            logger.info(f"Before Chebyshev:\t{self.deg_x} -> {deg_can / (self.deg_y + 1)}")
            logger.info(f"After change:\t{deg_s / (self.deg_y + 1)} -> {deg_ch / (self.deg_y + 1)}")
            logger.info(f"After conversion:\t{deg_tmp / (self.deg_y + 1)} -> {deg_conv / (self.deg_y + 1)}")
        logger.info("")

        return intervals

    def getGrid(self):
        return self.grid
