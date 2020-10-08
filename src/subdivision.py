import subprocess

import numpy as np

import flint as ft

from math import cos, pi

from codetiming import Timer
import time

import logging

from idcthandler import IDCTHandler

from complexity import Complexity
from branch import Branch

from scipy import optimize, stats

import warnings

# Logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('subdivision.log')
handler_format = logging.Formatter('%(message)s')
logger.addHandler(handler)


class Subdivision:

    def __init__(self, xs, ys, deg_x, deg_y, poly_file, use_der):
        self.xs = xs
        self.ys = ys
        self.deg_x = deg_x
        self.deg_y = deg_y
        self.poly_file = poly_file
        self.use_der = use_der
        self.grid = None
        self.cmplxty = Complexity(deg_x, len(xs))
        self.time_distr = None

    # Functions

    def __subdivide(self, val, low, up, poly):
        p = ft.arb_poly(poly)
        poly_der = np.polynomial.polynomial.polyder(poly).tolist()
        p_der = ft.arb_poly(poly_der)
        self.cmplxty.resetSubdivision()

        def aux(low, up, branch, node=None):
            min = val[low]
            max = val[up]
            mid = int(low + (up - low) / 2)
            median = (min + max) / 2
            radius = (max - min) / 2

            ball = ft.arb(median,radius)
            a = p(ball)
            if 0 in a:
                new_node = self.cmplxty.posIntEval(branch, node)
                if (up - low == 1 or (self.use_der and 0 not in p_der(ball))):
                    if self.use_der and p(min) * p(max) > 0:
                        return []
                    return [(low, up)]
                return aux(low, mid, Branch.LEFT, new_node) + aux(mid, up, Branch.RIGHT, new_node)
            else:
                self.cmplxty.negIntEval(branch, node)
                return []
        res = aux(low, up, Branch.ROOT)
        self.cmplxty.endSubdivision()
        return res

    def isolateIntervals(self, poly, n, switch, use_dsc):
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
                tmp[np.abs(tmp) < 1e-15] = 0
                tmp = np.trim_zeros(tmp, 'b')
                if (len(tmp) == 0):
                    tmp = [0]
                deg_q += len(tmp) - 1
                for i in range(n):
                    with Timer("evaluation", logger=None):
                        partial_poly[i,j] = np.polynomial.chebyshev.chebval(self.xs[i], tmp)
                    self.cmplxty.incrClenshaw()
            elif (switch == 0):
                # if we do not use the Chebyshev basis
                tmp = p
                for i in range(n):
                    with Timer("evaluation", logger=None):
                        partial_poly[i,j] = np.polynomial.polynomial.polyval(self.xs[i], tmp)
                    self.cmplxty.incrHorner()
            else:
                #if we want to use the IDCT with the Chebyshev basis
                idct = IDCTHandler(p, n, self.ys[0], self.ys[-1])
                self.cmplxty.incrIDCT()
                partial_poly[:,j] = idct.getResult()
                self.grid = idct.getGrid()
                deg_ch, deg_conv = map(sum,zip((deg_ch, deg_conv),idct.getDegrees()))
        intervals = np.empty(n, dtype="object")
        distr = np.empty(n, dtype=float)
        for i in range(n):
            start = time.perf_counter()
            if (not use_dsc):
                with Timer("subdivision", logger=None):
                    intervals[i] = self.__subdivide(self.ys, 0, n - 1, partial_poly[i].tolist())
            else:
                with Timer("subdivision", logger=None):
                    p_i = np.polynomial.Polynomial(partial_poly[i])
                    l = partial_poly[i].tolist()
                    with Timer("writing", logger=None):
                        with open('tmp_poly', 'w') as f:
                            f.write('{:d}\n'.format(len(l) - 1))
                            for v in l:
                                f.write('{:d}\n'.format(int(round(v))))
                    with Timer("dsc", logger=None):
                        adsc = 'test_descartes --subdivision 1 --newton 0 --truncate 0 --sqrfree 0 --intprog 0 tmp_poly'
                        anewdsc = 'test_descartes tmp_poly'
                        command = anewdsc
                        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        outs, errs = process.communicate()
                    if len(outs) > 4:
                        dsc_out = [[int(y.split('/')[0]) * 2**(-int(y.split('^')[1])) for y in x.split(',')] for x in outs.splitlines()[0][2:-2].split('],[')]
                    elif outs[:2] == '[]':
                        dsc_out = []
                    else:
                        warnings.warn("A bug in ANewDsc's output has been ignored (no value).")
                        print(outs)
                        dsc_out = [] 
                    indices = []
                    with Timer("brent", logger=None):
                        for intvl in dsc_out:
                            a = intvl[0]
                            b = intvl[1]
                            if (p_i(a) * p_i(b) < 0):
                                y0 = optimize.brentq(p_i, a, b)
                                idx = np.searchsorted(self.ys, y0)
                                if 0 < idx and idx < len(self.ys):
                                    indices.append([idx -1, idx])
                        intervals[i] = indices
                    if len(errs.splitlines()) > 3:
                        intvl_nb = int(errs.splitlines()[2].split('=')[1]) 
                    else:
                        warnings.warn("A bug in ANewDsc's error messages has been ignored (no statistic).")
                        intvl_nb = 0
                    self.cmplxty.descartes(intvl_nb)
            distr[i] = round(time.perf_counter() - start,4)
        self.time_distr = (distr, stats.relfreq(distr, numbins=20))
        print(Timer.timers)
        Timer.timers.pop("writing", None)
        Timer.timers.pop("dsc", None)
        Timer.timers.pop("brent", None)
        self.cmplxty.log()
        # self.cmplxty.subdivision_analysis()

        logger.info(self.poly_file)
        logger.info("="*len(self.poly_file))
        if (switch == 1):
            logger.info('Clenshaw polynomial degree')
            logger.info(f"Before Chebyshev:\t{deg_can / (self.deg_y + 1)}")
            logger.info(f"After conversion:\t{deg_conv / (self.deg_y + 1)}")
            logger.info(f"After Clenshaw:\t{deg_q / (self.deg_y + 1)}")
        elif (switch == 0):
            logger.info('Classical polynomial degree')
            logger.info(f"Actual polynomial:\t{deg_can / (self.deg_y + 1)}")
        else:
            logger.info('IDCT polynomial degree')
            logger.info(f"Before Chebyshev:\t{deg_can / (self.deg_y + 1)}")
            logger.info(f"After change:\t{deg_ch / (self.deg_y + 1)}")
            logger.info(f"After conversion:\t{deg_conv / (self.deg_y + 1)}")
        logger.info("")

        return intervals

    def getGrid(self):
        return self.grid

    def drawSubdivisions(self):
        self.cmplxty.draw()
    
    def printComplexity(self):
        print(self.cmplxty)
    
    def getSubdivisionTimeDistribution(self):
        return self.time_distr
