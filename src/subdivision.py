import subprocess
from traceback import print_tb

import numpy as np

import flint as ft

from codetiming import Timer
import time

import logging

from idcthandler import IDCTHandler
from idctnhandler import IDCTNHandler

from complexity import Complexity
from branch import Branch

from math import cos, pi
from scipy import optimize, stats

import warnings

import clenshaw as cs

from visu_utils import Verbose

# Logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('subdivision.log')
handler_format = logging.Formatter('%(message)s')
logger.addHandler(handler)


class Subdivision:

    def __init__(self, grid, deg_x, deg_y, poly_file, use_der):
        self.grid = grid
        self.deg_x = deg_x
        self.deg_y = deg_y
        self.poly_file = poly_file
        self.use_der = use_der
        self.cmplxty = Complexity(deg_x, grid.n)
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
                    if self.use_der:
                        if p(min) * p(max) <= 0:
                            y0 = optimize.brentq(p, min, max)
                            idx = np.searchsorted(val[low:up+1], y0) + low
                            return [(idx -1, idx, True)]
                    return [(low, up, up - low == 1)]
                return aux(low, mid, Branch.LEFT, new_node) + aux(mid, up, Branch.RIGHT, new_node)
            else:
                self.cmplxty.negIntEval(branch, node)
                return []
        res = aux(low, up, Branch.ROOT)
        self.cmplxty.endSubdivision()
        return res

    def isolateIntervals(self, poly, n, use_clen, use_idct, use_dsc, use_cs):
        partial_poly = np.empty((n, self.deg_y + 1), dtype=object)
        a = self.grid.x_min
        b = self.grid.x_max
        Verbose.verboseprint("Evaluation...")
        for j in range(self.deg_y + 1):
            p = np.trim_zeros(poly[:,j], 'b')
            if (len(p) == 0):
                p = [0]
            if use_clen:
                # if we want to use Clenshaw with the Chebyshev basis
                with Timer("conversion", logger=None):
                    tmp = np.polynomial.chebyshev.poly2cheb(p)
                tmp[np.abs(tmp) < 1e-15] = 0
                tmp = np.trim_zeros(tmp, 'b')
                if (len(tmp) == 0):
                    tmp = [0]
                for i in range(n):
                    with Timer("evaluation", logger=None):
                        partial_poly[i,j] = np.polynomial.chebyshev.chebval(self.grid.xs[i], tmp)
                    self.cmplxty.incrClenshaw()
            elif not use_idct:
                # if we do not use the Chebyshev basis
                tmp = p
                for i in range(n):
                    with Timer("evaluation", logger=None):
                        partial_poly[i,j] = np.polynomial.polynomial.polyval(self.grid.xs[i], tmp)
                    self.cmplxty.incrHorner()
            else:
                #if we want to use the IDCT with the Chebyshev basis
                idct = IDCTHandler(p, n)
                self.cmplxty.incrIDCT()
                partial_poly[:,j] = idct.getResult(self.grid)
        intervals = np.empty(n, dtype="object")
        distr = np.empty(n, dtype=float)
        Verbose.verboseprint("Subdivision...")
        for i in range(n):
            start = time.perf_counter()
            if (not use_dsc and not use_cs):
                with Timer("subdivision", logger=None):
                    intervals[i] = self.__subdivide(self.grid.ys, 0, n - 1, partial_poly[i].tolist())
            elif use_dsc:
                with Timer("subdivision", logger=None):
                    p_i = np.polynomial.Polynomial(partial_poly[i])
                    l = partial_poly[i].tolist()
                    # with Timer("writing", logger=None):
                    with open('tmp_poly', 'w') as f:
                        f.write('{:d}\n'.format(len(l) - 1))
                        for v in l:
                            f.write('{:d}\n'.format(int(round(v))))
                    # with Timer("dsc", logger=None):
                    adsc = 'test_descartes --subdivision 1 --newton 0 --truncate 0 --sqrfree 0 --intprog 0 tmp_poly'
                    anewdsc = 'test_descartes --intprog 0 tmp_poly'
                    command = anewdsc
                    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    outs, errs = process.communicate()
                    if len(outs) > 4:
                        dsc_out = [[int(y.split('/')[0]) * 2**(-int(y.split('^')[1])) for y in x.split(',')] for x in outs.splitlines()[0][2:-2].split('],[')]
                    elif outs[:2] == '[]':
                        dsc_out = []
                    else:
                        warnings.warn("The polynomial is probably not square-free (no output for ANewDsc).")
                        print(outs)
                        dsc_out = [] 
                    indices = []
                    # with Timer("brent", logger=None):
                    for intvl in dsc_out:
                        a = intvl[0]
                        b = intvl[1]
                        if (np.sign(p_i(a)) * np.sign(p_i(b)) <= 0):
                            y0 = optimize.brentq(p_i, a, b)
                            idx = np.searchsorted(self.grid.ys, y0)
                            if 0 < idx and idx < n:
                                indices.append((idx -1, idx, True))
                        else:
                            idx_a = np.searchsorted(self.grid.ys, a, side='right') -1
                            idx_b = np.searchsorted(self.grid.ys, b)
                            if idx_b <=0 or n <= idx_a:
                                continue
                            idx_a = max(idx_a, 0)
                            idx_b = min (n - 1, idx_b)
                            indices.append((idx_a, idx_b, False))
                    # if True in (e[1] - e[0] != 1 for e in indices):
                    #     print(indices)
                    #     print([[self.ys[x[0]], self.ys[x[1]]] for x in indices])
                    #     print(outs)
                    #     # print(p_i)
                    intervals[i] = indices
                    if len(errs.splitlines()) > 3:
                        intvl_nb = int(errs.splitlines()[2].split('=')[1]) 
                    else:
                        warnings.warn("A bug in ANewDsc's error messages has been ignored (no statistic). The polynomial may not be square-free.")
                        intvl_nb = 0
                    self.cmplxty.descartes(intvl_nb)
            else:
                with Timer("subdivision", logger=None):
                    sols, unks = cs.solve_polynomial_taylor(partial_poly[i], n=10)
                    if len(sols) == 0:
                        sols = np.empty((0, 2), int)
                    if len(unks) == 0:
                        unks = np.empty((0, 2), int)
                    cs_out = np.concatenate((sols, unks), 0)
                    p_i = np.polynomial.Polynomial(partial_poly[i])
                    # with Timer("brent", logger=None):
                    indices = []
                    for intvl in cs_out:
                        a = intvl[0] - intvl[1]
                        b = intvl[0] + intvl[1]
                        if a < self.grid.y_min or self.grid.y_max < b:
                            continue
                        if (np.sign(p_i(a)) * np.sign(p_i(b)) <= 0):
                            y0 = optimize.brentq(p_i, a, b)
                            idx = np.searchsorted(self.grid.ys, y0)
                            if 0 < idx and idx < n:
                                indices.append((idx -1, idx, True))
                        else:
                            idx_a = np.searchsorted(self.grid.ys, a, side='right') -1
                            idx_b = np.searchsorted(self.grid.ys, b)
                            if idx_b <=0 or n <= idx_a:
                                continue
                            idx_a = max(idx_a, 0)
                            idx_b = min (n - 1, idx_b)
                            indices.append((idx_a, idx_b, False))
                    intervals[i] = indices
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
        if use_clen:
            logger.info('Clenshaw')
        elif not use_idct:
            logger.info('Classical')
        else:
            logger.info('IDCT')
        logger.info("")

        return intervals

    def isolate2d(self, poly, n):
        Verbose.verboseprint("Evaluation...")
        idct = IDCTNHandler()
        eval = idct.idct2d(poly, (n,n))
        intervals = np.empty(n, dtype="object")
        for i in range(n):
            l = []
            for j in range(n -1):
                if np.sign(eval[i,j]) * np.sign(eval[i,j + 1]) <= 0:
                    l.append((j,j+1, True))
            intervals[i] = l
        nodes = [cos((2 * i + 1) * pi / (2 * n)) for i in range(n)]
        self.grid.xs = nodes
        self.grid.ys = nodes
        return intervals

    def getGrid(self):
        return self.grid

    def drawSubdivisions(self):
        self.cmplxty.draw()
    
    def printComplexity(self):
        print(self.cmplxty)
    
    def getSubdivisionTimeDistribution(self):
        return self.time_distr
