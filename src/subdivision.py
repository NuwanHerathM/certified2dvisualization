import subprocess

import numpy as np

import flint as ft

from math import cos, pi

from codetiming import Timer

import logging

from idcthandler import IDCTHandler

from complexity import Complexity
from branch import Branch


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
        self.cmplxty = Complexity(deg_x, len(xs))

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
                if (up - low == 1 or 0 not in p_der(ball)):
                    if p(min) * p(max) > 0:
                        return []
                    return [(low, up)]
                return aux(low, mid, Branch.LEFT, new_node) + aux(mid, up, Branch.RIGHT, new_node)
            else:
                self.cmplxty.negIntEval(branch, node)
                return []
        res = aux(low, up, Branch.ROOT)
        self.cmplxty.endSubdivision()
        return res

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
        for i in range(n):
            with Timer("subdivision", logger=None):
                intervals[i] = self.__subdivide(self.ys, 0, n - 1, partial_poly[i].tolist())
            # l = partial_poly[i].tolist()
            # with open('tmp_poly', 'w') as f:
            #     f.write('{:d}\n'.format(len(l) - 1))
            #     for v in l:
            #         f.write('{:d}\n'.format(round(v)))
            # command = '../../anewdsc/test_descartes_linux64 --subdivision 1 --newton 0 --truncate 0 --sqrfree 0 --intprog 0 tmp_poly'
            # process = subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            # outs, errs = process.communicate()
            # intvl_nb = int(errs.splitlines()[2].split('=')[1])
            # # print(f"{intvl_nb}\t{self.cmplxty.subTreeSize()}")
            # self.cmplxty.descartes(intvl_nb)
            # # self.cmplxty.leaves()
        
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
