import numpy as np
from math import cos, pi
from codetiming import Timer
import config as cfg

class IDCTHandler:

    def __init__(self, coef, n):
        """
            Constructor.

            Parameters
            ----------
            coef : array
                List of the polynomial coefficients
            n : int
                Number of points in the output
            lower : float
                Lower bound of the output range
            upper : float
                Upper bound of the output range
        """
        self.coef = coef
        self.d = len(coef)
        self.n = n

    def getResult(self, grid, idct_choice='interval', poly2cheb_choice='interval'):
        r_z = []
        if (grid.lower_x != grid.upper_x):
            with Timer("conversion", logger=None):
                p_z = cfg.functions['poly2cheb'][0](np.array(self.coef))
            with Timer("evaluation", logger=None):
                r_z = (grid.zero['N_x'] * cfg.functions['idct'][0](p_z, n=grid.zero['N_x']) + p_z[0] / 2)[grid.zero['i_min_x']:grid.zero['i_max_x']+1]

        with Timer("change", logger=None):
            inv_coef = self.coef[::-1]
        with Timer("conversion", logger=None):
            p_inv = cfg.functions['poly2cheb'][0](np.array(inv_coef))

        r_m = []
        if (grid.x_min != grid.lower_x):
            cos_m = [cos((2 * i + 1) * pi / (2 * grid.minus['N_x'])) for i in range(grid.minus['i_min_x'], grid.minus['i_max_x'] + 1)]
            with Timer("evaluation", logger=None):
                p_m = (grid.minus['N_x'] * cfg.functions['idct'][0](p_inv, n=grid.minus['N_x']) + p_inv[0] / 2)[grid.minus['i_min_x']:grid.minus['i_max_x']+1]
            with Timer("change", logger=None):
                pow_m = np.array([e**self.d for e in cos_m])
                r_m = np.divide(p_m, pow_m)

        r_p = []
        if (grid.upper_x != grid.x_max):
            cos_p = [cos((2 * i + 1) * pi / (2 * grid.plus['N_x'])) for i in range(grid.plus['i_min_x'], grid.plus['i_max_x'] + 1)]
            with Timer("evaluation", logger=None):
                p_p = (grid.plus['N_x'] * cfg.functions['idct'][0](p_inv, n=grid.plus['N_x']) + p_inv[0] / 2)[grid.plus['i_min_x']:grid.plus['i_max_x']+1]
            with Timer("change", logger=None):
                pow_p = np.array([e**self.d for e in cos_p])
                r_p = np.divide(p_p, pow_p)

        res = np.concatenate((np.flip(r_m), r_z, np.flip(r_p)))

        return res

    # def getDegrees(self):
    #     return (self.deg_ch, self.deg_conv)
