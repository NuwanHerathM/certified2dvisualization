import numpy as np
from math import cos, pi, floor, acos, ceil
from scipy.fft import idct
from codetiming import Timer
import itertools
from scipy.special import comb

class IDCTHandler:

    def __init__(self, coef, n, lower, upper):
        self.coef = coef
        self.d = len(coef)
        self.n = n
        self.lower = lower
        self.upper = upper
        self.c = (lower + upper) / 2
        self.alpha = (upper - lower) / 2
        self.grid = None
        self.deg_ch = 0
        self.deg_conv = 0
    
    def aux(self):
        cos_z = []
        u_z = []
        if (self.minus != self.plus):
            i_min = ceil(self.N_z / pi * acos(self.plus) - 0.5)
            i_max = ceil(self.N_z / pi * acos(self.minus) - 0.5)
            cos_z = [cos((2 * i + 1) * pi / (2 * self.N_z)) for i in range(i_min, i_max + 1)]
            with Timer("conversion", logger=None):
                q_z = np.polynomial.chebyshev.poly2cheb(self.coef)
            with Timer("evaluation", logger=None):
                u_z = [(self.N_z * x + (q_z[0] / 2)) for x in idct(q_z, n=self.N_z)][i_min:i_max+1]
        
        with Timer("change", logger=None):
            inv_coef = self.coef[::-1]
        with Timer("conversion", logger=None):
            q_inv = np.polynomial.chebyshev.poly2cheb(inv_coef)

        inv_cos_m = []
        t_m = []
        if (self.lower != self.minus):
            i_min = ceil(self.N_m / pi * acos(1 / self.lower) - 0.5)
            i_max = self.N_m - 1
            cos_m = [cos((2 * i + 1) * pi / (2 * self.N_m)) for i in range(i_min, i_max + 1)]
            inv_cos_m = list(map(lambda x: 1 / x, cos_m))
            inv_cos_m.reverse()
            with Timer("evaluation", logger=None):
                u_m = [(self.N_m * x + (q_inv[0] / 2)) for x in idct(q_inv, n=self.N_m)][i_min:i_max+1]
            with Timer("change", logger=None):
                pow_m = np.array([e**self.d for e in cos_m])
                t_m = np.divide(u_m, pow_m)
        
        inv_cos_p = []
        t_p = []
        if (self.plus != self.upper):
            i_min = 0
            i_max = floor(self.N_p / pi * acos(1 / self.upper) - 0.5)
            cos_p = [cos((2 * i + 1) * pi / (2 * self.N_p)) for i in range(i_min, i_max + 1)]
            inv_cos_p = list(map(lambda x: 1 / x, cos_p))
            inv_cos_p.reverse()
            with Timer("evaluation", logger=None):
                u_p = [(self.N_p * x + (q_inv[0] / 2)) for x in idct(q_inv, n=self.N_p)][i_min:i_max+1]
            with Timer("change", logger=None):
                pow_p = np.array([e**self.d for e in cos_p])
                t_p = np.divide(u_p, pow_p)
        
        res = np.concatenate((np.flip(t_m), u_z, np.flip(t_p)))
        self.grid = inv_cos_m + cos_z + inv_cos_p

        return res
    
    def getResult(self):
        self.plus = min(max(self.lower, 1), self.upper)
        self.minus = max(self.lower, min(-1, self.upper))

        self.n_m = round((self.minus - self.lower) / (self.upper - self.lower) * self.n)
        self.n_p = round((self.upper - self.plus) / (self.upper -self.lower) * self.n)
        self.n_z = self.n - self.n_m - self.n_p

        b = True

        if (self.minus != self.plus):
            N_min = round((self.n_z - 1) * pi / (acos(self.minus) - acos(self.plus)))
            N_max = round((self.n_z - 1 + 2) * pi / (acos(self.minus) - acos(self.plus)))

            self.N_z = int(N_min)
            while (self.N_z <= N_max):
                if (floor(self.N_z / pi * acos(self.minus) - 0.5) - ceil(self.N_z / pi * acos(self.plus) - 0.5) == self.n_z - 1):
                    break
                self.N_z += 1
            
            b = b and (self.N_z >= self.d)
        
        if (self.lower != self.minus):
            N_min = round((self.n_m - 0.5) * pi / (pi - acos(1 / self.lower)))
            N_max = round((self.n_m + 0.5) * pi / (pi - acos(1 / self.lower)))

            self.N_m = int(N_min)
            while (self.N_m <= N_max):
                if (self.N_m - ceil(self.N_m / pi * acos(1 / self.lower) - 0.5) == self.n_m):
                    break
                self.N_m += 1
            
            b = b and (self.N_m >= self.d)
        
        if (self.plus != self.upper):
            N_min = round((self.n_p - 0.5) * pi / acos(1 / self.upper))
            N_max = round((self.n_p + 0.5) * pi / acos(1 / self.upper))

            self.N_p = int(N_min)
            while (self.N_p <= N_max):
                if (floor(self.N_p / pi * acos(1 / self.upper) - 0.5) + 1 == self.n_p):
                    break
                self.N_p += 1
            
            b = b and (self.N_p >= self.d)
        
        assert b, "Not enough points to subdivide the interval along the x-axis for the change of basis"

        return self.aux()
    
    def getGrid(self):
        return self.grid

    def getDegrees(self):
        return (self.deg_ch, self.deg_conv)