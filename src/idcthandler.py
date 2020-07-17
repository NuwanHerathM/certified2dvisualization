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
        self.deg_s = 0
        self.deg_ch = 0
        self.deg_tmp = 0
        self.deg_conv = 0
    
    def noDCT(self):
        with Timer("change", logger=None):
            s = [sum(x) for x in itertools.zip_longest(*[[comb(i, k) * self.alpha ** k * self.c ** (i - k) * c for k in range(i + 1)] for i, c in enumerate(self.coef)], fillvalue=0)]
        # TO DO: correct computation when the polynomial is of degree 0 (it returns -1 for deg_ch and deg_conv)
        self.deg_s = len(s) - 1
        self.deg_ch = len(np.trim_zeros(s, 'b')) - 1
        with Timer("conversion", logger=None):
            tmp = np.polynomial.chebyshev.poly2cheb(s)
        self.deg_tmp = len(tmp) - 1
        self.deg_conv = len(np.trim_zeros(tmp, 'b')) - 1
        with Timer("evaluation", logger=None):
            res = [(self.n * x + (tmp[0] / 2)) for x in idct(tmp, n=self.n)]
        return res

    def aux1(self):
        p = np.polynomial.Polynomial(self.coef)
        cos1 = [cos((2*i+1)*pi/(2 * self.n1)) for i in range(self.n1)]
        t1 = [p(e) for e in cos1]
        q1 = np.polynomial.chebyshev.poly2cheb(self.coef)
        u1 = [(self.n1 * x + (q1[0] / 2)) for x in idct(q1, n=self.n1)]
        
        n2 = ceil((self.n2 - 1) /  2 * pi / acos(1/self.alpha))
        cos2 = [x for x in (cos((2*i+1)*pi/(2 * n2)) for i in range(n2)) if abs(x) >= 1/self.alpha] # From Python 3.8 this could be written with an assignment expression
        pow2 = np.array([e**self.d for e in cos2])
        q2 = np.polynomial.chebyshev.poly2cheb(self.coef[::-1])
        u2 = [(n2 * x + (q2[0] / 2)) for  x in idct(q2, n=n2)]
        l_u = len(u2)
        l_pow = len(pow2)
        for i in range(l_u - l_pow):
            u2.pop(l_pow // 2)
        t2 = np.divide(u2,pow2)
 
        (t2_1,t2_2) = np.split(t2,[l_pow // 2])
        res = np.concatenate((np.flip(t2_1),u1,np.flip(t2_2)))

        cos2_1 = cos2[0:l_pow // 2]
        cos2_1.reverse()
        inv_cos_2_1 = [1/x for x in cos2_1]
        
        cos2_2 = cos2[l_pow // 2:]
        cos2_2.reverse()
        inv_cos_2_2 = [1/x for x in cos2_2]
        self.grid = inv_cos_2_1 + cos1 + inv_cos_2_2

        return res
    
    def getResult(self):
        self.n1 = floor(self.n / self.alpha)
        # guarantee that self.n2 is even
        if ((self.n - self.n1) % 2 == 1):
            self.n1 += 1
        self.n2 = self.n - self.n1
        if (self.n1 >= self.d and self.n2 >= self.d):
            return self.aux1()
        else:
            return self.noDCT()
    
    def getGrid(self):
        return self.grid

    def getDegrees(self):
        return (self.deg_s, self.deg_ch, self.deg_tmp, self.deg_conv)