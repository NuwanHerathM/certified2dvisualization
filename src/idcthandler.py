import numpy as np
from math import cos, pi, floor
from scipy.fft import idct

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
    
    def aux1(self):
        p = np.polynomial.Polynomial(self.coef)
        cos1 = [cos((2*i+1)*pi/(2 * self.n1)) for i in range(self.n1)]
        t1 = [p(e) for e in cos1]
        q1 = np.polynomial.chebyshev.poly2cheb(self.coef)
        u1 = [(self.n1 * x + (q1[0] / 2)) for x in idct(q1, n=self.n1)]
        
        cos2 = [x for x in (cos((2*i+1)*pi/(2 * self.n2)) for i in range(self.n2)) if abs(x) > 1/self.alpha] # From Python 3.8 this could be written with an assignment expression
        pow2 = np.array([e**self.d for e in cos2])
        q2 = np.polynomial.chebyshev.poly2cheb(self.coef[::-1])
        u2 = [(self.n2 * x + (q2[0] / 2)) for  x in idct(q2, n=self.n2)]
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

        return idct(self.coef, n=self.n)
    
    def getResult(self):
        self.n1 = floor(self.n / self.alpha)
        self.n2 = self.n - self.n1
        if (self.n1 >= self.d and self.n2 >= self.d):
            return self.aux1()
        else:
            print("...")
    
    def getGrid(self):
        return self.grid
