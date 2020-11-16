from locale import NOEXPR
import numpy as np
from scipy.fftpack import idctn, idct
from math import isclose, cos, pi


class IDCTNHandler:
    """Class handling the multidimensional IDCT of a polynomial."""

    # def __init__(self, poly, n):
    #     self.poly = poly
    #     self.n = n
    #     self.shape = poly.shape

    @staticmethod
    def multipoly2cheb(poly, d):
        l = []
        for p in map(np.polynomial.chebyshev.poly2cheb, poly):
            z = np.zeros(d)
            z[:p.shape[0]] = p
            l.append(z)
        return l

    def __correctIDCT(self, c, n):
        return [(n * x + (c[0] / 2)) for x in idct(c, n=n)]

    def __eval(self, poly, shape):

        out = idctn(poly, shape=shape)

        idct_i = self.__correctIDCT(poly[:, 0], shape[0])
        idct_j = self.__correctIDCT(poly[0, :], shape[1])

        for i in range(shape[0]):
            out[i] += idct_j
        for j in range(shape[1]):
            out[:, j] += idct_i
        
        # out_border_x = np.tile(np.vstack(idct_i), (1, shape[1]))
        # out_border_y = np.tile(idct_j, (shape[0], 1))

        return out

    def idct2d(self, poly, shape):
        """Compute the 2D IDCT of the polynomial."""

        out = self.aux(poly, shape)
        return out

    def aux(self, poly, shape):
        border_x = np.zeros(poly.shape)
        border_y = np.zeros(poly.shape)
        corner = np.zeros(poly.shape)
        inside = np.zeros(poly.shape)

        tmp_i = np.stack(IDCTNHandler.multipoly2cheb(poly, poly.shape[0]))
        tmp_j = np.stack(IDCTNHandler.multipoly2cheb(tmp_i.transpose(), poly.shape[1])).transpose()

        border_y[0, 1:] = tmp_j[0, 1:]
        border_x[1:, 0] = tmp_j[1:, 0]
        corner[0, 0] = tmp_j[0, 0]
        inside[1:, 1:] = tmp_j[1:, 1:]

        out_border_x = self.__eval(border_x, shape)
        out_border_y = self.__eval(border_y, shape)
        out_corner = self.__eval(corner, shape)
        out_inside = self.__eval(inside, shape)

        out = out_border_x / (2 * (shape[0] + 1)) + out_border_y / (2 * (shape[1] + 1)) + out_corner / (shape[0] + shape[1] + 2) + out_inside / 4  # 4 = 2 ** 2 = 2 ** nb_dim, probably...

        return out

    # def idct2d(self):

    #     pxy = self.poly
    #     px_ = np.flip(self.poly, axis=1)
    #     p_y = np.flip(self.poly, axis=0)
    #     p__ = np.flip(self.poly, axis={0, 1})

    #     # pxy
    #     r_zz = self.aux(pxy)

    #     # px_
    #     r_zm
    #     r_zp

    #     # p_y
    #     r_mz
    #     r_mp

    #     # p__
    #     r_mm
    #     r_mp
    #     r_pm
    #     r_pp

if __name__ == "__main__":
    print("hello")
    n = 100
    d = 100

    poly = np.random.randn(d, d)

    id2t = IDCTNHandler()
    out = id2t.idct2d(poly, (n,n+1))

    nodes = [cos((2 * i + 1) * pi / (2 * n)) for i in range(n)]
    nodes_ = [cos((2 * i + 1) * pi / (2 * (n+1))) for i in range(n+1)]

    eval = np.polynomial.polynomial.polygrid2d(nodes, nodes_, poly)

    for i in range(n):
        for j in range(n+1):
            if not isclose(out[i, j], eval[i, j]):
                # print("oups")
                pass
    
    poly = np.array([[0,1,0],[0,0,0],[0,0,0]])
    n = 5
    m = 4

    def correctIDCT(c, n):
        return [(n * x + (c[0] / 2)) for x in idct(c, n=n)]
    
    out = idctn(poly, shape=(n,m))
    idct_i = correctIDCT(poly[:, 0], n)
    idct_j = correctIDCT(poly[0, :], m)
    for i in range(n):
        out[i] += idct_j
    for j in range(m):
        out[:, j] += idct_i
    print(out/10)
    print(idct_i)
    print(idct_j)
    nodes = [cos((2 * i + 1) * pi / (2 * n)) for i in range(n)]
    nodes_ = [cos((2 * i + 1) * pi / (2 * m)) for i in range(m)]

    eval = np.polynomial.chebyshev.chebgrid2d(nodes, nodes_, poly)
    print(eval)

    # for i in range(p.ndim):
    #     p[(slice(None),)*i + (0,)] *= 2
    # # Bug in fftpack : not accepting array for shape
    # V = fp.idctn(p, shape=tuple(N))/2**p.ndim