from locale import NOEXPR
import numpy as np
from scipy.fftpack import idctn, idct
from math import isclose, cos, pi


class IDCTNHandler:
    """Class handling the multidimensional IDCT of a polynomial."""

    def __init__(self, poly, grid):
        self.poly = poly
        self.grid = grid
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

    @staticmethod
    def correctIDCT(c, n):
        return np.array([(n * x + (c[0] / 2)) for x in idct(c, n=n)])

    @staticmethod
    def eval(poly, shape):

        poly_i = np.copy(poly[:, 0])
        poly_i[0] = 0
        idct_i = IDCTNHandler.correctIDCT(poly_i, shape[0])
        poly_j = np.copy(poly[0, :])
        poly_j[0] = 0
        idct_j = IDCTNHandler.correctIDCT(poly_j, shape[1])

        out_border_x = np.tile(np.vstack(idct_i/(2*shape[0])),(1,shape[1]))
        out_border_y = np.tile(idct_j/(2*shape[1]),(shape[0],1))
        out_corner = np.tile(poly[0,0],shape)
        
        inside = np.zeros(poly.shape)
        inside[1:,1:] = poly[1:,1:]
        out_inside = idctn(inside, shape=shape)
        
        out = out_border_x + out_border_y + out_corner + out_inside / 4  # 4 = 2 ** 2 = 2 ** nb_dim, probably...

        return out

    def idct2d(self, poly, shape):
        """Compute the 2D IDCT of the polynomial."""

        out = self.aux(poly, shape)
        return out

    @staticmethod
    def aux(poly, shape):
        tmp_i = np.stack(IDCTNHandler.multipoly2cheb(poly, poly.shape[0]))
        tmp_j = np.stack(IDCTNHandler.multipoly2cheb(tmp_i.transpose(), poly.shape[1])).transpose()

        out = IDCTNHandler.eval(tmp_j, shape)
        
        return out

    @staticmethod
    def getDim(dict_x, dict_y):
        if dict_x.get('n_x'):
            d_x = dict_x['n_x']
        else:
            d_x = 0
        if dict_y.get('n_y'):
            d_y = dict_y['n_y']
        else:
            d_y = 0
        return (d_x, d_y)

    def idct2d_improved(self):

        pxy = self.poly
        px_ = np.flip(self.poly, axis=1)
        p_y = np.flip(self.poly, axis=0)
        p__ = np.flip(self.poly, axis={0, 1})

        # pxy
        shape = IDCTNHandler.getDim(self.grid.zero, self.grid.zero)
        r_zz = np.empty(shape, dtype=object)
        if (self.grid.lower_x != self.grid.upper_x and self.grid.lower_y != self.grid.upper_y):
            tmp_zz = self.aux(pxy, (self.grid.zero['N_x'], self.grid.zero['N_y']))
            r_zz = tmp_zz[self.grid.zero['i_min_x']:self.grid.zero['i_max_x']+1,self.grid.zero['i_min_y']:self.grid.zero['i_max_y']+1]

        # px_
        shape = IDCTNHandler.getDim(self.grid.zero, self.grid.minus)
        r_zm = np.empty(shape, dtype=object)
        if (self.grid.lower_x != self.grid.upper_x and self.grid.y_min != self.grid.lower_y):
            tmp_zm = IDCTNHandler.aux(px_, (self.grid.zero['N_x'],self.grid.minus['N_y']))
            r_zm = tmp_zm[self.grid.zero['i_min_x']:self.grid.zero['i_max_x']+1,self.grid.minus['i_min_y']:self.grid.minus['i_max_y']+1]
            # cos_m = [cos((2 * i + 1) * pi / (2 * self.grid.minus['N_y'])) for i in range(self.grid.minus['i_min_y'], self.grid.minus['i_max_y'] + 1)]
            # d = px_.shape[1]
            # pow_m = np.array([e**d for e in cos_m])
            # cos_zm = np.tile(pow_m, (shape[0],1))
            # print(r_zm)
            # r_zm = np.divide(r_zm, cos_zm)
            # print()
            # print(r_zm)

        shape = IDCTNHandler.getDim(self.grid.zero, self.grid.plus)
        r_zp = np.empty(shape, dtype=object)
        if (self.grid.lower_x != self.grid.upper_x and self.grid.upper_y != self.grid.y_max):
            tmp_zp = IDCTNHandler.aux(px_, (self.grid.zero['N_x'],self.grid.plus['N_y']))
            r_zp = tmp_zp[self.grid.zero['i_min_x']:self.grid.zero['i_max_x']+1,self.grid.plus['i_min_y']:self.grid.plus['i_max_y']+1]

        # p_y
        shape = IDCTNHandler.getDim(self.grid.minus, self.grid.zero)
        r_mz = np.empty(shape, dtype=object)
        if (self.grid.y_min != self.grid.lower_x and self.grid.lower_y != self.grid.upper_y):
            tmp_mz = IDCTNHandler.aux(p_y, (self.grid.minus['N_x'],self.grid.zero['N_y']))
            r_mz = tmp_mz[self.grid.minus['i_min_x']:self.grid.minus['i_max_x']+1,self.grid.zero['i_min_y']:self.grid.zero['i_max_y']+1]
        shape = IDCTNHandler.getDim(self.grid.plus, self.grid.zero)
        r_pz = np.empty(shape, dtype=object)
        if (self.grid.upper_x != self.grid.x_max and self.grid.lower_y != self.grid.upper_y):
            tmp_pz = IDCTNHandler.aux(p_y, (self.grid.plus['N_x'],self.grid.zero['N_y']))
            r_pz = tmp_pz[self.grid.plus['i_min_x']:self.grid.plus['i_max_x']+1,self.grid.zero['i_min_y']:self.grid.zero['i_max_y']+1]

        # p__
        shape = IDCTNHandler.getDim(self.grid.minus, self.grid.minus)
        r_mm = np.empty(shape, dtype=object)
        if (self.grid.y_min != self.grid.lower_x and self.grid.lower_y != self.grid.upper_y):
            tmp_mm = IDCTNHandler.aux(p__, (self.grid.minus['N_x'],self.grid.minus['N_y']))
            r_mm = tmp_mm[self.grid.minus['i_min_x']:self.grid.minus['i_max_x']+1,self.grid.minus['i_min_y']:self.grid.minus['i_max_y']+1]
        shape = IDCTNHandler.getDim(self.grid.minus, self.grid.plus)
        r_mp = np.empty(shape, dtype=object)
        if (self.grid.y_min != self.grid.lower_x and self.grid.lower_y != self.grid.upper_y):
            tmp_mp = IDCTNHandler.aux(p__, (self.grid.minus['N_x'],self.grid.plus['N_y']))
            r_mp = tmp_mp[self.grid.minus['i_min_x']:self.grid.minus['i_max_x']+1,self.grid.plus['i_min_y']:self.grid.plus['i_max_y']+1]
        shape = IDCTNHandler.getDim(self.grid.plus, self.grid.minus)
        r_pm = np.empty(shape, dtype=object)
        if (self.grid.y_min != self.grid.lower_x and self.grid.lower_y != self.grid.upper_y):
            tmp_pm = IDCTNHandler.aux(p__, (self.grid.plus['N_x'],self.grid.minus['N_y']))
            r_pm = tmp_pm[self.grid.plus['i_min_x']:self.grid.plus['i_max_x']+1,self.grid.minus['i_min_y']:self.grid.minus['i_max_y']+1]
        shape = IDCTNHandler.getDim(self.grid.plus, self.grid.plus)
        r_pp = np.empty(shape, dtype=object)
        if (self.grid.y_min != self.grid.lower_x and self.grid.lower_y != self.grid.upper_y):
            tmp_pp = IDCTNHandler.aux(p__, (self.grid.plus['N_x'],self.grid.plus['N_y']))
            r_pp = tmp_pp[self.grid.plus['i_min_x']:self.grid.plus['i_max_x']+1,self.grid.plus['i_min_y']:self.grid.plus['i_max_y']+1]

        res = np.block([[np.flip(r_mm, axis={0, 1}), np.flip(r_mz, axis=0), np.flip(r_mp, axis={0, 1})],
                        [np.flip(r_zm, axis=1), r_zz, np.flip(r_zp, axis=1)],
                        [np.flip(r_pm, axis={0, 1}), np.flip(r_pz, axis=0), np.flip(r_pp, axis={0, 1})]])

        # res = np.block([[r_mm, r_mz, r_mp], [r_zm, r_zz, r_zp], [r_pm, r_pz, r_pp]])

        return res

if __name__ == "__main__":
    n = 5
    d = 3

    poly = np.random.randn(d, d)
    # print(poly)

    id2t = IDCTNHandler()
    out = id2t.idct2d(poly, (n,n+1))

    nodes = [cos((2 * i + 1) * pi / (2 * n)) for i in range(n)]
    nodes_ = [cos((2 * i + 1) * pi / (2 * (n+1))) for i in range(n+1)]

    eval = np.polynomial.polynomial.polygrid2d(nodes, nodes_, poly)

    for i in range(n):
        for j in range(n+1):
            if not isclose(out[i, j], eval[i, j]):
                print("oups")
                pass

    
    poly = np.array([[7,1,0],[-6,5,0],[0,0,8]])
    n = 5
    m = 4

    def correctIDCT(c, n):
        return np.array([(n * x + (c[0] / 2)) for x in idct(c, n=n)])
    

    inside = np.zeros(poly.shape)
    inside[1:,1:] = poly[1:,1:]
    out = idctn(inside, shape=(n,m))
    # print(out)
    # print()
    poly_i = np.copy(poly[:, 0])
    poly_i[0] = 0
    idct_i = correctIDCT(poly_i, n)
    poly_j = np.copy(poly[0, :])
    poly_j[0] = 0
    idct_j = correctIDCT(poly_j, m)
    # for i in range(n):
    #     out[i] += idct_j
    # for j in range(m):
    #     out[:, j] += idct_i


    # print(out)
    print(idct_i)
    print(idct_j)
    print()

    out_border_x = np.tile(np.vstack(idct_i/(2*n)),(1,m))
    out_border_y = np.tile(idct_j/(2*m),(n,1))
    out_corner = np.tile(poly[0,0],(n,m))
    out_inside = out

    print(out_border_x)
    print(out_border_y)
    print(out_corner)
    print(out_inside/4)
    print(out_border_x + out_border_y + out_inside / 4 + out_corner)
    nodes = [cos((2 * i + 1) * pi / (2 * n)) for i in range(n)]
    nodes_ = [cos((2 * i + 1) * pi / (2 * m)) for i in range(m)]

    eval = np.polynomial.chebyshev.chebgrid2d(nodes, nodes_, poly)
    print(eval)

    # for i in range(p.ndim):
    #     p[(slice(None),)*i + (0,)] *= 2
    # # Bug in fftpack : not accepting array for shape
    # V = fp.idctn(p, shape=tuple(N))/2**p.ndim