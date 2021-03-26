import flint as ft
from math import floor, pi, cos, factorial
from scipy.special import comb

def bound(a, points, i, m, d):
    r_left = ft.arb(points[i] - points[i-1]) / 2 if 0 < i else ft.arb(0)
    r_right = ft.arb(points[i+1] - points[i]) / 2 if i < len(points) - 1 else ft.arb(0)
    r = r_left.max(r_right)
    return a * r**(m+1) * ft.arb.max(1/(1 - ft.arb(points[i]).abs_upper() -r)**(m+2), comb(d+1, m+2))

def vanishes(poly, a, points, i, m, d):
    b = bound(a, points, i, m, d)
    # print(poly(ft.arb(points[i])))
    # print(b)    
    ball = poly(ft.arb(points[i])) + ft.arb(0, b.abs_upper())
    # print(ball)
    return 0 in ball

if __name__ == "__main__":
    import numpy as np
    import scipy.fftpack as fp

    def corrected_idct(poly, n):
        return np.array([(x + poly[0]) / 2 for x in fp.idct(poly, n=n)])

    p_can = [0, 0, 1]
    p_cheb = np.polynomial.chebyshev.poly2cheb(p_can)
    n = 20
    m = 1
    p_der = np.zeros((m+1,n))
    for k in range(m+1):
        tmp = np.polynomial.chebyshev.chebder(p_cheb, k)
        p_der[k,:] = 1/factorial(k) * corrected_idct(tmp, n)
    grid = [cos((2 * i + 1) * pi / (2 *n)) for i in range(0, n)]
    for i in range(n):
        print(vanishes(ft.arb_poly(p_der[:,i].tolist()), 1, grid, i, m, 2))

    print(p_cheb.shape)
    print(p_der[:,10].shape)

    print(grid[10])
    # print(grid[9])
    r = max(grid[11] - grid[10], grid[10] - grid[9]) / 2
    ball = ft.arb_poly(p_der[:,10].tolist())(ft.arb(grid[10],r))
    print(ball)
    print(0 in ball)
    # print(vanishes(ft.arb_poly([0,1]), 1,[0,1,2,3],2,2,1))
    # print(vanishes(ft.arb_poly([0,1]), 1,[0,1,2,3],0,2,1))
