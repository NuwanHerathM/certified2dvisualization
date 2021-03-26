import argparse
import os
import numpy as np
from math import cos, pi, factorial, isclose
import scipy.fftpack as fp
from utils_taylor import vanishes
import flint as ft

# Parse the input

np.seterr(all='raise')
dirname = os.path.dirname(__file__)
default_file = os.path.join(dirname, '../polys/unit_circle.poly')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of subdivision intervals)")
parser.add_argument('-poly', type=str, default=default_file, help="file of polynomial coefficients")
parser.add_argument('-m', type=int, default=3, help="precision of the approximation")

args = parser.parse_args()

n = args.n + 1 # number of points
m = args.m

# Read the polynomial

with open(args.poly) as inf:
    lines = inf.readlines()
    deg_x = len(lines) - 1
    if (deg_x > -1):
        deg_y = len(lines[0].split(" ")) - 1
    else:
        print("Empty file.")
        exit()

poly = np.loadtxt(args.poly, dtype=int)

# Core of the program

grid = [cos((2 * i + 1) * pi / (2 *n)) for i in range(0, n)]

def corrected_idct(poly, n):
    return np.array([(x + poly[0]) / 2 for x in fp.idct(poly, n=n)])

p = np.empty((n,deg_y+1))
for d in range(deg_y+1):
    _p = np.polynomial.chebyshev.poly2cheb(poly[:, d])
    p[:, d] = corrected_idct(_p, n)


# q = np.empty((n,n))
# for i in range(n):
#     for j in range(n):
#         q[i,j] = np.polynomial.polynomial.polyval2d(grid[i], grid[j], poly)

d = len(p[0]) - 1
p_der = np.zeros((m,n))

l = []
for i in range(n):
    _p = np.polynomial.chebyshev.poly2cheb(p[i])
    a = max(p[i], key=abs)
    for k in range(m):
        tmp = np.polynomial.chebyshev.chebder(_p, k)
        p_der[k,:] = 1/factorial(k) * corrected_idct(tmp, n)
    for j in range(n):
        # if not isclose(p_der[0, j], q[i,j]):
        #         print(f"{p_der[0, j]} {q[i, j]}")
        #         print("oups")
        if vanishes(ft.arb_poly(p_der[:,j].tolist()), a, grid, j, m, d):
            l.append((i,j))

print(l)
        
