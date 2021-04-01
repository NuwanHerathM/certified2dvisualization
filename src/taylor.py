import argparse
import os
import numpy as np
from math import cos, pi, factorial, isclose
import scipy.fftpack as fp
from utils_taylor import vanishes
import flint as ft
import matplotlib.pyplot as plt
from matplotlib import collections as mc

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

d = len(p[0]) - 1
p_der = np.zeros((m,n))

intervals = np.empty(n, dtype="object")
for i in range(n):
    indices = []
    _p = np.polynomial.chebyshev.poly2cheb(p[i])
    a = max(p[i], key=abs)
    for k in range(m):
        tmp = np.polynomial.chebyshev.chebder(_p, k)
        p_der[k,:] = 1/factorial(k) * corrected_idct(tmp, n)
    for j in range(n):
        if vanishes(ft.arb_poly(p_der[:,j].tolist()), a, grid, j, m, d):
            indices.append((i,j))
    intervals[i] = indices

# Show isolated intervals

fig1 = plt.figure(dpi=600)

ax1 = fig1.add_subplot(111, aspect='equal')
ax1.tick_params(axis='both', which='minor', labelsize=10)

interval_lut = [(grid[i+1] + grid[i]) / 2 for i in range(n-1)]
interval_lut.insert(0, 1)
interval_lut.append(-1)
segments = []
for i in range(n):
    for e in intervals[i]:
        segments.append([(grid[e[0]], interval_lut[e[1]]), (grid[e[0]], interval_lut[e[1]+1])])

lc = mc.LineCollection(segments, linewidths=0.1)
ax1.add_collection(lc)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()
        
