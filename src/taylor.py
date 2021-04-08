import argparse
import os
from sys import base_prefix

import numpy as np
from math import cos, pi, factorial, isclose
import scipy.fftpack as fp
from utils_taylor import vanishes
import flint as ft
from scipy.special import comb
from visu_utils import comb2D, factorial2D, Verbose

import matplotlib.pyplot as plt
from matplotlib import collections as mc

from codetiming import Timer

# Parse the input

np.seterr(all='raise')
dirname = os.path.dirname(__file__)
default_file = os.path.join(dirname, '../polys/unit_circle.poly')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of subdivision intervals)")
parser.add_argument('-poly', type=str, default=default_file, help="file of polynomial coefficients")
parser.add_argument('-m', type=int, default=3, help="precision of the approximation")
parser.add_argument('-elliptic', help="use elliptic coefficients for the polynomial", action="store_true")
parser.add_argument('-flat', help="use flat coefficients for the polynomial", action="store_true")
parser.add_argument('-v', '--verbose', help="turn on the verbosity", action="store_true")

args = parser.parse_args()

n = args.n + 1 # number of points
m = args.m

# Set function for verbosity
Verbose.classInit(args.verbose)

# Read the polynomial

Verbose.verboseprint("Reading the polynomial file...")
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

if args.elliptic:
    poly = np.multiply(poly, np.sqrt(comb2D(deg_x + 1, deg_y + 1)))
if args.flat:
    poly = np.multiply(poly, 1/np.sqrt(factorial2D(deg_x + 1, deg_y + 1)))

grid = np.array([cos((2 * i + 1) * pi / (2 *n)) for i in range(0, n)])

def corrected_idct(poly, n):
    return np.array([(x + poly[0]) / 2 for x in fp.idct(poly, n=n)])


p = np.empty((n,deg_y+1))
with Timer("conversion", logger=None):
    for d in range(deg_y+1):
        _p = np.polynomial.chebyshev.poly2cheb(poly[:, d].flat)
        p[:, d] = corrected_idct(_p, n)

d = len(p[0]) - 1
if d < m:
    print(f"m={m} is greater than the degree.")
    m = d -1
    print(f"Its value has been set to {m}.")
p_der = np.zeros((m+1,n))

hockeystick = comb(d+1, m+2)
radii = np.empty(n)
with Timer("radii", logger=None):
    for i in range(n):
        r_left = -(grid[i] - grid[i-1]) / 2 if 0 < i else 0
        r_right = -(grid[i+1] - grid[i]) / 2 if i < n - 1 else 0
        radii[i] = max(r_left, r_right)
with Timer("ogf", logger=None):
    ogf = 1 / (1 - grid - radii)
    ogf[0] = hockeystick + 1
    ogf[-1] = hockeystick + 1
radii_power = radii**(m+1)
intervals = np.empty(n, dtype="object")
Verbose.verboseprint("Evaluation...")
for i in range(n):
    indices = []
    with Timer("conversion 2", logger=None):
        _p = np.polynomial.chebyshev.poly2cheb(p[i])
    factor = radii_power * max(p[i], key=abs)
    with Timer("approximation", logger=None):
        for k in range(m+1):
            tmp = np.polynomial.chebyshev.chebder(_p, k)
            p_der[k,:] = 1/factorial(k) * corrected_idct(tmp, n)
    for j in range(n):
        with Timer("isolation", logger=None):
            b = min(ogf[j], hockeystick) * factor[j]
            # np.dot(p_der[:,j],radii[j]**np.arange(m+1))
            if 0 in ft.arb_poly(p_der[:,j].tolist())(ft.arb(0,radii[j])) + ft.arb(0,b):
                indices.append((i,j))
    intervals[i] = indices
timers = Timer.timers
print(f"""radii: {timers['radii']}
ogf: {timers['ogf']}
conversion (x): {timers['conversion']}
conversion (y): {timers['conversion 2']}
approximation: {timers['approximation']}
isolation: {timers['isolation']}""")

# Show isolated intervals

Verbose.verboseprint("Constructing the visualization...")
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
        
