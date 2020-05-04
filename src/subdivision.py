import argparse
import os

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import flint as ft

from scipy.fft import idct
from math import cos, pi
from scipy.special import comb
import itertools

from codetiming import Timer

import logging

# Takes care of the decorators set for the profiling
try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x   # if it's not defined simply ignore the decorator.

@profile
def subdivide(val, low, up, poly,r):
    min = val[low]
    max = val[up]
    mid = int(low + (up - low) / 2)
    median = (min + max) / 2
    radius = (max - min) / 2

    a = ft.arb_poly(poly)(ft.arb(median,radius))
    if 0 in a:
        if r < 1:
            return [(low, up)]
        return subdivide(val, low, mid, poly, r - 1) + subdivide(val, mid, up, poly, r - 1)
    else:
        return []

@profile
def isolateIntervals(poly, n, intervals, switch):
    partial_poly = np.empty((n, deg_y + 1), dtype=object)
    rad = (ys[-1] - ys[0]) / (2 * (n - 1))
    rad = 0
    a = xs[0]
    b = xs[-1]
    alpha = (xs[-1] - xs[0]) / 2
    shift = xs[-1] - alpha
    for j in range(deg_y + 1):
        if (switch == 1):
            # if we want to use Clenshaw with the Chebyshev basis
            with Timer("conversion", logger=None):
                tmp = np.polynomial.Polynomial(poly[j]).convert(kind=np.polynomial.Chebyshev)
            with Timer("change", logger=None):
                tmp.convert(domain=[a, b], window=[a, b])
            # TO DO: sqrt of the degree
            q = tmp.coef
            q[np.abs(q) < 1e-15] = 0
            for i in range(n):
                q = np.trim_zeros(q, 'b')
                if (len(q) == 0):
                    q = [0]
                with Timer("evaluation", logger=None):
                    partial_poly[i,j] = np.polynomial.chebyshev.chebval(xs[i], q)
        elif (switch == 0):
            # if we do not use the Chebyshev basis
            tmp = poly[j]
            for i in range(n):
                with Timer("evaluation", logger=None):
                    partial_poly[i,j] = np.polynomial.polynomial.polyval(xs[i], tmp)
        else:
            #if we want to use the IDCT with the Chebyshev basis
            with Timer("change", logger=None):
                s = [sum(x) for x in itertools.zip_longest(*[[comb(i, k) * alpha ** k * shift ** (i - k) * c for k in range(i + 1)] for i, c in enumerate(poly[j])], fillvalue=0)]
            # s = sum([np.polynomial.Polynomial([shift, alpha])**i * c for i, c in enumerate(poly[j])])
            with Timer("conversion", logger=None):
                tmp = np.polynomial.chebyshev.poly2cheb(s)
            # tmp = np.zeros(deg_y + 1)
            # tmp[:len(c)] = c
            with Timer("evaluation", logger=None):
                partial_poly[:,j] = [(n * x + (tmp[0] / 2)) for x in idct(tmp, n=n)]
    for i in range(n):
        with Timer("subdivision", logger=None):
            intervals[i] = subdivide(ys, 0, n - 1, partial_poly[i].tolist(), d)


# Parse the input

dirname = os.path.dirname(__file__)
default_file = os.path.join(dirname, '../polys/circle.poly')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of subdivisions)")
parser.add_argument('-poly', type=str, default=default_file, help="file of polynomial coefficients")
parser.add_argument('-x', nargs=2, type=int, default=[-8,8], help="bounds on the x-axis")
parser.add_argument('-y', nargs=2, type=int, default=[-8,8], help="bounds on the y-axis")
parser.add_argument('-clen', nargs='?', type=int, const=1, default=0, help="use the Chebyshev basis and Clenshaw's scheme")
parser.add_argument('-idct', nargs='?', type=int, const=2, default=0, help="use the Chebyshev basis and the IDCT")
parser.add_argument('-hide', nargs='?', type=bool, const=True, default=False, help="hide the plot")

args = parser.parse_args()

n = args.n + 1 # number of points
d = math.floor(np.log2(n))
xs = np.linspace(args.x[0], args.x[1], n)
ys = np.linspace(args.y[0], args.y[1], n)
max_dy = ys[1] - ys[0]
use_clen = args.clen
use_idct = args.idct

# Read the polynomial

with open(args.poly) as inf:
    lines = inf.readlines()
    deg_y = len(lines) - 1
    if (deg_y > -1):
        deg_x = len(lines[0].split(" ")) - 1
    else:
        print("Empty file.")
        exit()

poly = np.empty((deg_y + 1, deg_x + 1))

with open(args.poly) as inf:
    lines = inf.readlines()
    for i in range(deg_y + 1):
        for j in range(deg_x + 1):
            poly[i,j] = lines[i].split(" ")[j]

# Core of the program

intervals = np.empty(n, dtype="object")
isolateIntervals(poly, n, intervals, use_clen + use_idct)

# Logging

SORT_ORDER = {"conversion": 0, "change": 1, "evaluation": 2, "subdivision": 3}
sorted_dict = sorted(Timer.timers.items(), key=lambda x: SORT_ORDER[x[0]])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('subdivision_time.log', 'w')
handler_format = logging.Formatter('%(message)s')
logger.addHandler(handler)

for key, value in sorted_dict:
    logger.info(f"{key}\t{value}")

# Show isolated intervals

if (not args.hide):
    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111, aspect='equal')

    alpha = (xs[-1] - xs[0]) / 2
    shift = xs[-1] - alpha
    for i in range(n):
        for e in intervals[i]:
            if (use_idct):
                x = cos((2*i+1)*pi/(2 * n)) * alpha + shift
                plt.plot([x, x], [ys[e[0]], ys[e[1]]], '-k')
            else:
                plt.plot([xs[i], xs[i]], [ys[e[0]], ys[e[1]]], '-k')

    plt.xlim(xs[0],xs[-1])
    plt.ylim(ys[0],ys[-1])
    if (args.poly == default_file):
        # draw a circle if the default file is used
        circle = plt.Circle((0, 0), 2, color='r', fill=False)
        ax1.add_artist(circle)
    plt.show()
