import argparse
import os
import sys

import math
from math import cos, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from subdivision import Subdivision

from codetiming import Timer

import logging

# Parse the input

dirname = os.path.dirname(__file__)
default_file = os.path.join(dirname, '../polys/circle.poly')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of subdivision intervals)")
parser.add_argument('-poly', type=str, default=default_file, help="file of polynomial coefficients")
parser.add_argument('-x', nargs=2, type=int, default=[-8,8], help="bounds on the x-axis")
parser.add_argument('-y', nargs=2, type=int, default=[-8,8], help="bounds on the y-axis")
parser.add_argument('-clen', nargs='?', type=int, const=1, default=0, help="use the Chebyshev basis and Clenshaw's scheme")
parser.add_argument('-idct', nargs='?', type=int, const=2, default=0, help="use the Chebyshev basis and the IDCT")
parser.add_argument('-hide', nargs='?', type=bool, const=True, default=False, help="hide the plot")
parser.add_argument('-save', nargs='?', type=bool, const=True, default=False, help="save the plot in the output directory")
parser.add_argument('-der', nargs='?', type=bool, const=True, default=False, help="use the derivative as a subdivision termination criterion")
parser.add_argument('-dsc', nargs='?', type=bool, const=True, default=False, help="use Descartes' rule for the subdivision")

args = parser.parse_args()

n = args.n + 1 # number of points
xs = np.linspace(args.x[0], args.x[1], n)
ys = np.linspace(args.y[0], args.y[1], n)
use_clen = args.clen
use_idct = args.idct
use_dsc = args.dsc

# Read the polynomial

with open(args.poly) as inf:
    lines = inf.readlines()
    deg_y = len(lines) - 1
    if (deg_y > -1):
        deg_x = len(lines[0].split(" ")) - 1
    else:
        print("Empty file.")
        exit()

assert deg_x <= n or deg_y <= n, "Not enough points with respect to the degree of the polynomial"

poly = np.loadtxt(args.poly, dtype=int)

# Core of the program

sub = Subdivision(xs, ys, deg_x, deg_y, args.poly, args.der)

intervals = sub.isolateIntervals(poly, n, use_clen + use_idct, use_dsc)

# sub.drawSubdivisions()
# sub.printComplexity()

# Computation time logging

SORT_ORDER = {"change": 0, "conversion": 1, "evaluation": 2, "subdivision": 3}
sorted_dict = sorted(Timer.timers.items(), key=lambda x: SORT_ORDER[x[0]])

time_logger = logging.getLogger("timing")
time_logger.setLevel(logging.INFO)
handler_format = logging.Formatter('%(message)s')
time_handler = logging.FileHandler('subdivision_time.log', 'w')
time_handler.setFormatter(handler_format)
time_logger.addHandler(time_handler)

for key, value in sorted_dict:
    time_logger.info(f"{key}\t{value}")

# Show isolated intervals

def merge(intervals):
    """Return the union of the intervals."""

    iterator = iter(intervals)
    res = []
    last = (-1,-1)
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break  # Iterator exhausted: stop the loop
        else:
            if last[1] == item[0]:
                last = (last[0], item[1])
            else:
                res.append(item)
                last = item
    return res

if (not args.hide or args.save):
    fig1 = plt.figure()
    base = os.path.basename(args.poly)
    method = "clenshaw" * use_clen + "idct" * (use_idct - 1) + "classic" * (1 - max(use_clen, use_idct - 1))
    fig1.suptitle(f"{os.path.splitext(base)[0]}: n={n - 1}, " + method)

    ax1 = fig1.add_subplot(111, aspect='equal')

    alpha = (xs[-1] - xs[0]) / 2
    shift = xs[-1] - alpha
    grid = sub.getGrid()
    for i in range(n):
        for e in merge(intervals[i]):
            if (use_idct):
                x = grid[i]
                plt.plot([x, x], [ys[e[0]], ys[e[1]]], '-k')
            else:
                plt.plot([xs[i], xs[i]], [ys[e[0]], ys[e[1]]], '-k')

    plt.xlim(xs[0],xs[-1])
    plt.ylim(ys[0],ys[-1])
    if (args.poly == default_file):
        # draw a circle if the default file is used
        circle = plt.Circle((0, 0), 2, color='r', fill=False)
        ax1.add_artist(circle)
    
    if args.save:
        filename = os.path.splitext(os.path.basename(args.poly))[0]
        plt.savefig(f"../output/{filename}_{n-1}.png", bbox_inches='tight')
    if not args.hide:
        plt.show()
