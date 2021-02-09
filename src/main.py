import argparse
import os

import numpy as np
from subdivision import Subdivision
from visualization import Visualization
from grid import Grid

from codetiming import Timer

import logging

from visu_utils import Verbose, comb2D

# Parse the input
np.seterr(all='raise')
dirname = os.path.dirname(__file__)
default_file = os.path.join(dirname, '../polys/circle.poly')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of subdivision intervals)")
parser.add_argument('-poly', type=str, default=default_file, help="file of polynomial coefficients")
parser.add_argument('-x', nargs=2, type=int, default=[-8,8], help="bounds on the x-axis")
parser.add_argument('-y', nargs=2, type=int, default=[-8,8], help="bounds on the y-axis")
group_eval = parser.add_mutually_exclusive_group()
group_eval.add_argument('-clen', help="use the Chebyshev basis and Clenshaw's scheme", action="store_true")
group_eval.add_argument('-idct', help="use the Chebyshev basis and the IDCT", action="store_true")
parser.add_argument('-hide', help="hide the plot", action="store_true")
parser.add_argument('-freq', help="show the subdivision time distribution", action="store_true")
parser.add_argument('-save', help="save the plot in the output directory", action="store_true")
parser.add_argument('-der', help="use the derivative as a subdivision termination criterion", action="store_true")
group_sub = parser.add_mutually_exclusive_group()
group_sub.add_argument('-dsc', help="use Descartes's rule for the subdivision", action="store_true")
group_sub.add_argument('-cs', help="use GM's clenshaw for the isolation", action="store_true")
parser.add_argument('-v', '--verbose', help="turn on the verbosity", action="store_true")
parser.add_argument('-idct2d', help="use the 2D IDCT", action="store_true")
parser.add_argument('-elliptic', help="use elliptic coefficients for the polynomial", action="store_true")

args = parser.parse_args()

n = args.n + 1 # number of points
use_clen = args.clen
use_idct = args.idct
use_dsc = args.dsc
use_cs = args.cs
use_idct2d = args.idct2d

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

if use_idct or use_idct2d:
    assert deg_x <= n or deg_y <= n, "Not enough points with respect to the degree of the polynomial"

if use_idct2d:
    assert args.x==[-1,1] and args.y==[-1,1], "The 2D IDCT works only on [-1,1]*[-1,1] for now..."

grid = Grid(n, args.x[0], args.x[1], args.y[0], args.y[1])
if use_idct:
    grid.computeXsYsForIDCT(deg_x, 'nodes', 'linear')
elif use_idct2d:
    grid.computeXsYsForIDCT(max(deg_x, deg_y), 'nodes', 'nodes')
else:
    grid.computeXsYs()

poly = np.loadtxt(args.poly, dtype=int)

# Core of the program

if args.elliptic:
    poly = np.multiply(poly, comb2D(deg_x + 1, deg_y + 1))

sub = Subdivision(grid, deg_x, deg_y, args.poly, args.der)

if not args.idct2d:
    intervals = sub.isolateIntervals(poly, n, use_clen, use_idct, use_dsc, use_cs)
else:
    intervals = sub.isolate2d(poly, n)
# sub.printComplexity()

# Computation time logging

Verbose.verboseprint("Logging...")
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

Verbose.verboseprint("Constructing the visualization...")
Visualization.show(intervals,args.poly, default_file, use_clen, use_idct, use_dsc, use_cs, args.hide, args.save, args.freq, sub.getSubdivisionTimeDistribution(), n, grid, args.idct2d)