import argparse
import os
from matplotlib import patches

import numpy as np
from subdivision import Subdivision
from visualization import Visualization
from grid import Grid

from codetiming import Timer

import logging

from verbosity import Verbose
from utils import UpwardRounding, comb2D, factorial2D, loop_interval_idct_eval, interval_polys2cheb_dct, interval_idct, error_polys2cheb_dct, error_idct_eval, subdivide
from scipy.special import comb

import copy

# Parse the input
np.seterr(all='raise')
dirname = os.path.dirname(__file__)
default_file = os.path.join(dirname, '../polys/circle.poly')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of points or number of subdivision intervals - 1)")
parser.add_argument('-poly', type=str, default=default_file, help="file of polynomial coefficients")
# parser.add_argument('-x', nargs=2, type=int, default=[-8,8], help="bounds on the x-axis")
# parser.add_argument('-y', nargs=2, type=int, default=[-8,8], help="bounds on the y-axis")
group_eval = parser.add_mutually_exclusive_group()
# group_eval.add_argument('-clen', help="use the Chebyshev basis and Clenshaw's scheme", action="store_true")
# group_eval.add_argument('-idct', help="use the Chebyshev basis and the IDCT", action="store_true")
parser.add_argument('-hide', help="hide the plot", action="store_true")
# parser.add_argument('-freq', help="show the subdivision time distribution", action="store_true")
parser.add_argument('-save', help="save the plot in the output directory", action="store_true")
parser.add_argument('-der', help="use the derivative as a subdivision termination criterion", action="store_true")
group_sub = parser.add_mutually_exclusive_group()
group_sub.add_argument('-dsc', help="use Descartes's rule for the subdivision", action="store_true")
group_sub.add_argument('-cs', help="use GM's clenshaw for the isolation", action="store_true")
parser.add_argument('-v', '--verbose', help="turn on the verbosity", action="store_true")
# parser.add_argument('-idct2d', help="use the 2D IDCT", action="store_true")
group_method = parser.add_mutually_exclusive_group()
group_method.add_argument('-kac', help="use kac coefficients for the polynomial", action="store_true")
group_method.add_argument('-elliptic', help="use elliptic coefficients for the polynomial", action="store_true")
parser.add_argument('-flat', help="use elliptic coefficients for the polynomial", action="store_true")
parser.add_argument('-error', help="computation carrying the error bounds", action="store_true")
parser.add_argument('-taylor', help="taylor approximation on fibers", action="store_true")
parser.add_argument('-m', type=int, default=3, help="precision of the approximation")

args = parser.parse_args()

n = args.n # number of points
# use_clen = args.clen
# use_idct = args.idct
use_dsc = args.dsc
use_cs = args.cs
# use_idct2d = args.idct2d

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

# if use_idct or use_idct2d:
assert deg_x <= n or deg_y <= n, "Not enough points with respect to the degree of the polynomial"

# if use_idct2d:
#     assert args.x==[-1,1] and args.y==[-1,1], "The 2D IDCT works only on [-1,1]*[-1,1] for now..."

# grid = Grid(n, args.x[0], args.x[1], args.y[0], args.y[1])
grid = Grid(n, -1, 1, -1, 1)
# if use_idct:
#     grid.computeXsYsForIDCT(deg_x, 'nodes', 'linear')
# elif use_idct2d:
#     grid.computeXsYsForIDCT(max(deg_x, deg_y), 'nodes', 'nodes')
# else:
#     grid.computeXsYs()

input = np.loadtxt(args.poly, dtype=int)

# Core of the program

if args.elliptic:
    input = np.multiply(input, np.sqrt(comb2D(deg_x + 1, deg_y + 1)))
if args.flat:
    input = np.multiply(input, 1/np.sqrt(factorial2D(deg_x + 1, deg_y + 1)))

for direction in {'x', 'y'}:
    if direction == 'x':
        Verbose.verboseprint("Evaluation along vertical fibers...")
        poly = input
    else:
        Verbose.verboseprint("Evaluation along horizontal fibers...")
        poly = input.T
    sub = None
    Verbose.verboseprint("\tPartial evaluation...")
    with Timer("partial evaluation", logger=None):
        if args.error:
            # error tracking
            my_poly2cheb = error_polys2cheb_dct
            my_idct_eval = error_idct_eval
        else:
            # interval arithmetic
            my_poly2cheb = interval_polys2cheb_dct
            my_idct_eval = loop_interval_idct_eval
        cheb = my_poly2cheb(poly.T)
        poly_y = my_idct_eval(cheb,n)
        import flint as ft
        intervals = np.empty(n, dtype="object")
    with Timer("second step", logger=None):
        if not args.taylor:
            Verbose.verboseprint("\tSubdivision...")
            grid.computeXsYsForIDCT(deg_x, 'nodes', 'nodes')
            for i in range(n):
                p = ft.arb_poly(poly_y[:,i].tolist())
                intervals[i] = subdivide(grid.ys, p)
        else:
            Verbose.verboseprint("\tLocal approximation...")
            grid.computeXsYsForIDCT(max(deg_x, deg_y), 'nodes', 'nodes')
            m = args.m
            p_der = np.zeros((m+1,n))
            radii = np.empty(n)
            with UpwardRounding():
                for i in range(n):
                    r_left = -(grid.ys[i] - grid.ys[i-1]) / 2 if 0 < i else 0
                    r_right = -(grid.ys[i+1] - grid.ys[i]) / 2 if i < n - 1 else 0
                    radii[i] = max(r_left, r_right)

                hockeystick = comb(deg_x+1, m+2)
                ogf = 1 / (1 - np.abs(grid.ys) - radii)
                ogf[0] = hockeystick + 1
                ogf[-1] = hockeystick + 1
                radii_power = radii**(m+1)

            poly_der = np.empty((n,m+1,poly_y.shape[0]), dtype=object)
            poly_der[:,0,:] = poly_y.T
            with Timer("derivative", logger=None):
                for i in range(n):
                    for j in range(m):
                        tmp = ft.arb_poly(poly_der[i,j,:poly_der.shape[2]-j].tolist()).derivative()
                        poly_der[i,j+1,:len(tmp)] = tmp
                        poly_der[i,j+1,len(tmp):] = 0
            poly_approx = np.empty((n,n,m+1), dtype=object)
            with Timer("approximation computation", logger=None): # <--- slow
                for i in range(m+1):
                    with Timer("poly2cheb", logger=None):
                        tmp = my_poly2cheb(poly_der[:,i,:])
                    with Timer("idct_eval", logger=None): # <--- very slow with interval arithmetic
                        poly_approx[:,:,i] = 1 / ft.arb(i).fac() * my_idct_eval(tmp,n)
            # intervals = []
            with Timer("eval", logger=None):
                for i in range(n):
                    intervals[i] = []
                    with UpwardRounding():
                        factor = radii_power * max(poly_y[:,i], key=abs)
                        bound = np.minimum(ogf, hockeystick) * factor
                    for j in range(n):
                        val = ft.arb_poly(poly_approx[i,j].tolist())(ft.arb(0,radii[j]))
                        ball = ft.arb(0, bound[j])
                        if 0 in val + ball:
                            intervals[i].append(j)
    if direction == 'x':
        vertical = intervals
    else:
        horizontal = intervals

# print(Timer.timers["partial evaluation"] + Timer.timers["second step"])

# if not args.idct2d:
#     intervals = sub.isolateIntervals(poly, n, use_clen, use_idct, use_dsc, use_cs)
# else:
#     intervals = sub.isolate2d(poly, n)
# sub.printComplexity()

# Computation time logging

Verbose.verboseprint("Logging...")
# SORT_ORDER = {"change": 0, "conversion": 1, "evaluation": 2, "subdivision": 3}
# sorted_dict = sorted(Timer.timers.items(), key=lambda x: SORT_ORDER[x[0]])

time_logger = logging.getLogger("timing")
time_logger.setLevel(logging.INFO)
handler_format = logging.Formatter('%(message)s')
time_handler = logging.FileHandler('subdivision_time.log', 'w')
time_handler.setFormatter(handler_format)
time_logger.addHandler(time_handler)

# for key, value in Timer.timers.items(): #sorted_dict:
#     time_logger.info(f"{key}\t{value}")
time_logger.info(f"partial_eval\t{Timer.timers['partial evaluation']}")
time_logger.info(f"second_step\t{Timer.timers['second step']}")

# Show isolated intervals

Verbose.verboseprint("Constructing the visualization...")
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle

fig1 = plt.figure()
base = os.path.basename(args.poly)
filename = os.path.splitext(base)[0]
weight = "kac" * (1 - max(args.elliptic, args.flat)) + "elliptic" * args.elliptic + "flat" * args.flat
method = "sub" * (1 - args.taylor) + "taylor" * args.taylor
error = "intvl" * (1 - args.error) + "error" * args.error
fig1.canvas.manager.set_window_title(f"{filename}: n={n}, " + weight + ", " + method + ", " + error)

if not args.taylor:

    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.tick_params(axis='both', which='minor', labelsize=10)

    segments = []
    colors = []
    rects = []
    extended_xs = np.empty(n+2)
    extended_xs[:n] = grid.xs
    extended_xs[n] = -1
    extended_xs[-1] = 1
    for i in range(n):
        for e in Visualization.merge_intervals(vertical[i]):
            segments.append([(grid.xs[i], grid.ys[e[0]]), (grid.xs[i], grid.ys[e[1]])])
            colors.append((not e[2], 0, 0, 1))
            rects.append(Rectangle((extended_xs[i+1], grid.ys[e[1]]),extended_xs[i-1] - extended_xs[i+1], grid.ys[e[0]] - grid.ys[e[1]]))
        for e in Visualization.merge_intervals(horizontal[i]):
            segments.append([(grid.ys[e[0]], grid.xs[i]), (grid.ys[e[1]], grid.xs[i])])
            colors.append((not e[2], 0, 0, 1))
            rects.append(Rectangle((grid.ys[e[1]], extended_xs[i+1]), grid.ys[e[0]] - grid.ys[e[1]], extended_xs[i-1] - extended_xs[i+1]))

    lc = mc.LineCollection(segments, colors=colors, linewidths=0.1)
    pc = mc.PatchCollection(rects, alpha=1)
    ax1.add_collection(lc)
    ax1.add_collection(pc)
    plt.xlim(grid.x_min, grid.x_max)
    plt.ylim(grid.y_min, grid.y_max)
else:
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.tick_params(axis='both', which='minor', labelsize=10)

    interval_lut = [(grid.ys[i+1] + grid.ys[i]) / 2 for i in range(n-1)]
    interval_lut.insert(0, 1)
    interval_lut.append(-1)
    edges = []
    colors = []
    rects = []
    for i in range(n):
        for e in Visualization.merge_nodes(vertical[i]):
            edges.append([(grid.xs[i], interval_lut[e[0]]), (grid.xs[i], interval_lut[e[0]+e[1]+1])])
            colors.append((0,0,0,1))
            rects.append(Rectangle((interval_lut[i], interval_lut[e[0]]), interval_lut[i+1] - interval_lut[i], interval_lut[e[0]+e[1]+1] - interval_lut[e[0]]))
    
    interval_lut = [(grid.xs[i+1] + grid.xs[i]) / 2 for i in range(n-1)]
    interval_lut.insert(0, 1)
    interval_lut.append(-1)
    for i in range(n):
        for e in Visualization.merge_nodes(horizontal[i]):
            edges.append([(interval_lut[e[0]], grid.ys[i]), (interval_lut[e[0]+e[1]+1], grid.ys[i])])
            colors.append((0,0,0,1))
            rects.append(Rectangle((interval_lut[e[0]], interval_lut[i]), interval_lut[e[0]+e[1]+1] - interval_lut[e[0]], interval_lut[i+1] - interval_lut[i]))

    lc = mc.LineCollection(edges, colors=colors, linewidths=0.1)
    pc = mc.PatchCollection(rects)
    ax1.add_collection(lc)
    ax1.add_collection(pc)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

if args.save:
    outpath_png = f"../output/{filename}_{n}_{weight}_{method}_{error}.png"
    plt.savefig(outpath_png, bbox_inches='tight')
    # outpath_pdf = f"../output/{filename}_{n}_{weight}_{method}_{error}.pdf"
    # plt.savefig(outpath_pdf, bbox_inches='tight', dpi=1200)
    Verbose.verboseprint("Figure saved at the following locations:\n\t" + outpath_png) # + "\n\t" + outpath_pdf)

if not args.hide:
    Verbose.verboseprint("Done.")
    plt.draw()
    plt.show(block=True)
    # plt.pause(0.0001)
    # plt.close()
