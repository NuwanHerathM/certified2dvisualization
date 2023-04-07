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
from utils import UpwardRounding, DownwardRounding, comb2D, factorial2D, loop_interval_idct_eval, interval_polys2cheb_dct, interval_idct, error_polys2cheb_dct, error_idct_eval, subdivide, polys2cheb, Pixel
from scipy.special import comb

# Parse the input
np.seterr(all='raise')
dirname = os.path.dirname(__file__)
default_file = os.path.join(dirname, '../polys/unit_circle.poly')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of points or number of subdivision intervals - 1)")
parser.add_argument('-poly', type=str, default=default_file, help="file of polynomial coefficients")
parser.add_argument('-hide', help="hide the plot", action="store_true")
parser.add_argument('-freq', help="show the subdivision time distribution", action="store_true")
parser.add_argument('-save', help="save the plot in the output directory", action="store_true")
parser.add_argument('-noaxis', help="hide the axes", action="store_true")
parser.add_argument('-v', '--verbose', help="turn on the verbosity", action="store_true")
# parser.add_argument('-idct2d', help="use the 2D IDCT", action="store_true")
# group_method = parser.add_mutually_exclusive_group()
# group_method.add_argument('-kac', help="use kac coefficients for the polynomial", action="store_true")
# group_method.add_argument('-elliptic', help="use elliptic coefficients for the polynomial", action="store_true")
# parser.add_argument('-flat', help="use elliptic coefficients for the polynomial", action="store_true")
parser.add_argument('-error', help="computation carrying the error bounds", action="store_true")
parser.add_argument('-taylor', help="taylor approximation on fibers", action="store_true")
parser.add_argument('-m', type=int, default=3, help="precision of the approximation")

args = parser.parse_args()

n = args.n # number of points
# use_idct2d = args.idct2d

# Set function for verbosity
Verbose.classInit(args.verbose)

# Read the polynomial

Verbose.verboseprint("Reading the polynomial file...")
with open(args.poly) as inf:
    lines = inf.readlines()
    deg = len(lines) - 1
    if (deg < 0):
        print("Empty file.")
        exit()

# if use_idct or use_idct2d:
assert deg <= n, "Not enough points with respect to the degree of the polynomial"

# grid = Grid(n, args.x[0], args.x[1], args.y[0], args.y[1])
grid = Grid(n, -1, 1, -1, 1)
# if use_idct:
#     grid.computeXsYsForIDCT(deg_x, 'nodes', 'linear')
# elif use_idct2d:
#     grid.computeXsYsForIDCT(max(deg_x, deg_y), 'nodes', 'nodes')
# else:
#     grid.computeXsYs()
grid.computeXsYsForIDCT(deg, 'nodes', 'nodes')

input = np.loadtxt(args.poly, dtype=float)

# Core of the program

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
        poly_y = my_idct_eval(cheb,n).T
        import flint as ft
        intervals = np.empty(n, dtype="object")
    with Timer("second step", logger=None):
        if not args.taylor:
            Verbose.verboseprint("\tSubdivision...")
            for i in range(n):
                p = ft.arb_poly(poly_y[i,:].tolist())
                intervals[i] = subdivide(grid.ys, p)
        else:
            Verbose.verboseprint("\tLocal approximation...")
            m = args.m
            p_der = np.zeros((m+1,n))
            radii = np.empty(n)
            with UpwardRounding():
                for i in range(n):
                    r_left = (grid.ys[i-1] - grid.ys[i]) / 2 if 0 < i else 0
                    r_right = (grid.ys[i] - grid.ys[i+1]) / 2 if i < n - 1 else 0
                    radii[i] = max(r_left, r_right)

                hockeystick = comb(deg+1, m+2)
            with DownwardRounding():
                ogf_inv = 1 - np.abs(grid.ys) - radii
            with UpwardRounding():
                ogf = 1 / ogf_inv
                ogf[0] = hockeystick + 1
                ogf[-1] = hockeystick + 1
                radii_power = radii**(m+1)

            poly_der = np.empty((n,m+1,poly_y.shape[1]), dtype=object)
            poly_der[:,0,:] = poly_y
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
                        factor = radii_power * abs(max(poly_y[i,:], key=abs))
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

tmp = my_poly2cheb(input.T)
poly_nodes = np.empty((n+2, deg+1), dtype=object)
poly_nodes[:n,:] = my_idct_eval(tmp,n).T
poly_nodes[n] = np.sum(input, axis=0)
minusones = np.array([(-1)^i for i in range(deg+1)])
poly_nodes[n+1] = np.dot(input.T,minusones).T

# Computation time logging

Verbose.verboseprint("Logging...")
# SORT_ORDER = {"change": 0, "conversion": 1, "evaluation": 2, "subdivision": 3}
# sorted_dict = sorted(Timer.timers.items(), key=lambda x: SORT_ORDER[x[0]])

time_logger = logging.getLogger("timing")
time_logger.setLevel(logging.INFO)
handler_format = logging.Formatter('%(message)s')
time_handler = logging.FileHandler('main_time.log', 'w')
time_handler.setFormatter(handler_format)
time_logger.addHandler(time_handler)

# for key, value in Timer.timers.items(): #sorted_dict:
#     time_logger.info(f"{key}\t{value}")
time_logger.info(f"partial_eval\t{Timer.timers['partial evaluation']}")
time_logger.info(f"second_step\t{Timer.timers['second step']}")

# Show isolated intervals and pixels

Verbose.verboseprint("Constructing the visualization...")
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle

fig1 = plt.figure()
# fig1 = plt.figure(frameon=False)
base = os.path.basename(args.poly)
filename = os.path.splitext(base)[0]
method = "sub" * (1 - args.taylor) + "taylor" * args.taylor
error = "intvl" * (1 - args.error) + "error" * args.error
fig1.canvas.manager.set_window_title(f"{filename}: n={n}, " + ", " + method + ", " + error)

ax1 = fig1.add_subplot(111, aspect='equal')
ax1.tick_params(axis='both', which='minor', labelsize=10)
if args.noaxis:
    ax1.set_axis_off() # hide the axis

pixels = []
polys = np.empty(n, dtype=object)

if not args.taylor:
    for i in range(0,n):
        polys[i] = ft.arb_poly(poly_nodes[i,:].tolist())
        for e in vertical[i]:
            if i != 0:
                pixels.append(Pixel(i1=i, i2=i-1, j1=e[0], j2=e[1]))
            if i != n-1:
                pixels.append(Pixel(i1=i+1, i2=i, j1=e[0], j2=e[1]))
        for e in horizontal[i]:
            if i != 0:
                pixels.append(Pixel(i1=e[1], i2=e[0], j1=i-1, j2=i))
            if i != n-1:
                pixels.append(Pixel(i1=e[1], i2=e[0], j1=i, j2=i+1))
else:
    for i in range(n):
        polys[i] = ft.arb_poly(poly_nodes[i,:].tolist())
        for j in vertical[i]:
            if i!= 0:
                if j!= 0:
                    pixels.append(Pixel(i1=i, i2=i-1, j1=j-1, j2=j))
                if j!= n-1:
                    pixels.append(Pixel(i1=i, i2=i-1, j1=j, j2=j+1))
            if i != n-1:
                if j!= 0:
                    pixels.append(Pixel(i1=i+1, i2=i, j1=j-1, j2=j))
                if j!= n-1:
                    pixels.append(Pixel(i1=i+1, i2=i, j1=j, j2=j+1))
    for i in range(n):
        for j in horizontal[i]:
            if i != 0:
                if j != 0:
                    pixels.append(Pixel(i1=j, i2=j-1, j1=i-1, j2=i))
                if j != n-1:
                    pixels.append(Pixel(i1=j+1, i2=j, j1=i-1, j2=i))
            if i != n-1:
                if j != 0:
                    pixels.append(Pixel(i1=j, i2=j-1, j1=i, j2=i+1))
                if j!= n-1:
                    pixels.append(Pixel(i1=j+1, i2=j, j1=i, j2=i+1))

# remove duplicates from the list pixels
pixels = {(pixel.i1, pixel.j1):pixel for pixel in pixels}.values()
merged_pixels = []
rects = []
c_red = 0
c_black = 0

# merge vertically adjacent pixels, with respect to their color
i1_prev = -1
j1_prev = -1
j2_prev = -1
color_prev = ''
for pixel in pixels:
    i1 = pixel.i1
    i2 = pixel.i2
    j1 = pixel.j1
    j2 = pixel.j2
    if pixel.isblack(polys, grid.ys):
        # c_black += 1
        color = 'black'
    else:
        # c_red += 1
        color = 'red'
    if i1 == i1_prev and j1 == j2_prev and color == color_prev:
        del merged_pixels[-1]
        j1 = j1_prev
    else:
        j1_prev = j1
    merged_pixels.append((Pixel(i1=i1, i2=i2, j1=j1, j2=j2),color))
    i1_prev = i1
    j2_prev = j2
    color_prev = color
    
# create colored rectangles for the visualization
for (pixel, color) in  merged_pixels:
    x = grid.xs[pixel.i1]
    dx = grid.xs[pixel.i2] - x
    y = grid.ys[pixel.j1]
    dy = grid.ys[pixel.j2] - y
    rects.append(Rectangle((x, y), dx, dy, color=color))

# print(f"Number of black pixels: {c_black}")
# print(f"Number of red pixels: {c_red}")
# print(f"Ratio of black pixels among lighted pixels: {c_black / (c_red + c_black):.1%}")
# print(f"Ratio of lighted pixels in the picture: {(c_red + c_black) / ((n+1) * (n+1)):.1%}")
# lc = mc.LineCollection(segments, colors=colors, linewidths=0.1)
pc = mc.PatchCollection(rects, alpha=1, match_original=True)
# ax1.add_collection(lc)
ax1.add_collection(pc)
plt.xlim(grid.x_min, grid.x_max)
plt.ylim(grid.y_min, grid.y_max)

if args.save:
    dir = "../output"
    if not os.path.exists(dir):
        os.makedirs(dir)
    outpath_png = f"{dir}/{filename}_{n}_{method}_{error}.png"
    plt.savefig(outpath_png, bbox_inches='tight', dpi=1200)
    # outpath_pdf = f"../output/{filename}_{n}_{weight}_{method}_{error}.pdf"
    # plt.savefig(outpath_pdf, bbox_inches='tight', dpi=1200)
    Verbose.verboseprint("Figure saved at the following locations:\n\t" + outpath_png) # + "\n\t" + outpath_pdf)

if not args.hide:
    Verbose.verboseprint("Done.")
    plt.draw()
    plt.show(block=True)
    # plt.pause(0.0001)
    # plt.close()
