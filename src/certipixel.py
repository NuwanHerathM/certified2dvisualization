# Certified pixel program
# ====================
# It uses Taylor approximation followed by subdivision
# It also colors pixel in red and black

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
from utils import UpwardRounding, DownwardRounding, loop_interval_idct_eval, interval_polys2cheb_dct, subdivide, error_polys2cheb_dct, error_idct_eval, Pixel
from scipy.special import comb
import flint as ft

# Parse the input
np.seterr(all='raise')
dirname = os.path.dirname(__file__)
default_file = os.path.join(dirname, '../polys/unit_circle.poly')

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of points or number of subdivision intervals - 1)")
parser.add_argument('-poly', type=str, default=default_file, help="file of polynomial coefficients")
parser.add_argument('-hide', help="hide the plot", action="store_true")
parser.add_argument('-save', help="save the plot in the output directory", action="store_true")
parser.add_argument('-noaxis', help="hide the axes", action="store_true")
parser.add_argument('-v', '--verbose', help="turn on the verbosity", action="store_true")
parser.add_argument('-m', type=int, default=3, help="precision of the approximation")

args = parser.parse_args()

n = args.n # number of points

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

grid = Grid(n, -1, 1, -1, 1)

input = np.loadtxt(args.poly, dtype=float)

# Core of the program

# # interval methods return an incorrect output
# my_poly2cheb = interval_polys2cheb_dct
# my_idct_eval = loop_interval_idct_eval
# error methods is better with some gaps in the curve
my_poly2cheb = error_polys2cheb_dct
my_idct_eval = error_idct_eval

with Timer("first step", logger=None):
    Verbose.verboseprint("\tLocal approximation...")
    grid.computeXsYsForIDCT(deg, 'nodes', 'nodes')
    m = args.m
    radii = np.empty(n)
    with UpwardRounding():
        for i in range(n):
            r_left = (grid.xs[i-1] - grid.xs[i]) / 2 if 0 < i else 0
            r_right = (grid.xs[i] - grid.xs[i+1]) / 2 if i < n - 1 else 0
            radii[i] = max(r_left, r_right)

        hockeystick = comb(deg+1, m+2)
    with DownwardRounding():
        ogf_inv = 1 - np.abs(grid.xs) - radii
    with UpwardRounding():
        ogf = 1 / ogf_inv
        ogf[0] = hockeystick + 1
        ogf[-1] = hockeystick + 1
        radii_power = radii**(m+1)

    # taylor approximation
    poly_der = np.empty((deg+1,m+1,deg+1), dtype=object) # d^j/dx^j sum a_{i,k} x^i y^k
    poly_der[:,0,:] = input # sum a_{i,k} x^i y^k
    for i in range(deg+1):
        for j in range(m):
            tmp = ft.arb_poly(poly_der[:poly_der.shape[0]-j,j,i].tolist()).derivative()
            poly_der[:len(tmp),j+1,i] = tmp
            poly_der[len(tmp):,j+1,i] = 0
    poly_approx = np.empty((n,deg+1,m+1), dtype=object) # d^k/dx^k sum a_{i,k} c_i^l y^j
    for i in range(m+1):
        tmp = my_poly2cheb(poly_der[:,i,:].T)
        poly_approx[:,:,i] = 1 / ft.arb(i).fac() * (my_idct_eval(tmp,n).T)

    # evaluation around Chebyshev nodes
    poly_y = np.empty((n, deg+1), dtype="object") # P(c_i + [-r,r)],y) = sum a_{i,j} y^j
    for j in range(deg+1):
        with UpwardRounding():
            factor = radii_power * abs(max(input[:,j], key=abs))
            bound = np.minimum(ogf, hockeystick) * factor
        for i in range(n):
            val = ft.arb_poly(poly_approx[i,j].tolist())(ft.arb(0,radii[i]))
            ball = ft.arb(0, bound[i])
            poly_y[i,j] = val + ball

with Timer("second step", logger=None):
    Verbose.verboseprint("\tSubdivision...")
    intervals = np.empty(n, dtype="object")
    for i in range(n):
        p = ft.arb_poly(poly_y[i,:].tolist())
        intervals[i] = subdivide(grid.ys, p)


tmp = my_poly2cheb(input.T)
poly_nodes = np.empty((n+2, deg+1), dtype=object)
poly_nodes[:n,:] = my_idct_eval(tmp,n).T
poly_nodes[n] = np.sum(input, axis=0)
minusones = np.array([(-1)^i for i in range(deg+1)])
poly_nodes[n+1] = np.dot(input.T,minusones).T

# Computation time logging

Verbose.verboseprint("Logging...")

time_logger = logging.getLogger("timing")
time_logger.setLevel(logging.INFO)
handler_format = logging.Formatter('%(message)s')
time_handler = logging.FileHandler('certi_time.log', 'w')
time_handler.setFormatter(handler_format)
time_logger.addHandler(time_handler)

# for key, value in Timer.timers.items(): #sorted_dict:
#     time_logger.info(f"{key}\t{value}")
time_logger.info(f"partial_eval\t{Timer.timers['first step']}")
time_logger.info(f"second_step\t{Timer.timers['second step']}")

# Show isolated intervals and pixels

Verbose.verboseprint("Constructing the visualization...")
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle

fig1 = plt.figure()
base = os.path.basename(args.poly)
filename = os.path.splitext(base)[0]
fig1.canvas.manager.set_window_title(f"{filename}: n={n}, certified pixels")

ax1 = fig1.add_subplot(111, aspect='equal')
ax1.tick_params(axis='both', which='minor', labelsize=10)
if args.noaxis:
    ax1.set_axis_off() # hide the axis

# with Timer("append", logger=None):
pixels = []
polys = np.empty(n, dtype=object)
for i in range(n):
    polys[i] = ft.arb_poly(poly_nodes[i,:].tolist())
    for e in intervals[i]:
        if i != 0:
            pixels.append(Pixel(i1=i, i2=i-1, j1=e[0], j2=e[1]))
        if i != n-1:
            pixels.append(Pixel(i1=i+1, i2=i, j1=e[0], j2=e[1]))

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

# print(f"append\t{Timer.timers['append']}")
# print(f"test\t{Timer.timers['test']}")

# print(f"Number of black pixels: {c_black}")
# print(f"Number of red pixels: {c_red}")
# print(f"Ratio of black pixels among lighted pixels: {c_black / (c_red + c_black):.1%}")
# print(f"Ratio of lighted pixels in the picture: {(c_red + c_black) / ((n+1) * (n+1)):.1%}")

# lc = mc.LineCollection(segments, colors=colors, linewidths=0.1)
pc = mc.PatchCollection(rects, alpha=1, match_original=True)
ax1.add_collection(pc)
plt.xlim(grid.x_min, grid.x_max)
plt.ylim(grid.y_min, grid.y_max)

if args.save:
    dir = "../output"
    if not os.path.exists(dir):
        os.makedirs(dir)
    outpath_png = f"{dir}/{filename}_{n}_certified.png"
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
