import argparse
import os

import numpy as np
from math import cos, pi, factorial

from numpy.lib import utils
from utils import polys2cheb_dct, idct_eval, comb2D, factorial2D
import flint as ft
from scipy.special import comb
from verbosity import Verbose

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
parser.add_argument('-hide', help="hide the plot", action="store_true")
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

if deg_x != deg_y:
    print("Partial degrees in x and y seem to be different. It may break the code or not...")

poly = np.loadtxt(args.poly, dtype=int)

# Core of the program

if args.elliptic:
    poly = np.multiply(poly, np.sqrt(comb2D(deg_x + 1, deg_y + 1)))
if args.flat:
    poly = np.multiply(poly, 1/np.sqrt(factorial2D(deg_x + 1, deg_y + 1)))

grid = np.array([cos((2 * i + 1) * pi / (2 *n)) for i in range(0, n)])

if deg_x < m:
    print(f"m={m} is greater than the degree.")
    m = deg_x -1
    print(f"Its value has been set to {m}.")
p_der = np.zeros((m+1,n))

Verbose.verboseprint("Evaluation...")

p = np.empty((n,deg_y+1))
with Timer("conversion", logger=None):
    _p = polys2cheb_dct(poly.T)
    p = idct_eval(_p, n).T

radii = np.empty(n)
with Timer("radii", logger=None):
    for i in range(n):
        r_left = -(grid[i] - grid[i-1]) / 2 if 0 < i else 0
        r_right = -(grid[i+1] - grid[i]) / 2 if i < n - 1 else 0
        radii[i] = max(r_left, r_right)
monomials = np.empty((n, m+1))
for i in range(n):
    monomials[i] = radii[i]**np.arange(m+1)

hockeystick = comb(deg_x+1, m+2)
with Timer("ogf", logger=None):
    ogf = 1 / (1 - np.abs(grid) - radii)
    ogf[0] = hockeystick + 1
    ogf[-1] = hockeystick + 1
radii_power = radii**(m+1)

intervals = np.empty(n, dtype="object")

with Timer("conv 2", logger=None):
    dct_eval = polys2cheb_dct(p)

intervals = []
for i in range(n):
    _p = dct_eval[i]
    factor = radii_power * max(p[i], key=abs)
    bound = np.minimum(ogf, hockeystick) * factor
    with Timer("approximation", logger=None):
        for k in range(m+1):
            tmp = np.polynomial.chebyshev.chebder(_p, k)
            p_der[k,:] = 1/factorial(k) * idct_eval(tmp, n).T
    with Timer("isolation", logger=None):
        a_0 = p_der[0,:]
        q = np.abs(p_der)
        q[0,:] = 0
        val = np.einsum('ij,ji->j', q, monomials)
        bools = np.logical_and(a_0 - val - bound < 0, 0 < a_0 + val + bound)
        for j in range(n):
            if bools[j]:
                intervals.append((i,j))

timers = Timer.timers
print(f"""radii: {timers['radii']}
ogf: {timers['ogf']}
conversion (x) + idct: {timers['conversion']}
conversion (y): {timers['conv 2']}
approximation: {timers['approximation']}
isolation: {timers['isolation']}""")

# Show isolated intervals

if not args.hide:
    Verbose.verboseprint("Constructing the visualization...")
    fig1 = plt.figure(dpi=600)

    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.tick_params(axis='both', which='minor', labelsize=10)

    interval_lut = [(grid[i+1] + grid[i]) / 2 for i in range(n-1)]
    interval_lut.insert(0, 1)
    interval_lut.append(-1)
    edges = []
    nodes = []
    for e in intervals:
        edges.append([(grid[e[0]], interval_lut[e[1]]), (grid[e[0]], interval_lut[e[1]+1])])
        nodes.append([grid[e[0]], interval_lut[e[1]]])
        nodes.append([grid[e[0]], interval_lut[e[1]+1]])

    lc = mc.LineCollection(edges, linewidths=0.1)
    ax1.add_collection(lc)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.draw()
    plt.show(block=True)

    import sys
    from vispy import scene, app
    import vispy.io

    canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True, bgcolor='white')
    grid = canvas.central_widget.add_grid(margin=10)
    grid.spacing = 0

    title = scene.Label("Plot Title", color='black')
    title.height_max = 40
    grid.add_widget(title, row=0, col=0, col_span=2)

    yaxis = scene.AxisWidget(orientation='left',
                            axis_label='Y Axis',
                            axis_font_size=12,
                            axis_label_margin=50,
                            tick_label_margin=5,
                            text_color='black')
    yaxis.width_max = 80
    grid.add_widget(yaxis, row=1, col=0)

    xaxis = scene.AxisWidget(orientation='bottom',
                            axis_label='X Axis',
                            axis_font_size=12,
                            axis_label_margin=50,
                            tick_label_margin=5,
                            text_color='black')

    xaxis.height_max = 80
    grid.add_widget(xaxis, row=2, col=1)

    right_padding = grid.add_widget(row=1, col=2, row_span=1)
    right_padding.width_max = 50

    view = grid.add_view(row=1, col=1, border_color='black')
    nodes = np.asarray(nodes)
    N = len(nodes)
    edges = np.empty((int(N / 2), 2))
    edges[:, 0] = np.arange(0, N, 2)
    edges[:, 1] = np.arange(1, N, 2)
    plot = scene.Line(pos=nodes, connect=edges, width=3, parent=view.scene)
    view.camera = scene.PanZoomCamera(rect=(-1, -1, 2, 2))

    xaxis.link_view(view)
    yaxis.link_view(view)

    # img = canvas.render()
    # vispy.io.write_png("image.png", img)

    if __name__ == '__main__' and sys.flags.interactive == 0:
        app.run()
