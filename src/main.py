import argparse
import os

import numpy as np
import statistics
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from subdivision import Subdivision

from codetiming import Timer

import logging

from visu_utils import Verbose

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
parser.add_argument('-save', help="save the plot in the output directory")
parser.add_argument('-der', help="use the derivative as a subdivision termination criterion", action="store_true")
group_sub = parser.add_mutually_exclusive_group()
group_sub.add_argument('-dsc', help="use Descartes's rule for the subdivision", action="store_true")
group_sub.add_argument('-cs', help="use GM's clenshaw for the isolation", action="store_true")
parser.add_argument('-v', '--verbose', help="turn on the verbosity", action="store_true")

args = parser.parse_args()

n = args.n + 1 # number of points
xs = np.linspace(args.x[0], args.x[1], n)
ys = np.linspace(args.y[0], args.y[1], n)
use_clen = args.clen
use_idct = args.idct
use_dsc = args.dsc
use_cs = args.cs

# Function for verbosity
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

if use_idct:
    assert deg_x <= n or deg_y <= n, "Not enough points with respect to the degree of the polynomial"

poly = np.loadtxt(args.poly, dtype=int)
# poly = np.random.randint(0,101, (deg_x + 1, deg_y + 1))

# Core of the program

sub = Subdivision(xs, ys, deg_x, deg_y, args.poly, args.der)

intervals = sub.isolateIntervals(poly, n, use_clen, use_idct, use_dsc, use_cs)

# sub.drawSubdivisions()
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
    # Drawing of the polynomial
    fig1 = plt.figure(dpi=600)
    base = os.path.basename(args.poly)
    eval_method = "clenshaw" * use_clen + "idct" * use_idct + "horner" * (1 - max(use_clen, use_idct))
    isol_method = "interval" * (1 - max(use_dsc, use_cs)) + "dsc" * use_dsc + "gm" * use_cs
    fig1.canvas.set_window_title(f"{os.path.splitext(base)[0]}: n={n - 1}, " + eval_method + ", " + isol_method)

    ax1 = fig1.add_subplot(111, aspect='equal')

    alpha = (xs[-1] - xs[0]) / 2
    shift = xs[-1] - alpha
    grid = sub.getGrid()
    segments = []
    colors = []
    for i in range(n):
        for e in merge(intervals[i]):
            if (use_idct):
                x = grid[i]
                segments.append([(x, ys[e[0]]), (x, ys[e[1]])])
            else:
                segments.append([(xs[i], ys[e[0]]), (xs[i], ys[e[1]])])
            colors.append((not e[2], 0, 0, 1))
    
    lc = mc.LineCollection(segments, colors=colors, linewidths=0.1)
    ax1.add_collection(lc)
    plt.xlim(xs[0],xs[-1])
    plt.ylim(ys[0],ys[-1])

    if (args.poly == default_file):
        # draw a circle if the default file is used
        circle = plt.Circle((0, 0), 2, color='r', fill=False)
        ax1.add_artist(circle)

    if args.save:
        filename = os.path.splitext(os.path.basename(args.poly))[0]
        plt.savefig(f"../output/{filename}_{n-1}_{eval_method}_{isol_method}.png", bbox_inches='tight')
        plt.savefig(f"../output/{filename}_{n-1}_{eval_method}_{isol_method}.pdf", bbox_inches='tight', dpi=1200)
    # Frequency analysis
    if args.freq:
        (distr, res) = sub.getSubdivisionTimeDistribution()
        x = np.linspace(distr.min(), distr.max(), res.frequency.size)
        fig2 = plt.figure(figsize=(5, 4))

        ax2 = fig2.add_subplot(1, 1, 1)

        ax2.bar(x, res.frequency, width=res.binsize)

        fig2.canvas.set_window_title('Relative frequency histogram')
        vert_mean = plt.axvline(x=distr.mean(), figure=fig2, color='k')
        vert_median = plt.axvline(x=statistics.median(distr), figure=fig2, color='r')
        plt.legend([vert_mean, vert_median], ['mean', 'median'])

        ax2.set_xlim([x.min(), x.max()])
        plt.xlabel('time (s)')
    
    if not args.hide:
        plt.show()