import argparse

import numpy as np
import math
from sets import Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import flint as ft

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


# Parse the input

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
parser.add_argument('-poly', default='../polys/circle.poly')
parser.add_argument('-x', nargs=2, type=int, default=[-8,8])
parser.add_argument('-y', nargs=2, type=int, default=[-8,8])

args = parser.parse_args()

n = args.n + 1 # number of points
d = math.floor(np.log2(n))
xs = np.linspace(args.x[0], args.x[1], n)
ys = np.linspace(args.y[0], args.y[1], n)
max_dy = ys[1] - ys[0]

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

rad = (ys[-1] - ys[0]) / (2 * (n - 1))
rad = 0

partial_poly = np.empty((n, deg_y + 1), dtype=object)
for i in range(n):
    x = xs[i]
    for j in range(deg_y + 1):
        partial_poly[i,j] = ft.arb_poly(list(poly[j])[::-1])(ft.arb(x, rad))

intervals = np.empty(n, dtype="object")
for i in range(n):
    intervals[i] = subdivide(ys, 0, n - 1, list(partial_poly[i])[::-1], d)

# squares = np.empty(n - 1, dtype="object")
# for i in range(n - 1):
#     s = Set()
#     for j in {i, i + 1}:
#         for e in intervals[j]:
#             s.add(e[0])
#     squares[i] = s

# Show selected boxes

fig1 = plt.figure()

ax1 = fig1.add_subplot(111, aspect='equal')
# c = (xs[-1] - xs[0]) / (n - 1)
# for i in range(n - 1):
#     x = xs[i]
#     for e in squares[i]:
#         y = ys[e]
#         ax1.add_patch(patches.Rectangle((x,y),c,c))

for i in range(n):
    for e in intervals[i]:
        plt.plot([xs[i], xs[i]], [ys[e[0]], ys[e[1]]], '-k')

plt.xlim(xs[0],xs[-1])
plt.ylim(ys[0],ys[-1])
circle = plt.Circle((0, 0), 2, color='r', fill=False)
ax1.add_artist(circle)
plt.show()

print("Exited without trouble")