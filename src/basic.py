"""
This code tests the two basic approaches in order to evaluate a polynomial P(x,y) on a grid.
It works only for Python 2.7.
"""

import argparse

import timeit

import numpy as np

from subdivision import *

import os, sys

if sys.version_info[:2] > (2, 7):
    print("This code does not handle versions higher than Python 2.7")
    print("So were are exiting now.")
    exit()

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
parser.add_argument('-poly', default='../polys/circle.poly')
parser.add_argument('-x', nargs=2, type=int, default=[0,5])
parser.add_argument('-y', nargs=2, type=int, default=[0,5])

args = parser.parse_args()

n = args.n
xs = np.linspace(args.x[0], args.x[1], n)
ys = np.linspace(args.y[0], args.y[1], n)

p = list()

with open(args.poly) as inf:
    for line in inf:
        p.append(map(int, line.split()))

# naive algorithm
naive  = """
for x in xs:
    for y in ys:
        res = np.polyval(map(lambda c : np.polyval(c,x),p), y)
        # print(str(x) + " " + str(y) + " " + str(res))
"""

# partial evaluation
partial = """
p_x = map(lambda x : map(lambda c : np.polyval(c,x),p),xs)
for y in ys:
    for i in range(n):
        res = np.polyval(p_x[i], y)
        # print(str(xs[i]) + " " + str(y) + " " + str(res))
"""

""" #subdivision
p_x = map(lambda x : (map(lambda c : ft.arb(np.polyval(c,x),0),p)),xs)
print(list(p_x))
for i in range(n):
    l = subdivide(0, n, p_x[i], math.floor(math.log2(n)))
    print(l) """

naive_time = timeit.timeit(naive, setup="from __main__ import n, xs, ys, p, np", number=100)/100
partial_time = timeit.timeit(partial, setup="from __main__ import n, xs, ys, p, np", number=100)/100
print("Computation time for the naive algorithm: {}".format(naive_time))
print("Computation time with partial evaluation: {}".format(partial_time))