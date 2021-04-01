import flint as ft
from math import floor, pi, cos, factorial
from numpy.core.defchararray import rpartition
from scipy.special import comb

def bound(a, points, i, m, d, r):
    # bound with the hockey-stick identity
    hs = ft.arb(comb(d+1, m+2))
    # bound with an ordinary generating function
    ogf = 1/(1 - ft.arb(points[i]).abs_upper() -r)**(m+2)
    if (ogf.lower() < 0): # force hs if ogf has negative values
        ogf = hs + 1
    return a * r**(m+1) * ft.arb.min(ogf, hs)

def vanishes(poly, a, points, i, m, d):
    r_left = -ft.arb(points[i] - points[i-1]) / 2 if 0 < i else ft.arb(0)
    r_right = -ft.arb(points[i+1] - points[i]) / 2 if i < len(points) - 1 else ft.arb(0)
    r = r_left.max(r_right)
    b = bound(a, points, i, m, d, r)
    ball = poly(ft.arb(0,r)) + ft.arb(0, b.abs_upper())
    return 0 in ball
