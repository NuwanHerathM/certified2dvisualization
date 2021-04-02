import flint as ft
from math import pi


from codetiming import Timer

def bound(a, c, m, r, hs):
    # bound with an ordinary generating function
    with Timer("ogf", logger=None):
        ogf = 1/(1 - ft.arb(c).abs_upper() -r)**(m+2)
    if (ogf.lower() < 0): # force hs if ogf has negative values
        ogf = hs + 1
    with Timer("res", logger=None):
        res = a * r**(m+1) * ft.arb.min(ogf, hs)
    return res

def vanishes(poly, a, c, m, hs, r):
    with Timer("bound", logger=None):
        b = bound(a, c, m, r, hs)
    with Timer("eval", logger=None):
        ball = poly(ft.arb(0,r.abs_upper())) + ft.arb(0, b.abs_upper())
    return 0 in ball
