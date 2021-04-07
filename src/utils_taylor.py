import flint as ft
from math import pi

from codetiming import Timer

# Obsolete file

def vanishes(poly, hs, ogf, factor, r):
    with Timer("bound", logger=None):
        b = min(ogf, hs) * factor
    with Timer("eval", logger=None):
        ball = poly(ft.arb(0,r)) #+ ft.arb(0, b)
    return 0 in ball
