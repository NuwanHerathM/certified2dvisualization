import flint as ft
from math import floor
from scipy.special import comb

def bound(interval, a, points, i, m, d):
    r_left = ft.arb(points[i] - points[i-1]) / 2 if 0 < i else ft.arb(0)
    r_right = ft.arb(points[i+1] - points[i]) / 2 if i < len(points) - 1 else ft.arb(0)
    r = r_left.max(r_right)
    return a * r**(m+1) * ft.arb.max(1/(1 - interval.abs_upper() -r)**(m+2), comb(d+1, m+2))

def vanishes(poly, a, interval, points, i, m, d):
    b = bound(interval, a, points, i, m, d)
    ball = poly(interval) + ft.arb(0, b.abs_upper())
    # print(ball)
    return 0 in ball

if __name__ == "__main__":
    from math import cos, pi
    import numpy as np
    n = 128
    i = floor(n/2)
    grid = [cos((2 * i + 1) * pi / (2 *n)) for i in range(0, n)]
    a = ft.arb((2*i+1)/(2*n)).cos_pi()
    b = ft.arb((2*(i-1)+1)/(2*n)).cos_pi()
    print((a - b).abs_upper())
    print((a - b).abs_lower())
    diff = np.abs(np.diff(grid))
    print(np.max(diff))
