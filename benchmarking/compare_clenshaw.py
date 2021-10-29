import numpy as np
import clenshaw as cs

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('n', type=int, help="size of the grid (number of points or number of subdivision intervals - 1)")
# parser.add_argument('poly', type=str, help="file of polynomial coefficients")

# args = parser.parse_args()

p = np.loadtxt("../polys/diagonal.poly", dtype=int)
cs.implicitplot(p, method='dct', resolution=256)