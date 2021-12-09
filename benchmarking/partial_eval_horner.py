# Partial evaluation with interval arithmetic Matrix multiplication with numpy
# instead of a Python loop on a Horner evaluation 
# Inspired from interval_polys2cheb_dct from ../src/utils.py

import numpy as np
import argparse
from time import process_time
import flint as ft

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of points or number of subdivision intervals - 1)")
parser.add_argument('poly', type=str, help="file of polynomial coefficients")

args = parser.parse_args()

# Construct test data
start = process_time()
n= args.n
c = np.loadtxt(args.poly, dtype=float)
d = c.shape[0]
nodes_power = np.empty((d, n), dtype=object)
N = ft.acb(n)
nodes = np.array([ft.acb.cos_pi((2 * ft.acb(i) + 1) / (2 * N)).real for i in range(n)])
for i in range(n):
    for j in range(d):
        nodes_power[j,i] = nodes[i]**j
node_eval = c @ nodes_power

end = process_time()

print(f"{end - start:.3f}")