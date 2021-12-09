# Part of the code from ../src/main.py which performs the partial evaluation
# with the IDCT using error tracking

import numpy as np
import argparse
from time import process_time

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')
from utils import error_polys2cheb_dct, error_idct_eval

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of points or number of subdivision intervals - 1)")
parser.add_argument('poly', type=str, help="file of polynomial coefficients")

args = parser.parse_args()

start = process_time()

n = args.n
input = np.loadtxt(args.poly, dtype=float)

for direction in {'x', 'y'}:
  if direction == 'x':
      poly = input
  else:
      poly = input.T
  my_poly2cheb = error_polys2cheb_dct
  my_idct_eval = error_idct_eval
  cheb = my_poly2cheb(poly.T)
  poly_y = my_idct_eval(cheb,n)

end = process_time()

print(f"{end - start:.3f}")