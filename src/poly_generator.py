import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help="directory where the polynomial will be stored")
parser.add_argument('filename', type=str, help="name of the file")
parser.add_argument('n', type=int, help="degree on x")
parser.add_argument('m', type=int, help="degree on y")

args = parser.parse_args()

# if the coeffient distribution is centered on zero there are some properties
matrix = np.random.random_integers(0, 100, (args.m, args.n))
print(matrix)

outfile = os.path.join(args.dir, args.filename)
print(outfile)

np.savetxt(outfile, matrix, fmt='%d')