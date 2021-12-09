import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help="directory where the polynomial will be stored")
parser.add_argument('filename', type=str, help="name of the file")
parser.add_argument('d', type=int, help="total degree")

args = parser.parse_args()

# if the coeffient distribution is centered on zero there are some properties
d = args.d
matrix = np.random.randint(-100, 101, (d+1, d+1))
for i in range(d+1):
    for j in range(d+1):
        if i+j > d:
            matrix[i,j] = 0
print(matrix)

outfile = os.path.join(args.dir, args.filename)
print(outfile)

np.savetxt(outfile, matrix, fmt='%d')