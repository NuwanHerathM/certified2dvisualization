import numpy as np
import matplotlib.pyplot as plt

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')

from skimage import measure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="size of the grid (number of points or number of subdivision intervals - 1)")
parser.add_argument('poly', type=str, help="file of polynomial coefficients")

args = parser.parse_args()

# Construct test data
n= args.n
# x = np.tile(np.linspace(-1,1,n),(n,1))
# y = np.tile(np.linspace(1,-1,n),(n,1)).T
c = np.loadtxt(args.poly, dtype=float)
# r = np.polynomial.polynomial.polyval2d(x,y,c)
x = np.linspace(-1,1,n)
d = c.shape[0]
t = np.ndarray((d,n))
r = np.ndarray((n,n))
for i in range(d):
    t[i,:] = np.polyval(c[i,-1::-1], x)
for i in range(n):
    r[:,i] = np.polyval(t[-1::-1,i],x)
r = np.rot90(r)

# Find contours
contours = measure.find_contours(r, 0)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.set_aspect('equal')

for contour in contours:
    ax.plot(contour[:, 1] * 2 / n - 1, -contour[:, 0] * 2/ n + 1, linewidth=2)

plt.xlim(-1,1)
plt.ylim(-1,1)
plt.savefig("image.png", dpi=1200)