import numpy as np
from math import cos, pi, acos, floor, ceil


class Grid:

    def __init__(self, n, x_min, x_max, y_min, y_max):
        self.n = n
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.minus = {}
        self.zero = {}
        self.plus = {}
        self.lower = None
        self.upper = None

    def __divideCoord(self, coord, d):
        """
        Divide the evaluation interval in three subintervals.

        For the interval I, compute the intersection between I and respectively ]-infty,-1], [-1,1] and [1,+infty[.
        Determine the number of points in each subintervals.

        Parameters
        ----------
        coord: 'x' or 'y'
        d: Maximum partial degree of the polynomial which will be evaluated on the grid
        """
        if coord == 'x':
            coord_min = self.x_min
            coord_max = self.x_max
        if coord == 'y':
            coord_min = self.y_min
            coord_max = self.y_max

        self.lower = max(coord_min, min(-1, coord_max))
        self.upper = min(max(coord_min, 1), coord_max)

        n_m = round((self.lower - coord_min) / (coord_max - coord_min) * self.n)
        n_p = round((coord_max - self.upper) / (coord_max - coord_min) * self.n)
        n_z = self.n - n_m - n_p

        b = True

        if (self.lower != self.upper):
            N_min = round((n_z - 1) * pi / (acos(self.lower) - acos(self.upper)))
            N_max = round((n_z - 1 + 2) * pi / (acos(self.lower) - acos(self.upper)))

            self.zero['N_' + coord] = int(N_min)
            while (self.zero['N_' + coord] <= N_max):
                if (floor(self.zero['N_' + coord] / pi * acos(self.lower) - 0.5) - ceil(self.zero['N_' + coord] / pi * acos(self.upper) - 0.5) == n_z - 1):
                    break
                self.zero['N_' + coord] += 1

            b = b and (self.zero['N_' + coord] >= d)

        if (coord_min != self.lower):
            N_min = round((n_m - 0.5) * pi / (pi - acos(1 / coord_min)))
            N_max = round((n_m + 0.5) * pi / (pi - acos(1 / coord_min)))

            self.minus['N_' + coord] = int(N_min)
            while (self.minus['N_' + coord] <= N_max):
                if (self.minus['N_' + coord] - ceil(self.minus['N_' + coord] / pi * acos(1 / self.x_min) - 0.5) == n_m):
                    break
                self.minus['N_' + coord] += 1

            b = b and (self.minus['N_' + coord] >= d)

        if (self.upper != coord_max):
            N_min = round((n_p - 0.5) * pi / acos(1 / coord_max))
            N_max = round((n_p + 0.5) * pi / acos(1 / coord_max))

            self.plus['N_' + coord] = int(N_min)
            while (self.plus['N_' + coord] <= N_max):
                if (floor(self.plus['N_' + coord] / pi * acos(1 / coord_max) - 0.5) + 1 == n_p):
                    break
                self.plus['N_' + coord] += 1

            b = b and (self.plus['N_' + coord] >= d)

        assert b, f"Not enough points to subdivide the interval along the {coord}-axis for the change of basis"

    def __chebNodes(self, coord, d):
        """
        Compute the Chebyshev nodes.
        
        Parameters
        ----------
        coord: 'x' or 'y'
        d: Maximum partial degree of the polynomial which will be evaluated on the grid
        """
        self.__divideCoord(coord, d)

        cos_z = []
        if (self.lower != self.upper):
            self.zero['i_min_' + coord] = ceil(self.zero['N_' + coord] / pi * acos(self.upper) - 0.5)
            self.zero['i_max_' + coord] = floor(self.zero['N_' + coord] / pi * acos(self.lower) - 0.5)
            cos_z = [cos((2 * i + 1) * pi / (2 * self.zero['N_' + coord])) for i in range(self.zero['i_min_' + coord], self.zero['i_max_' + coord] + 1)]

        inv_cos_m = []
        t_m = []
        if (self.x_min != self.lower):
            self.minus['i_min_' + coord] = ceil(self.minus['N_' + coord] / pi * acos(1 / self.x_min) - 0.5)
            self.minus['i_max_' + coord] = self.minus['N_' + coord] - 1
            cos_m = [cos((2 * i + 1) * pi / (2 * self.minus['N_' + coord])) for i in range(self.minus['i_min_' + coord], self.minus['i_max_' + coord] + 1)]
            inv_cos_m = list(map(lambda x: 1 / x, cos_m))
            inv_cos_m.reverse()
        
        inv_cos_p = []
        t_p = []
        if (self.upper != self.x_max):
            self.plus['i_min_' + coord] = 0
            self.plus['i_max_' + coord] = floor(self.plus['N_' + coord] / pi * acos(1 / self.x_max) - 0.5)
            cos_p = [cos((2 * i + 1) * pi / (2 * self.plus['N_' + coord])) for i in range(self.plus['i_min_' + coord], self.plus['i_max_' + coord] + 1)]
            inv_cos_p = list(map(lambda x: 1 / x, cos_p))
            inv_cos_p.reverse()

        return inv_cos_m + cos_z + inv_cos_p

    def computeXsYsForIDCT(self, d, x_scale, y_scale):
        """
        Compute the points on the x-axis and the y-axis, for the IDCT.

        Parameters
        ----------
        d: Maximum partial degree of the polynomial which will be evaluated on the grid
        x_scale: 'linear' or 'nodes' (Chebyshev nodes or modified nodes if the interval is not [-1,1])
        y_scale: 'linear' or 'nodes' (Chebyshev nodes or modified nodes if the interval is not [-1,1])
        """

        if x_scale == 'linear':
            self.xs = np.linspace(self.x_min, self.x_max, self.n)
        if x_scale == 'nodes':
            self.xs = self.__chebNodes('x', d)
        
        if y_scale == 'linear':
            self.ys = np.linspace(self.y_min, self.y_max, self.n)
        if y_scale == 'nodes':
            self.ys = self.__chebNodes('y', d)
    
    def computeXsYs(self):
        """Compute the points on the x-axis and the y-axis linearly."""

        self.xs = np.linspace(self.x_min, self.x_max, self.n)
    
        self.ys = np.linspace(self.y_min, self.y_max, self.n)
