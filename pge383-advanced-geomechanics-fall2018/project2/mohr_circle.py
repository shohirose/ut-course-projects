import numpy as np
import math

class MohrCircle():

    def __init__(self, s1, s2, s3, p):
        self.set(s1, s2, s3, p)

    def set(self, s1, s2, s3, p):
        """
        Set principal stress and pore pressure

        Parameters
        ----------
        s1 : double
            Maximum principal stress
        s2 : double
            Intermediate principal stress
        s3 : double
            Minimum principal stress
        p : double
            Pore pressure
        """

        assert s1 > s2 and s2 > s3

        self.s1 = s1 - p
        self.s2 = s2 - p
        self.s3 = s3 - p

    def calc_each_circle(self, smax, smin, size=100, flip=False):
        """
        Calculate a part of Mohr circle

        Parameters
        ----------
        smax : double
            Larger normal effective stress
        smin : double
            Smaller normal effective stress
        size : int
            The number of points calculated along the circle
        flip : bool
            The direction of the circle. Default is counter-clockwise direction

        Returns
        -------
        x : numpy.array
            x coordinates along the circle
        y : numpy.array
            y coordinates along the circle
        """

        assert smax > smin

        radius = 0.5*(smax - smin)
        center = 0.5*(smax + smin)
        theta = np.linspace(0, math.radians(180), size)

        if flip:
            theta = np.flipud(theta)

        x = radius*np.cos(theta) + center
        y = radius*np.sin(theta)

        return x, y

    def calc_circle(self):
        x1, y1 = self.calc_each_circle(self.s1, self.s3)
        x2, y2 = self.calc_each_circle(self.s2, self.s3, flip=True)
        x3, y3 = self.calc_each_circle(self.s1, self.s2, flip=True)

        x = np.append(x1, np.append(x2, x3))
        y = np.append(y1, np.append(y2, y3))
        return x, y


