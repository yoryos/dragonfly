'''
2D/3D Coordinates inheriting from a standard Vector

__author__: Dragonfly Project 2016 - Imperial College London ({anc15, cps15, dk2015, gk513, lm1015,
zl4215}@imperial.ac.uk)

'''

import numpy as np
from copy import deepcopy

class Vector(object):
    """Class representing Vectors """
    def __init__(self, coords):
        """

        Args:
            coords (ndarray): array of coordinates of type [e1, e2, e3 .. eN]
        """
        self.coords = np.array(coords)
        self.dims = len(coords)

    def unit_vector(self):
        """ Returns the unit vector of the vector  """
        return self.coords / np.linalg.norm(self.coords)

    def copy(self):
        return deepcopy(self)

    def angle(self, other):
        """ Returns the angle in radians between """
        return np.arccos(np.clip(np.dot(self.unit_vector(), other.unit_vector()), -1.0, 1.0))

    def distance(self, other):
        difference = (self - other).to_list()
        return np.linalg.norm(difference)

    def to_list(self):
        return list(self.coords)

    def to_array(self):
        return self.coords

    def midpoint(self, other):
        return self.mid(other, 0.5)

    def mid(self, other, fraction):
        return self + (other - self) * fraction

    def __mul__(self, scalar):
        return self.__class__(coords = self.coords * float(scalar))

    def __sub__(self, other):
        return self.__class__(coords = self.coords - other.coords)

    def __add__(self, other):
        return self.__class__(coords = self.coords + other.coords)

    def __div__(self, scalar):
        return self.__class__(coords = self.coords / float(scalar))

    def __str__(self):
        return str(self.coords)

    def __eq__(self, other):
        return np.all(self.coords == other.coords)

    def __iadd__(self, other):
        self.coords += other.coords
        return self

    def __isub__(self, other):
        self.coords -= other.coords
        return self


class Vector_3D(Vector):
    """ A simple 3D cartesian coordinate

    Attributes:
           x(float): x-coordinate
           y(float): y-coordinate
           z(float): z-coordinate
    """
    x_index = 0
    y_index = 1
    z_index = 2

    def __init__(self, x = None, y = None, z = None, coords = None):
        """
        Constructor for 3d coordinate

        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            z (float): z-coordinate
        """
        if coords is None:
            Vector.__init__(self, np.array([float(x), float(y), float(z)]))
        else:
            Vector.__init__(self, coords)

    @property
    def x(self):
        return self.coords[self.x_index]

    @x.setter
    def x(self, val):
        self.coords[self.x_index] = val


    @property
    def y(self):
        return self.coords[self.y_index]

    @y.setter
    def y(self, val):
        self.coords[self.y_index] = val

    @property
    def z(self):
        return self.coords[self.z_index]

    @z.setter
    def z(self, val):
        self.coords[self.z_index] = val

    def project(self, out):
        """
        Project out a dimensions from the 3d coordinate into a 2d coordinate
        Args:
            out (char): either x,y or z

        Returns:
            Vector_2D: projected coordinate
        """
        if out == "x":
            return Vector_2D(self.coords[self.y_index], self.coords[self.z_index])
        elif out == "y":
            return Vector_2D(self.coords[self.x_index], self.coords[self.z_index])
        elif out == "z":
            return Vector_2D(self.coords[self.x_index], self.coords[self.y_index])

        return None

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"


class Vector_2D(Vector):
    """ A simple 2D cartesian coordinate

    Attributes:
           x(float): x-coordinate
           y(float): y-coordinate
    """

    x_index = 0
    y_index = 1

    def __init__(self, x = None, y = None, coords = None):
        """
        Constructor for 2d coordinate

        Args:
            x (float): x-coordinate
            y (float): y-coordinate
        """
        if coords is None:
            Vector.__init__(self, np.array([x, y]))
        else:
            Vector.__init__(self, coords)


    @property
    def x(self):
        return self.coords[self.x_index]

    @x.setter
    def x(self, val):
        self.coords[self.x_index] = val

    @property
    def y(self):
        return self.coords[self.y_index]

    @y.setter
    def y(self, val):
        self.coords[self.y_index] = val

    def to_angle(self):
        """ Converts vector into an angle """
        return np.arctan2(self.coords[1], self.coords[0])

    def compare_angles(self, other_angle):
        """ Compares difference between vector's angle and another angle """
        d = self.to_angle() - other_angle
        while d > np.pi:
            d -= 2*np.pi
        while d < -np.pi:
            d += 2*np.pi
        return d

    def __repr__(self):
        return self.__str__()

    def __str__(self):

        other = ""
        if hasattr(self,"idx"):
            other+= str(self.idx)
        return "(" + str(self.x) + "," + str(self.y) + other + ")"

