"""
Environment

Class for environment controller

@author: Dragonfly Project 2016 - Imperial College London
        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)
"""

import numpy as np
from Helper.Vectors import Vector_2D
import copy

class Dragonfly(object):
    """
    Class for the dragonflies focal point

    Attributes:
        pos (List[float, float]): position of the dragonflies focal point
        position_history (List[List[float,float]]): a history of the focal point
            positions
        velocity (float): velocity of the dragonflies focal point
        dt (foat): time step for update

    """

    def __init__(self, position=None, velocity=None, visible=False):
        """
        Constructor

        Args:
            position (List[float, float]): A list in the form of [x, y]
            velocity (List[float, float]): A list in the form of [delta x, delta y] in pixel/frame
        """
        if position is None:
            position = Vector_2D(0.0,0.0)
        if velocity is None:
            velocity = Vector_2D(0.0,0.0)

        self.perspective_position = position
        self.absolute_position = copy.deepcopy(position)
        self.position_history = [copy.deepcopy(position)]
        self.velocity = velocity
        self.visible = visible

    def update(self, velocity = None):
        """
        Update the dragonflies focal position using the new velocity

        Args:
            new_velocity (List[int, int]): new velocity in x,y m/s
        """

        if velocity is not None:
            self.velocity = velocity
        self.absolute_position += self.velocity
        self.position_history.append(copy.copy(self.absolute_position))


    def __str__(self):

        return "Dragonfly " + "perspective: " + str(self.perspective_position) + "px absolute: " + str(
                self.absolute_position) + "px velocity: " + str(self.velocity) + "ppf"
