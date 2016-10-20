'''
Environment

Class for environment controller

@author: Dragonfly Project 2016 - Imperial College London
        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)
'''
from random import uniform, seed

import numpy as np
from Helper.Vectors import Vector_2D
import copy


class Target(object):
    """
    This class represents different targets that will move on the screen.
    """

    def __init__(self, start=None, velocity=None,
                 wobble=5.0, size=10, color=(0, 0,0), idx=0):
        """

        Args:
            start ():
            velocity ():
            wobble ():
            size ():
            color ():
            idx ():

        Returns:

        """
        if start is None:
            start = Vector_2D(0.0,0.0) # cannot put it in the def as mutable
        if velocity is None:
            velocity = Vector_2D(0.0,0.0)

        np.random.seed(1)
        self.wobble = wobble
        self.internal_velocity = Vector_2D(np.random.uniform(-wobble,wobble),
                                            np.random.uniform(-wobble,wobble)) * 0.2

        self.start = start
        self.position = start

        self.bounds_position = start.copy()
        self.initial_velocity = velocity.copy()


        self.size = size
        self.color = color
        self.base_velocity = velocity
        self.position_history = [copy.copy(start)]
        self.idx = idx

        print "Created target " + str(self)

    def update(self, d_velocity = None):
        """
        Moves Target's position

        Args:
            d_velocity (): Dragonly's velocity, for relative movement
        """
        if d_velocity is None:
            d_velocity = Vector_2D(0.0,0.0)

        self.bounds_position += self.base_velocity - d_velocity
        self.position += self.base_velocity + self.internal_velocity - d_velocity

        difference = self.position - self.bounds_position

        if abs(difference.coords[0]) > self.wobble:
            self.internal_velocity.coords[0] *= -1
            self.internal_velocity.coords[1] = np.random.uniform(-self.wobble,self.wobble) * 0.2
        elif abs(difference.coords[1]) > self.wobble:
            self.internal_velocity.coords[1] *= -1
            self.internal_velocity.coords[0] = np.random.uniform(-self.wobble,self.wobble) * 0.2

        """
        dx = difference.coords[0]**2
        dy = difference.coords[1]**2
        x,y = self.internal_velocity.coords
        self.internal_velocity.coords[0] = x - x*dx
        self.internal_velocity.coords[1] = y - y*dy
        """

        self.position_history.append(copy.copy(self.position))

    def __str__(self):
        return "[" + str(self.idx) + " position: " + str(self.position) + "px velocity:" + str(
            self.base_velocity) + "ppf wobble: " + str(self.wobble) + "ppf size:" + str(self.size) + "px color:" + str(
                self.color)
