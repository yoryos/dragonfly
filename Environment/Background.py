'''
Background

Class for background used in the environment

@author: Dragonfly Project 2016 - Imperial College London
        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)

CITE OLD AUTHORS
'''

import os
import cv2
import numpy as np
from Helper.Vectors import Vector_2D

class Background(object):
    """
    Stores background used in animations.

    Attributes:
        directory (str): path to image
        img (?):
        data (ndarray): background as numpy asarray
        absolute_position (List[float,float]): position of the background's top left corner?
    """

    def __init__(self, directory, position=(0.0,0.0)):
        """
        Constructor

        Args:
            directory (str): Path to background image
            speed (Optional[float]): Speed in pixels per frame

        """
        if not os.path.isfile(directory):
            raise NameError

        print "Creating background from " + directory
        self.directory = directory

        self.data = cv2.imread(directory)
        self.shape = self.data.shape

        self.absolute_position = Vector_2D(position[0], position[1])

    def update(self, velocity=None):
        """
        Update the backgrounds position using it's speed

        Args:
            velocity ():
        """
        if velocity is None:
            velocity = Vector_2D(0.0,0.0)

        self.absolute_position += velocity

    def grab(self, result_height, result_width):
        """
        Creates a canvas with backgrounds
        """

        result = np.empty((result_height, result_width, 3), np.uint8)
        result.fill(255)

        bw_start, bh_start = 0, 0
        rw_start, rh_start = 0, 0
        if self.absolute_position.x < 0:
            rw_start = int(np.abs(np.rint(self.absolute_position.x)))
        else:
            bw_start = int(np.abs(np.rint(self.absolute_position.x)))
        if self.absolute_position.y < 0:
            rh_start = int(np.abs(np.rint(self.absolute_position.y)))
        else:
            bh_start = int(np.abs(np.rint(self.absolute_position.y)))

        cw = min(result_width - rw_start, self.shape[1] - bw_start)
        ch = min(result_height - rh_start, self.shape[0] - bh_start)

        cw = max(cw, 0)
        ch = max(ch, 0)

        rw_end = rw_start + cw
        rh_end = rh_start + ch

        bw_end = bw_start + cw
        bh_end = bh_start + ch

        np.copyto(result[rh_start:rh_end, rw_start:rw_end],
                  self.data[bh_start:bh_end, bw_start:bw_end])

        return result
