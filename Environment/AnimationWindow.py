"""
AnimationWindow

Tool to animate dragonfly targets. User only interacts with class Animation.
To create new animation create new instance of class Animation.
For any information on how to use this module refer to class Animation.

@author: Dragonfly Project 2016 - Imperial College London
        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)

CITE OLD AUTHORS
"""

from Target import Target
from Background import Background

import cv2
import numpy as np
import os


class AnimationWindow(object):
    """
    This class keeps track of current animation frame

    Attributes:
        background (Optinal[Background]): background of canvas
        target_list (List[Target]): targets that will be drawing
        width (int): width of canvas
        height (int): height of canvas
        index (int): index of saved canvas starting from 0

    """

    def __init__(self, run_id, target_list, width, height, background=False, dragonfly=None):
        """
        Constructor

        Args:
            target_list (List[Target]): list of targets to draw
            width (int): width of the drawing canvas
            height (int): height of the drawing canvas
            background (Optional[Background]): add a background to the canvas
        """
        self.background = background
        self.dragonfly = dragonfly
        self.target_list = target_list
        self.width = width
        self.height = height
        self.index = 0
        self.run_id = run_id

    def draw(self, savePNG=False):
        """
        Draw the current states of the targets and background

        Args:
            savePNG (bool): save drawing to png

        Returns:
            ndarray: frame
        """

        if self.background:
            canvas = self.background.grab(self.height, self.width)

        else:
            canvas = np.empty((self.height, self.width, 3), np.uint8)
            canvas.fill(255)

        for target in self.target_list:
            cv2.circle(canvas, tuple(np.rint([target.position.x, target.position.y]).astype(int)), int(target.size),
                       target.color, -1)

        if self.dragonfly and self.dragonfly.visible:
            cv2.line(canvas, tuple(np.rint([self.dragonfly.perspective_position.x - 7, self.dragonfly.perspective_position.y]).astype(int)), tuple(np.rint([self.dragonfly.perspective_position.x + 7, self.dragonfly.perspective_position.y]).astype(int)), (255, 255, 255), 2)
            cv2.line(canvas, tuple(np.rint([self.dragonfly.perspective_position.x, self.dragonfly.perspective_position.y - 7]).astype(int)), tuple(np.rint([self.dragonfly.perspective_position.x, self.dragonfly.perspective_position.y + 7]).astype(int)), (255, 255, 255), 2)

        if savePNG:
            if not os.path.isdir(self.run_id):
                os.mkdir(self.run_id)
            img_name = os.path.join(self.run_id,str(self.index) + ".png")
            self.index += 1
            cv2.imwrite(img_name, canvas)

        return canvas
