'''
WebCamHandler

Class for webcam controller for use as environment

@author: Dragonfly Project 2016 - Imperial College London
        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)
'''

import numpy as np
import cv2
from GeneralEnvironment import GeneralEnvironment


class WebCamHandler(GeneralEnvironment):
    """
    Class for running DragonflyBrain using WebCam input
    """

    def __init__(self, run_id, dt=1.0, camera=0):

        GeneralEnvironment.__init__(self, dt, run_id)

        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise IOError

        self.frame_dimensions = (int(self.cap.get(3)), int(self.cap.get(4)))

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError
        return frame

    def step(self, velocity=(0,0,0,0)):

        frame = self.read()
        self.frames.append(frame)
        return (self.green_filter(frame), 0)
