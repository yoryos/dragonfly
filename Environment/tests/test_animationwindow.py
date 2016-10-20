"""
Unit tests for module AnimationWindow
"""

import unittest
import os

from Environment.AnimationWindow import AnimationWindow
from Environment.Background import Background
from Environment.Target import Target
from Helper.Vectors import Vector_2D
import numpy as np
import os

TEST_BACKGROUND = os.path.join(os.path.dirname(__file__), "testBackground.jpg")

class TestAnimationWindow(unittest.TestCase):
    """
    This class represents sequence of tests for class AnimationWindow.
    """

    def setUp(self):
        """
        Method that runs at start of each test.
        """

        self.background = Background(TEST_BACKGROUND)
        self.window_nb = AnimationWindow("test", [Target(start= Vector_2D(1,1), size = 1),
                                                  Target(start= Vector_2D(3,3), size = 1)], 5, 5)

        self.window_bg = AnimationWindow("test", [Target(start= Vector_2D(1,1), size = 1)], 5, 5, self.background)


    def test_draw_no_background(self):

        result = self.window_nb.draw()

        expected = np.array([[255,   0, 255, 255, 255],
                             [  0,   0,   0, 255, 255],
                             [255,   0, 255,   0, 255],
                             [255, 255,   0,   0,   0],
                             [255, 255, 255,   0, 255]]).reshape([5,5,1]).repeat(3,2)

        np.testing.assert_array_equal(result, expected)

    def test_draw_with_background(self):

        result = self.window_bg.draw()

        expected = self.background.grab(5,5)
        expected[0,1,:] = 0
        expected[1,0,:] = 0
        expected[1,1,:] = 0
        expected[1,2,:] = 0
        expected[2,1,:] = 0

        np.testing.assert_array_equal(result, expected)

    def test_save(self):

        self.window_bg.draw(True)

        self.assertTrue(os.path.isfile("test/0.png"))

        if os.path.exists("test/0.png"):
            os.remove("test/0.png")
            os.rmdir("test")

