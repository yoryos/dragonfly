"""
Unit tests for module Background
"""

import unittest
import numpy as np
from Environment.Background import Background
from Helper.Vectors import Vector_2D
import os

TEST_BACKGROUND = os.path.join(os.path.dirname(__file__), "testBackground.jpg")


class TestBackground(unittest.TestCase):
    """
    This class represents sequence of tests for class Background.
    """

    def setUp(self):
        self.background = Background(TEST_BACKGROUND)

    def test_wrong_path(self):

        with self.assertRaises(NameError):
            wong_path = Background("wong_path")

    def test_update(self):

        self.assertEqual(self.background.absolute_position, Vector_2D(0.0, 0.0))
        self.background.update()
        self.assertEqual(self.background.absolute_position, Vector_2D(0.0, 0.0))
        self.background.update()
        self.assertEqual(self.background.absolute_position, Vector_2D(0.0, 0.0))

    def test_grab(self):

        result = self.background.grab(640,480)

        self.assertEqual(result.shape, (640,480,3), "Shape of grabbed frame is wrong")
        self.assertTrue(np.any(result), "Why all zeros?")

        self.background.update(Vector_2D(-1.0,-1.0))
        result = self.background.grab(640,480)
        self.assertEqual(result.shape, (640,480,3), "Shape of grabbed frame is wrong")
        self.assertTrue(np.any(result), "Why all zeros?")
