"""
Unit tests for module Dragonfly
"""

import unittest
from Environment.Dragonfly import Dragonfly
from Helper.Vectors import Vector_2D


class TestDragonfly(unittest.TestCase):
    """
    This class represents sequence of tests for class Dragonfly.
    """

    def setUp(self):
        """
        Method that runs at start of each test.
        """

        self.d1 = Dragonfly()
        self.d2 = Dragonfly(position=Vector_2D(5.0, 5.0), velocity=Vector_2D(1.0, 2.5))

    def test_update_no_change_to_velocity(self):
        self.d2.update()
        self.assertEqual(self.d2.perspective_position, Vector_2D(5.0, 5.0), "Perspective position shouldn't change")
        self.assertEqual(self.d2.absolute_position, Vector_2D(6.0, 7.5), "Wrong absolute position after update")

    def test_update_with_velocity(self):
        self.d1.update(Vector_2D(1.0, 1.0))
        self.assertEqual(self.d1.perspective_position, Vector_2D(0.0, 0.0), "Perspective position shouldn't change")
        self.assertEqual(self.d1.absolute_position, Vector_2D(1.0, 1.0), "Wrong absolute position after update")

        self.d2.update(Vector_2D(1.0, 1.0))
        self.assertEqual(self.d2.perspective_position, Vector_2D(5.0, 5.0), "Perspective position shouldn't change")
        self.assertEqual(self.d2.absolute_position, Vector_2D(6.0, 6.0), "Wrong absolute position after update")
