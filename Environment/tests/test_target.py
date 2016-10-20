"""
Unit tests for module Target
"""

import unittest
from Environment.Target import Target
from Helper.Vectors import Vector_2D


class TestTarget(unittest.TestCase):
    """
    This class represents sequence of tests for class Target.
    """

    def setUp(self):
        """
        Method that runs at start of each test.
        """

        self.target_0 = Target(wobble=2.0)
        self.target_1 = Target(velocity=Vector_2D(1.0, 2.0), wobble=0.0)

    def test_update_wobble(self):
        self.target_0.update()

        target_0_pos = self.target_0.position
        self.assertTrue(-2.0 <= target_0_pos.x <= 2.0 and
                        -2.0 <= target_0_pos.y <= 2.0 and
                        target_0_pos.y != 0.0 and
                        target_0_pos.x != 0.0,
                        "Wobble is wrong")

        self.target_0.update()
        target_0_pos = self.target_0.position
        self.assertTrue(-4.0 <= target_0_pos.x <= 4.0 and -4.0 <= target_0_pos.y <= 4.0, "Wobble is wrong")

        self.assertEqual(len(self.target_0.position_history), 3, "Incorrect length of position history")

    def test_update_stationary_dragonfly(self):
        self.target_1.update()
        self.assertEqual(self.target_1.position, Vector_2D(1.0, 2.0), "Position of target after update is wrong")

    def test_update_moving_dragonfly(self):
        self.target_1.update(Vector_2D(-2.0, -1.0))
        self.assertEqual(self.target_1.position, Vector_2D(3.0, 3.0),
                         "Position of target after update with dragonfly movement is wrong")
