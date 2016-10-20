"""
Unit tests for module Environment
"""

import unittest
from Environment.Environment import Environment
from Helper.Vectors import Vector_2D
from Environment.Background import Background
import numpy as np
import os

TEST_BACKGROUND = os.path.join(os.path.dirname(__file__), "testBackground.jpg")


class TestEnvironment(unittest.TestCase):
    """
    This class represents sequence of tests for class Environment.
    """

    def setUp(self):
        """
        Method that runs at start of each test.
        """
        self.targets_config = [{"velocity": np.array([0.0,1.0]), "position": np.array([1.0,1.0]), "size":1,
                                "wobble":0},
                               {"velocity": np.array([0.0,0.0]), "position": np.array([4.0,4.0]), "size":1,
                                "wobble":0, "color":[0,0,0]}]

    def without_bg(self):

        return Environment("test", dt = 1000.0, ppm = 1.0, width=5, height=6, target_config=self.targets_config,
                           reward_version=1)

    def with_bg(self):

        return Environment("test", dt = 1000.0, ppm = 1.0, width=5, height=6, background_path=TEST_BACKGROUND,
                                 target_config=self.targets_config)


    def test_no_target(self):

        env = Environment("test", dt = 1000.0, ppm = 1.0, width=5, height=6)
        self.assertIsNone(env.get_closest_target_to_dragonfly())
        self.assertEqual(env.get_distance_to_closest_target(),0.0)

    def test_distance_to_closest_target(self):

        env = self.without_bg()
        self.assertEqual(env.distance_to_closest_target, np.sqrt(np.power(1.5,2) + 1))

    def test_get_closest_target(self):

        env = self.without_bg()
        self.assertEqual(env.get_closest_target_to_dragonfly(), env.target_list[1])


    def test_get_distance_reward(self):

        env = self.without_bg()
        del env.target_list[0] #only want one target
        _,r = env.step((-0.1,0,0,0))
        self.assertEqual(r,-1.0/3.0)
        _,r = env.step((0.1,0,0,0))
        self.assertEqual(r,1.0)

    def test_get_angle_reward_step(self):

        env = self.without_bg()
        env.reward_version = 2
        del env.target_list[0]
        _,r  = env.step((-0.1,0,0,0))
        self.assertEqual(r,-1.0/3.0)
        _,r  = env.step((0.1,0,0.1,0))
        self.assertEqual(r,1.0)



    def test_get_angle_reward(self):

        env = self.without_bg()

        angles = env.get_target_angles(Vector_2D(0.0,1.0))

        self.assertAlmostEqual(angles[0], np.arctan(-2.0/-1.5) + np.pi / 2, 5)
        self.assertAlmostEqual(angles[1], np.arctan(1.5), 5)

    def test_get_smallest_angle(self):

        env = self.without_bg()

        smallest_angle = env.get_smallest_angle(Vector_2D(0.0,1.0))
        self.assertAlmostEqual(smallest_angle, np.arctan(1.5), 5)


    def test_get_gausiian_reward(self):

        env = self.without_bg()
        env.reward_version = 3
        del env.target_list[0]

        _, r = env.step((0.1,0,0,0))
        self.assertGreater(r,0)
        _, r = env.step((-0.1,0,0,0))
        self.assertLess(r,0)


    def test_step_with_background(self):

        env = self.with_bg()

        frame, _ = env.step((0.1,0,0,0))
        frame *= 256 #in order to easily compare
        expected = Background(TEST_BACKGROUND).grab(6,6)[:,1:,1]
        expected[1,0] = 0
        expected[2,0:2] = 0
        expected[3,0] = 0
        expected[3,3] = 0
        expected[4,2:5] = 0
        expected[5,3] = 0

        np.testing.assert_array_equal(expected, frame)

    def test_step_without_background(self):

        env = self.without_bg()
        frame, _ = env.step((-0.1,0,0,0))
        frame *= 256
        expected = np.ones((6,5)) * 255
        expected[1,2] = 0
        expected[2,1:4] = 0
        expected[3,2] = 0
        expected[4,4] = 0
        np.testing.assert_array_equal(expected, frame)

