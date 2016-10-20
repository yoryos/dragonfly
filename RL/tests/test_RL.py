from RL.RL4 import RL
import numpy as np
import unittest
import os

TEST_WEIGHTS = os.path.join(os.path.dirname(__file__), "test_weights.dat")


class TestRL(unittest.TestCase):
    def setUp(self):
        self.rl = RL("test", [4,4])

    def test_false_step(self):
        self.assertEqual(self.rl.step([False, False, False, False], 0), [0,0,0,0])

    def test_true_frames(self):
        self.assertEqual(np.any(self.rl.step([False, True, False, False], 0)), True)

    def test_positive_learning(self):
        reward = 0
        for i in xrange(200):
            action = self.rl.step([False, False, True, False], reward)
            reward = -1
            if action[1] == 1:
                reward = 1
        self.assertEqual(self.rl.step([False,False,True,False], reward), [0,1,0,0])

    def test_negative_learning(self):
        reward = 0
        for i in xrange(200):
            action = self.rl.step([True, False, False, False], reward)
            reward = 0
            if action[3] == 1:
                reward = -1
        self.assertNotEqual(self.rl.step([True,False,False,False], reward), [0,0,0,1])


    def test_load_weights(self):
        test_rl = RL('test2', [4,4], load_weights=TEST_WEIGHTS)




if __name__ == '__main__':
    unittest.main()
