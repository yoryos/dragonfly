from Integration.DragonflyBrain import DragonflyBrain
import numpy as np
import unittest
import os
import shutil

TEST_DIR = os.path.dirname(__file__)


class TestDragonflyBrain(unittest.TestCase):

    def setUp(self):
        self.dragon = DragonflyBrain(dt = 1.0,
                                cstmd_buffer_size=100,
                                environment_config="Integration/config/Environment_test_1.ini",
                                estmd_config="Integration/config/ESTMD_norm.ini",
                                dragonfly_config="Integration/config/Dragonfly_test_1.ini",
                                cstmd_config="Integration/config/CSTMD1_norm.ini",
                                stdp_config ="Integration/config/STDP_test_1.ini",
                                run_until=DragonflyBrain.stage_ALL,
                                pyplot=False,
                                webcam=False,
                                cuda_device=1)

    def test_100_steps(self):
        self.dragon.run(100, False)
        directory = os.path.join(TEST_DIR, 'TEST_DATA')
        self.dragon.data_dump(Environment=True, ESTMD=True, CSTMD=True, STDP=True, RL=True, directory=directory)
        self.assertTrue(os.path.isdir(directory))
        shutil.rmtree(directory)


if __name__ == '__main__':
    unittest.main()
