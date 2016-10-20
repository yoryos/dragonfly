from ESTMD.Estmd import Estmd
import numpy as np
import unittest
import os

TEST_DIR = os.path.dirname(__file__)
TEST_FILE = os.path.join(TEST_DIR, "test_frames.dat")


class TestESTMD(unittest.TestCase):

    def setUp(self):
        self.estmd = Estmd("test", resize_factor= 1.0)

    def test_run_no_frames(self):
        with self.assertRaises(ValueError):
            frame = np.array([[]])
            self.estmd.step(frame)

    def test_run_frames(self):
        frames = np.loadtxt(TEST_FILE)
        for f in frames:
            self.assertIsNotNone(self.estmd.step(f))

    def test_without_preprocess(self):

        self.estmd2 = Estmd("test", preprocess_resize=False, resize_factor=1.0)
        frames = np.loadtxt(TEST_FILE)
        for f in frames:
            self.assertIsNotNone(self.estmd2.step(f))

    def test_with_parameters(self):
        H_filter = np.array([[2, 2, 2, 2, 2],
                             [2, 1, 1, 1, 2],
                             [2, 1, -1, 1, 2],
                             [2, 1, 1, 1, 2],
                             [2, 2, 2, 2, 2]])
        b = [0.1, 0.0003, -0.005, 0.02, -0.06, 0.08,
             -0.09, 0.34, -0.18]
        a = [1.3, -5.6, 9.34, -12.3, 11.1, -4.87, 3.24, -0.678, 0.089]
        CSK = np.array([[8.0 / 9.0, 8.0 / 9.0, 8.0 / 9.0],
                        [8.0 / 9.0, 8.0 / 9.0, 8.0 / 9.0],
                        [8.0 / 9.0, 8.0 / 9.0, 8.0 / 9.0]])
        b1 = [1.4, 1.2]
        a1 = [32.4, -68, 7]

        self.opt_estmd = Estmd(run_id="ESTMD1",
                               H_filter=H_filter,
                               b=b,
                               a=a,
                               CSKernel=CSK,
                               b1=b1,
                               a1=a1,
                               preprocess_resize=False,
                               resize_factor=1.0,
                               time_step=0.01,
                               threshold=0.4,
                               LMC_rec_depth=4
                               )

    def test_get_video(self):

        frames = np.loadtxt(TEST_FILE)
        for f in frames:
            self.assertIsNotNone(self.estmd.step(f))

        self.estmd.get_video(10,TEST_DIR,"test_output.avi", run_id_prefix=False)
        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR,"test_output.avi")))
        os.remove(os.path.join(TEST_DIR,"test_output.avi"))

if __name__ == '__main__':
    unittest.main()
