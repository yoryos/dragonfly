import os
import unittest

import cv2

from Visualiser.VideoVisualiser import VideoVisualiser

TEST_COLOR = os.path.join(os.path.dirname(__file__), "testColorVid.avi")
TEST_BW = os.path.join(os.path.dirname(__file__), "testBWColor.avi")
TEST_DIR = os.path.join(os.path.dirname(__file__))


class TestVideoVisualiser(unittest.TestCase):
    def test_black_and_white(self):

        vv = VideoVisualiser("testVisualiser", TEST_BW)
        self.assertEqual(len(vv.frames), 9)

        for i in xrange(10):
            vv.update()

        self.assertEqual(len(vv.frames), 9)

    def test_color(self):

        vv = VideoVisualiser("testVisualiser", TEST_COLOR)
        self.assertEqual(len(vv.frames), 9)

        for i in xrange(10):
            vv.update()

        self.assertEqual(len(vv.frames), 9)

    def test_reset(self):

        vv = VideoVisualiser("testVisualiser", TEST_BW)
        self.assertEqual(10,
                         vv.vp.video.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        vv.reset_to_start()
        self.assertEqual(0,
                         vv.vp.video.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))

    def test_steps(self):

        vv = VideoVisualiser("testVisualiser", TEST_BW)
        self.assertEqual(vv.steps, 300)

    def test_save(self):

        vv = VideoVisualiser("testVisualiser", TEST_COLOR)
        i = vv.save(name="test", dir=TEST_DIR)
        e = os.path.join(TEST_DIR, "test.png")
        self.assertTrue(os.path.isfile(e))
        os.remove(e)
