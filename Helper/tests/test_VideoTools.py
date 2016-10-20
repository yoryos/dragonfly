from Helper.Video_Tools import VideoGenerator, VideoToFrameConverter
import unittest
import os
import numpy as np

TEST_VID = os.path.join(os.path.dirname(__file__), "test_vid.avi")
TEST_PREMADE = os.path.join(os.path.dirname(__file__), "test_video.avi")

class TestVG(unittest.TestCase):

    def setUp(self):

        self.vg = VideoGenerator()
        self.frame_black = np.zeros([480,640,3])
        self.frame_white = np.ones([480,640,3]) * 255
        self.bf = [self.frame_black for i in xrange(10)]
        self.wf = [self.frame_white for i in xrange(10)]
        self.frames = self.bf + self.wf

    def test_generate_video(self):

        self.vg.generate_video(self.frames, 10.0, TEST_VID)
        assert os.path.isfile(TEST_VID)

    def tearDown(self):

        try:
            os.remove(TEST_VID)
        except:
            pass

class TestVTFC(unittest.TestCase):

    def setUp(self):

        self.vtfc = VideoToFrameConverter()
        self.vtfc.open_video(TEST_PREMADE)

    def test_frame_count(self):

        self.assertEquals(self.vtfc.get_frame_count(),20)

    def test_get_frame(self):

        s,f1 = self.vtfc.get_frame(color = True)
        self.assertTrue(s)
        np.testing.assert_array_equal(f1, np.zeros([480,640,3]))

        for i in xrange(10):
            _,f1 = self.vtfc.get_frame(color = True)

        self.assertTrue(s)
        np.testing.assert_array_equal(f1, np.ones([480,640,3]) * 255)

    def test_get_fps(self):

        self.assertEquals(self.vtfc.get_fps(),10.0)

    def test_get_all_frames(self):

        frames = self.vtfc.get_all_frames()
        self.assertEquals(len(frames), 20)

    def test_get_frames(self):

        _,frames = self.vtfc.get_frames(10)
        self.assertEquals(len(frames), 10)






