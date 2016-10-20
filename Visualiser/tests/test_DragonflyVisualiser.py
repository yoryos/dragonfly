import os
import unittest

from Visualiser.DragonflyVisualiser import DragonflyVisualiser
from Visualiser.VideoVisualiser import VideoVisualiser

TEST_COLOR = os.path.join(os.path.dirname(__file__), "testColorVid.avi")
TEST_BW = os.path.join(os.path.dirname(__file__), "testBWColor.avi")
TEST_DIR = os.path.join(os.path.dirname(__file__))


class TestDragonflyVisualiser(unittest.TestCase):
    def setUp(self):
        self.vv1 = VideoVisualiser("v1", TEST_COLOR, True)
        self.vv2 = VideoVisualiser("v2", TEST_COLOR, True)
        self.dv = DragonflyVisualiser(1.0, [self.vv1, self.vv2])
        self.dv.run(5)

    def test_layout(self):

        self.assertEqual(self.dv.num_cols, 2)
        self.assertEqual(self.dv.num_rows, 1)

    def test_double_start(self):

        self.dv.start()
        self.assertFalse(self.dv.start())

    def test_pause_not_running(self):

        self.assertFalse(self.dv.pause())

    def test_update(self):

        i = self.dv.index
        self.assertGreater(self.dv.update(), 0)
        self.assertEqual(self.dv.index, i + 1)

    def test_full_screen(self):

        fs = self.dv.isFullScreen()
        self.dv.toggle_screen()
        self.assertNotEqual(fs, self.dv.isFullScreen())
        self.dv.toggle_screen()
        self.assertEqual(fs, self.dv.isFullScreen())

    def test_save(self):
        dir = os.path.join(TEST_DIR, "test_dump")

        i = self.dv.save(dir)
        self.assertTrue(os.path.isdir(dir))
        self.assertEqual(len(os.listdir(dir)), i)
        for the_file in os.listdir(dir):
            file_path = os.path.join(dir, the_file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(dir)
