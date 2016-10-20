import os
import unittest

import numpy as np

from Visualiser.StdpVisualiser import StdpVisualiser

TEST_DIR = os.path.join(os.path.dirname(__file__))


class TestStdpVisualisers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestStdpVisualisers, self).__init__(*args, **kwargs)

        self.data1 = np.random.randint(0, 2, 700 * 4).reshape([700, 4])
        self.data2 = np.random.randint(0, 2, 700 * 4).reshape([700, 4])

    def setup_complete(self):
        return StdpVisualiser(self.data1, self.data2, default_history=50)

    def setup_afferent_only(self):
        self.stdp = StdpVisualiser(None, self.data2)
        self.assertEqual(len(self.stdp.widgets), 1)

    def setup_output_only(self):
        self.stdp = StdpVisualiser(None, self.data2)
        self.assertEqual(len(self.stdp.widgets), 1)

    def test_reset_zoom(self):
        stdp = self.setup_complete()
        stdp.widgets[0].history_depth = 10
        stdp.reset_zoom()
        for w in stdp.widgets:
            self.assertEqual(w.history_depth, 50)

    def test_steps(self):
        stdp = self.setup_complete()
        self.assertEqual(stdp.steps, 700)

    def test_different_steps(self):
        stdp = StdpVisualiser(self.data1, self.data2[:-1, :])
        self.assertEqual(stdp.steps, 699)

    def test_reset_to_start(self):
        stdp = self.setup_complete()
        stdp.update()
        for w in stdp.widgets:
            self.assertGreater(w.index, 0)

        stdp.reset_to_start()

        for w in stdp.widgets:
            self.assertEqual(w.index, 0)

    def test_update(self):

        stdp = self.setup_complete()
        stdp.update()
        stdp.update()
        self.assertTrue(stdp.widgets[0].index == stdp.widgets[1].index == 2)

    def test_save(self):
        stdp = self.setup_complete()

        w = os.path.join(TEST_DIR, "test.png")
        stdp.save()

        i = stdp.save(w)
        if i > 0:
            self.assertEqual(i, 1)
            self.assertTrue(os.path.isfile(w))
            os.remove(w)
