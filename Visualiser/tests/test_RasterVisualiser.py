import unittest

import numpy as np

from Visualiser.RasterVisualiser import RasterVisualiser


class TestRasterVisualiser(unittest.TestCase):

    def setUp(self):

        self.data = np.array([[True,True,False],
                              [False,True,True],
                              [False,False,False]])

        self.rv = RasterVisualiser(self.data)

    def test_convert_to_raster(self):

        t,d,s,n = RasterVisualiser.convert_to_raster(self.data)
        print t,d,s,n
        self.assertEqual(s,3)
        self.assertEqual(n,3)
        np.testing.assert_array_equal(t,np.array([0,0,1,1]))
        np.testing.assert_array_equal(d,np.array([1,2,2,3]))

    def test_steps(self):

        self.assertEqual(self.rv.steps,3)

    def test_update(self):

        self.assertTrue(self.rv.update())
        self.assertTrue(self.rv.update())
        self.assertTrue(self.rv.update())
        self.assertFalse(self.rv.update())

    def test_y_ticks(self):
        self.rv = RasterVisualiser(self.data, y_ticks=True)

    def test_short_depth(self):

        r = RasterVisualiser(self.data, depth=1)
        np.testing.assert_array_equal(r.to_get(), np.array([True,True,False,False]))
        r.update()
        np.testing.assert_array_equal(r.to_get(), np.array([False,False,True,True]))

    def test_reset_zoom(self):

        self.rv.reset_zoom(10)
        lims = self.rv.plot.viewRange()
        self.assertListEqual(lims[0],[-10,0])
