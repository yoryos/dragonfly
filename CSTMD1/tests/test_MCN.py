import os
import unittest

import numpy as np
from mock import patch

from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron

TEST_SWC_1 = os.path.join(os.path.dirname(__file__), "testdummySWC_0.swc")


class TestMCN(unittest.TestCase):
    def setUp(self):
        self.mcn = MultiCompartmentalNeuron(0)
        self.mcn.construct_from_SWC(TEST_SWC_1, (0, 0, -1), 1)

    def test_wrong_SWC_path(self):

        self.mcn2 = MultiCompartmentalNeuron()

        with self.assertRaises(IOError):
            self.mcn2.construct_from_SWC("WRONG", (0, 0, -1))

    def test_correctly_loaded(self):

        self.assertEqual(self.mcn.number_of_compartments(), 6, "Not all compartments found")

    def test_connection_matrix(self):

        found = self.mcn.connection_matrix()

        expected = np.array([[1, -1, 0, 0, 0, 0],
                             [-1, 4, -1, -1, -1, 0],
                             [0, -1, 3, -1, -1, 0],
                             [0, -1, -1, 3, -1, 0],
                             [0, -1, -1, -1, 4, -1],
                             [0, 0, 0, 0, -1, 1]])

        np.testing.assert_array_equal(expected, found, "Wrong connection martix")

    def test_compartment_data(self):

        length, radii = self.mcn.compartment_data()
        np.testing.assert_array_equal(length, [1, 1, 1, 2, 0.5, 0.5], "Length data wrong")
        np.testing.assert_array_equal(radii, [None] * 6, "Radii information shouldn't exist yet")

    def test_rebase(self):

        del self.mcn.compartments[0]
        self.assertEqual(self.mcn.number_of_compartments(), 5, "Compartment could not be removed")

        self.mcn.rebase_idx()

        for i, c in enumerate(self.mcn.compartments):
            self.assertEqual(i, c.idx, "Idx inconsistency")

    def test_steps_to_root(self):

        steps = [0, 1, 2, 2, 2, 3]

        for i, c in enumerate(self.mcn.compartments):
            self.assertEqual(c.steps_to_root(), steps[i], "Wrong number of steps to root for compartment" + str(i))

    def test_homoginise(self):

        self.mcn.homogenise_lengths(low=0.9, high=1.1)

        self.assertEqual(self.mcn.number_of_compartments(), 6, "Incorrect number of new compartments")
        length, _ = self.mcn.compartment_data()

        np.testing.assert_array_equal(length, [1, 1, 1, 1, 1, 1], "Incorrect homogenised lengths")

    def test_homoginise_median_stdev(self):

        self.mcn.homogenise_lengths(offset=0.1)

        self.assertEqual(self.mcn.number_of_compartments(), 6, "Incorrect number of new compartments")
        length, _ = self.mcn.compartment_data()

        np.testing.assert_array_equal(length, [1, 1, 1, 1, 1, 1], "Incorrect homogenised lengths")

    def test_radii(self):

        self.mcn.generate_radii(2, 5)

        expected = np.array([0, 1, 2, 2, 2, 3], dtype=float)
        expected[expected != 0] = 5 / (2.0 * expected[expected != 0])
        expected[expected == 0] = 5

        _, radii = self.mcn.compartment_data()
        np.testing.assert_array_equal(radii, expected, "Radii information is wrong")

    def test_string(self):
        self.assertEqual(str(self.mcn), "[idx: 0, compartments: 6, somaID: 0: (0.0,0.0,-1.0) (0.0,0.0,0.0)]")

    def test_length_median(self):

        self.assertEqual(self.mcn.median_length(), 1, "Incorrect median length")

    def test_length_stDev(self):

        self.assertEqual(self.mcn.length_stdev(), np.std(np.array([0.5, 0.5, 1, 1, 1, 2])),
                         "Incorrect stdev of lengths")

    @patch("matplotlib.pyplot.show")
    def test_plot_compartments(self, mock_show):
        mock_show.return_value = None
        self.mcn.plot_compartment_data(True,False)
