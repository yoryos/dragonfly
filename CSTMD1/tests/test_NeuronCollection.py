import os
import unittest
from itertools import product

import numpy as np
from mock import patch

from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
from Helper.Vectors import Vector_2D

TEST_SWC_1 = os.path.join(os.path.dirname(__file__), "testdummySWC_0.swc")
TEST_SWC_2 = os.path.join(os.path.dirname(__file__), "testdummySWC_1.swc")
TEST_SYN = os.path.join(os.path.dirname(__file__), "testdummySynapses.dat")
TEST_VOL = os.path.join(os.path.dirname(__file__), "testdummyVoltage.dat")
TEST_SPK = os.path.join(os.path.dirname(__file__), "testdummySpikes.dat")
TEST_ELC = os.path.join(os.path.dirname(__file__), "testdummyElectrodes.dat")
TEST_ETM = os.path.join(os.path.dirname(__file__), "testdummyESTMDMapping.dat")

class TestNeuronCollection(unittest.TestCase):

    def setUp(self):

        self.mcn1 = MultiCompartmentalNeuron()
        self.mcn1.construct_from_SWC(TEST_SWC_1, (0, 0, -1))

        self.mcn2 = MultiCompartmentalNeuron()
        self.mcn2.construct_from_SWC(TEST_SWC_2, (1, 0, 0),1)

        self.nc = NeuronCollection()
        self.nc.add_neuron(self.mcn1)
        self.nc.add_neuron(self.mcn2)

    def test_get_compartment(self):

        self.assertEqual(self.nc.get_compartment(6), self.mcn2.compartments[0], "Couldn't get the right compartment")

    def test_get_wrong_compartment(self):

        with self.assertRaises(AssertionError):
            self.nc.get_compartment(100)

    def test_compartment_offsets(self):
        np.testing.assert_array_equal(self.nc.collection_compartment_Idx_offset(),
                                      np.array([0, 6, 9]),
                                      "Compartment offsets are wrong")

    def test_cstmd1_electrical_connections(self):

        expected = np.array([[ 1, -1,  0,  0,  0,  0,  0,  0,  0],
                             [-1,  4, -1, -1, -1,  0,  0,  0,  0],
                             [ 0, -1,  3, -1, -1,  0,  0,  0,  0],
                             [ 0, -1, -1,  3, -1,  0,  0,  0,  0],
                             [ 0, -1, -1, -1,  4, -1,  0,  0,  0],
                             [ 0,  0,  0,  0, -1,  1,  0,  0,  0],
                             [ 0,  0,  0,  0,  0,  0,  1, -1,  0],
                             [ 0,  0,  0,  0,  0,  0, -1,  2, -1],
                             [ 0,  0,  0,  0,  0,  0,  0, -1,  1]])
        np.testing.assert_array_equal(expected,
                                      self.nc.cstmd1_sim_get_electical_connections(),
                                      "Wrong electrical connection matrix found")

    def test_import_estmd_mapping(self):

        self.nc.import_estmd1_mapping_from_file(TEST_ETM, (2,2))

        found = self.nc.cstmd1_sim_get_estmd1_mapping(2, 2, generate_new = False)

        self.assertDictEqual({(0,0):0,(0,1):1,(1,0):2,(1,1):3}, found, "ESTMD mapping not imported correctly")

    def test_import_electrodes(self):

        self.nc.import_electrodes_from_file(TEST_ELC)
        expected = np.array([3,6])

        np.testing.assert_array_equal(self.nc.electrodes, expected, "Electrodes not imported properly")

        np.testing.assert_array_equal(self.nc.cstmd1_sim_get_electrodes(generate_new = False), expected,
         "Could not retreive loaded electrodes")

    def test_map_electrodes_to_compartments(self):

        expected = [self.nc.neurons[0].compartments[3], self.nc.neurons[1].compartments[0]]
        self.nc.import_electrodes_from_file(TEST_ELC)
        found = self.nc.map_electrodes_to_compartments()
        self.assertListEqual(expected, found, "Incorrect mapping of electrodes")

    def test_generate_electrodes(self):

        found = self.nc.cstmd1_sim_get_electrodes(generate_new = True, random = True, number = 4)
        self.assertEqual(len(found), 4, "Not enough electrodes found")

        found =  self.nc.cstmd1_sim_get_electrodes(generate_new = True, random = False, number = 9)
        np.testing.assert_array_equal(found, np.arange(0,9), "Could not generate non random electrodes")

    def test_generate_soma_electrodes(self):

        np.testing.assert_array_equal(self.nc.cstmd1_sim_get_electrodes(generate_new=True, soma = True),
                                      np.array([0,6]))

    def test_get_neigbour_idx(self):

        np.testing.assert_array_equal(self.nc.get_neighbour_idx(2), np.array([1,3,4]))

    def test_get_global_idx(self):

        self.assertEqual(self.nc.get_global_idx(self.mcn1.compartments[2]), 2)
        self.assertEqual(self.nc.get_global_idx(self.mcn2.compartments[2]), 8)

    def test_get_all_compartments(self):

        self.assertEqual(len(self.nc.get_all_compartments(include_axon=True)), 9)

    def test_pixel_midpoints(self):

        expected = [Vector_2D(i,j) for i,j in product([-1.3125, 0.0625], [-0.25, 0.25, 0.75])]
        self.assertListEqual(expected, self.nc.pixel_midpoints(3,2, include_axon=True))

    def test_topological_mapping(self):

        self.assertDictEqual(self.nc.topological_mapping(2,1),{(0,1):0, (0,0):1})

    @patch("matplotlib.pyplot.show")
    def test_plot_plan(self, mock_show):

        self.nc.plot_plan()

    @patch("matplotlib.pyplot.show")
    def test_plotting_estmd_mapping(self, mock_show):

        self.nc.plot_estmd_mapping_from_file(2,2,TEST_ETM)


    def test_NumberOfNeurons(self):

        self.assertIn(self.mcn1, self.nc.neurons, "MCN1 not added to NC")
        self.assertIn(self.mcn2, self.nc.neurons, "MCN2 not added to NC")

    def test_NoSynapses(self):
        self.assertEqual(self.nc.generate_synapses(0, 1, 2, 1), ([0, 1, []],[1, 0, []]))

    def test_Synapses(self):

        f = self.nc.cstmd1_sim_get_synapses(True, number_between_adjacent = 2, min_distance=2)
        self.assertEqual(len(f), 2, "All synapses not found")

    def test_compartment_data(self):

        expected_radii = np.array([0, 1, 2, 2, 2, 3, 0, 1, 2], dtype=float)
        expected_radii[expected_radii != 0] = 5 / (2.0 * expected_radii[expected_radii != 0])
        expected_radii[expected_radii == 0] = 5
        expected_lengths = np.array([1,1,1,2,0.5,0.5,1,1,2])

        self.mcn1.generate_radii(2,5)
        self.mcn2.generate_radii(2,5)

        data = self.nc.cstmd1_sim_get_radii_lengths()

        np.testing.assert_array_equal(data[0], expected_lengths, "Unexpected Lengths")
        np.testing.assert_array_equal(data[1], expected_radii, "Unexpected Radii")


    def test_get_median_length(self):

        self.assertEqual(self.nc.cstmd1_sim_get_median_length(), 1.0)

    def test_import_synapses_from_file(self):

        self.nc.import_synapses_from_file(TEST_SYN)

        expected = [[0,1,[(self.mcn1.compartments[3],self.mcn2.compartments[0])]],
                    [1,0,[(self.mcn2.compartments[2],self.mcn1.compartments[5])]]]

        self.assertListEqual(self.nc.synapse_groups,
                             expected, "Synapses not loaded properly")


    def test_cstmd_get_synapses(self):

        self.nc.import_synapses_from_file(TEST_SYN)
        expected = np.array([[3,6],[8,5]])
        np.testing.assert_array_equal(expected, self.nc.cstmd1_sim_get_synapses(generate_new = False))

    def test_SynapsesFail(self):

       with self.assertRaises(AssertionError):
            self.nc.generate_synapses(0, 2, 10, 10)


    def test_incremental_load_spikes(self):

        spikes = np.array([[False, False],[True,False]])
        self.nc.import_electrodes_from_file(TEST_ELC)
        self.nc.load_electrode_data(spikes[0])

        self.assertEqual(self.mcn1.compartments[3].spike_record[0], spikes[0][0], "First spike could not be loaded")
        self.assertEqual(self.mcn2.compartments[0].spike_record[0], spikes[0][1], "First spike could not be loaded")

        self.nc.load_electrode_data(spikes[1])
        self.assertEqual(len(self.mcn1.compartments[3].spike_record), 2, "Wrong length spike record")


    def test_load_spikes(self):

        self.nc.import_electrodes_from_file(TEST_ELC)
        self.nc.load_spikes_from_file(TEST_SPK)

        compartment0 = np.array([False, True, False, False])
        compartment1 = np.array([False, False, True, False])

        np.testing.assert_array_equal(compartment0, self.mcn1.compartments[3].spike_record, "Could not bulk load "
                                                                                            "spikes")
        np.testing.assert_array_equal(compartment1, self.mcn2.compartments[0].spike_record, "Could not bulk load "
                                                                                            "spikes")

    def test_get_spikes_from_compartments(self):

        record0 = np.array([False, True, False, False])
        record1 = np.array([False, False, True, False])
        self.nc.electrodes = np.array([3,6])
        self.nc.get_compartment(3).spike_record = record0
        self.nc.get_compartment(6).spike_record = record1

        found =  self.nc.get_spikes_from_compartments([3,6])
        np.testing.assert_array_equal(found, np.column_stack((record0,record1)).astype(int))

    def test_load_voltages(self):

        self.nc.import_electrodes_from_file(TEST_ELC)
        self.nc.load_voltage_from_file(TEST_VOL)

        expected_voltages0 = np.array([0,2,3,4])
        expected_voltages1 = np.array([0,5,6,7])

        np.testing.assert_array_equal(expected_voltages0,
                                      self.mcn1.compartments[3].voltage_record,
                                      "Voltages not loaded correctly")

        np.testing.assert_array_equal(expected_voltages1,
                                      self.mcn2.compartments[0].voltage_record,
                                      "Voltages not loaded correctly")
