import os
import unittest

import numpy as np

from CSTMD1.Morphology.CSTMD1Visualiser import CSTMD1Visualiser
from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection

TEST_SYN = os.path.join(os.path.dirname(__file__), "testdummySynapses.dat")
TEST_SPK = os.path.join(os.path.dirname(__file__), "testdummySpikes.dat")
TEST_ELC = os.path.join(os.path.dirname(__file__), "testdummyElectrodes.dat")
TEST_SWC_1 = os.path.join(os.path.dirname(__file__), "testdummySWC_0.swc")
TEST_SWC_2 = os.path.join(os.path.dirname(__file__), "testdummySWC_1.swc")
TEST_DIR = os.path.join(os.path.dirname(__file__))

class TestCSTMD1Visualiser(unittest.TestCase):

    def setUp(self):
        self.m1 = MultiCompartmentalNeuron()
        self.m1.construct_from_SWC(TEST_SWC_1, [0,0,-0.2],1)
        self.m2 = MultiCompartmentalNeuron()
        self.m2.construct_from_SWC(TEST_SWC_2, [0,0,-0.2],1)
        self.nc = NeuronCollection()
        self.nc.add_neuron(self.m1)
        self.nc.add_neuron(self.m2)
        self.nc.import_synapses_from_file(TEST_SYN)
        self.nc.import_electrodes_from_file(TEST_ELC)

    def test_steps(self):

        cstmd_vis = self.test_animated_visualiser()
        self.assertEqual(cstmd_vis.steps,4)

    def test_static_visualiser(self):

        cstmd1_vis = CSTMD1Visualiser(self.nc, synapses=True, plot_static_electrodes=True, expand=0, enable_run=False)
        self.assertEqual(len(cstmd1_vis.compartment_plots),2)
        return cstmd1_vis

    def test_animated_visualiser(self):
        self.nc.load_spikes_from_file(TEST_SPK)
        cstmd1_vis = CSTMD1Visualiser(self.nc, synapses=True, plot_static_electrodes=True, expand=5, spikes=True,
                                      enable_run=False)
        self.assertEqual(len(cstmd1_vis.compartment_plots),2)
        return cstmd1_vis

    def test_synapses_loaded(self):

        cstmd_vis = self.test_static_visualiser()

        _,m1_to_m2,_, m2_to_m1 = cstmd_vis.synapse_coordinates[0]
        np.testing.assert_array_equal(m1_to_m2,
                                      np.array([self.nc.get_compartment(3).midpoint().to_list() +
                                               self.nc.get_compartment(6).midpoint().to_list()]))
        np.testing.assert_array_equal(m2_to_m1,
                                      np.array([self.nc.get_compartment(5).midpoint().to_list() +
                                               self.nc.get_compartment(8).midpoint().to_list()]))

    def test_compartments_loaded(self):

        cstmd_vis = self.test_static_visualiser()
        m1_expected = np.array([c.start.to_list() + c.end.to_list() for c in self.m1.compartments])
        m2_expected = [c.start.to_list() + c.end.to_list() for c in self.m2.compartments]
        _,m1 = cstmd_vis.compartments[0]
        _,m2 = cstmd_vis.compartments[1]
        np.testing.assert_array_equal(m1,m1_expected)
        np.testing.assert_array_equal(m2,m2_expected)

    def test_static_electrodes(self):

        cstmd_vis = self.test_static_visualiser()
        _,e1 = cstmd_vis.electrodes[0]
        _,e2 = cstmd_vis.electrodes[1]
        self.assertEqual(e1,self.nc.get_compartment(self.nc.electrodes[0]).midpoint().to_list())
        self.assertEqual(e2,self.nc.get_compartment(self.nc.electrodes[1]).midpoint().to_list())

    def test_spike_loading(self):
        cstmd_vis = self.test_animated_visualiser()
        np.testing.assert_array_equal(cstmd_vis.electrode_size,np.zeros(2))
        np.testing.assert_array_equal(cstmd_vis.electrode_color,np.zeros((2,4)))
        expected = np.zeros((2,4)).astype(bool)
        expected[0,1] = True
        expected[1,2] = True
        np.testing.assert_array_equal(cstmd_vis.electrode_spikes,expected)

    def test_plotted_synapse(self):

        cstmd_vis = self.test_static_visualiser()
        self.assertEqual(len(cstmd_vis.synapse_plots),2)

    def test_plotted_compartments(self):
        cstmd_vis = self.test_static_visualiser()
        self.assertEqual(len(cstmd_vis.compartment_plots),2)

    def test_expanded_compartments(self):

        cstmd_vis_expanded = self.test_animated_visualiser()
        cstmd_vis_not_expanded = self.test_static_visualiser()
        np.testing.assert_array_equal(cstmd_vis_expanded.compartments[0][1],
                                      cstmd_vis_not_expanded.compartments[0][1])

        np.testing.assert_array_equal(cstmd_vis_not_expanded.compartments[1][1][:,[2,5]] + 5,
                                      cstmd_vis_expanded.compartments[1][1][:,[2,5]])

    def test_reset_to_start(self):

        cstmd_vis_expanded = self.test_animated_visualiser()

        cstmd_vis_expanded.update()
        self.assertEqual(cstmd_vis_expanded.index,1)
        cstmd_vis_expanded.reset_to_start()
        self.assertEqual(cstmd_vis_expanded.index,0)

    def test_update(self):

        cstmd_vis_expanded = self.test_animated_visualiser()

        cstmd_vis_expanded.update()

        np.testing.assert_array_equal(cstmd_vis_expanded.electrode_size,
                                      np.array([cstmd_vis_expanded.electrode_base_size,
                                                cstmd_vis_expanded.electrode_base_size]))

        np.testing.assert_array_equal(cstmd_vis_expanded.electrode_color,
                                      np.array([cstmd_vis_expanded.electrode_base_color,
                                                cstmd_vis_expanded.electrode_base_color]))
        cstmd_vis_expanded.update()

        np.testing.assert_array_equal(cstmd_vis_expanded.electrode_size,
                                      np.array([cstmd_vis_expanded.electrode_spike_size,
                                                cstmd_vis_expanded.electrode_base_size]))

        np.testing.assert_array_equal(cstmd_vis_expanded.electrode_color,
                                      np.array([cstmd_vis_expanded.electrode_spike_color,
                                                cstmd_vis_expanded.electrode_base_color]))

    def test_save(self):

        cstmd_vis_expanded = self.test_animated_visualiser()
        expected = os.path.join(TEST_DIR, "test")
        self.assertEqual(cstmd_vis_expanded.save(expected),1)

        self.assertTrue(os.path.isfile(expected + ".jpeg"))
        os.remove(expected + ".jpeg")

    def test_white(self):
        cstmd_vis_expanded =  CSTMD1Visualiser(self.nc, synapses=True, plot_static_electrodes=True, expand=0,
                                            enable_run=False, white=True)