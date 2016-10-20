from CSTMD1.Cstmd1 import Cstmd1
from mock import patch
import unittest
import os
import numpy as np

TEST_DIR = os.path.dirname(__file__)


class TestCstmd1(unittest.TestCase):
    def setUp(self):
        self.morphology_parameters = {
            "morphology_path": TEST_DIR,
            "morphology_prefix": "testdummySWC_",
            "number_of_neurons": 2,
            "homogenise": False,
            "synapses": True,
            "synapses_file_name": "testdummySynapses.dat",
            "number_of_synapses": None,
            "minimum_synapse_distance": None,
            "soma_electrodes": True,
            "electrodes_f_name": None,
            "number_of_electrodes": None,
            "random_electrodes": False,
            "estmd_map_f_name": "testdummyESTMDMapping.dat",
            "topological": False
        }

        self.sim_parameters = {
            "v_rest": 0.0,
            "cm": 1.0,
            "gbar_na": 120.0,
            "gbar_k": 36.0,
            "gbar_l": 0.3,
            "e_na": 115.0,
            "e_K": -12.0,
            "e_l": 10.613,
            "sra": 0.1,
            "r": 0.0002,
            "l": 0.00001,
            "tau_gaba": 400.0,
            "e_gaba": 0.0,
            "estmd_gain": 10.0,
            "spike_threshold": 50.0,
            "synapse_max_conductance": 1.0,
            "noise_stddev": 0.0
        }

    def test_default_constructor(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters, self.sim_parameters)

    def test_no_synapses(self):
        self.morphology_parameters["synapses"] = False
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters, self.sim_parameters)
        self.assertFalse(cstmd.enable_synapses)

    def test_homogenise_lengths(self):
        self.morphology_parameters["homogenise"] = True
        self.morphology_parameters["synapses"] = False
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters, self.sim_parameters)

    def test_get_new_synapses(self):
        self.morphology_parameters["synapses_file_name"] = None
        self.morphology_parameters["number_of_synapses"] = 2
        self.morphology_parameters["minimum_synapse_distance"] = 2

        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters, self.sim_parameters)

        self.assertEqual(cstmd.synapses.shape[0], 2)
        os.remove(os.path.join(TEST_DIR, "test_2_2_synapses.dat"))

    def test_enable_dump(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters, self.sim_parameters,
                       enable_spike_dump=True)

        self.assertTrue(hasattr(cstmd, "all_compartments"))

    def test_get_prebuild_electrodes(self):
        self.morphology_parameters["electrodes_f_name"] = "testdummyElectrodes.dat"
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        self.assertTrue(len(cstmd.electrodes), 2)

    def test_map_estmd_input(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        found = cstmd.map_estmd_input([(0, 0, 5), (0, 1, 10)])
        expected = np.array([[0, 5], [2, 10]])

        np.testing.assert_array_equal(found, expected)

    def test_step_with_input(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        status, spikes = cstmd.step(10, [(0, 0, 2)])

        self.assertTrue(status)
        self.assertTrue(len(spikes) > 0)

    def test_step_without_input(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        status, spikes = cstmd.step(1)

        self.assertTrue(status)
        self.assertTrue(len(spikes) == 0)

    def test_save_parameters(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        cstmd.save_parameters(TEST_DIR)

        sim = "CSTMD_Simulation_Parameters.dat"
        mor = "CSTMD_Morphology_Parameters.dat"

        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, sim)))
        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, mor)))

        os.remove(os.path.join(TEST_DIR, sim))
        os.remove(os.path.join(TEST_DIR, mor))

    def test_save_morphology(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        cstmd.save_morphology(TEST_DIR)
        elec = "electrodes.npy"
        syn = "synapses.npy"

        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, elec)))
        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, syn)))

        os.remove(os.path.join(TEST_DIR, elec))
        os.remove(os.path.join(TEST_DIR, syn))

    def test_save_graphs(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        cstmd.step(10, [(0, 0, 1)])

        splot1 = "spike_plot.svg"
        splot2 = "spike_plot.pdf"
        srp = "spike_rate_plot.svg"

        cstmd.save_graphs(TEST_DIR)

        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, splot1)))
        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, splot2)))
        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, srp)))

        os.remove(os.path.join(TEST_DIR, splot1))
        os.remove(os.path.join(TEST_DIR, splot2))
        os.remove(os.path.join(TEST_DIR, srp))

    def test_save_spikes(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        cstmd.step(1, [(0, 0, 1)])
        cstmd.step(1, [(0, 0, 1)])

        cstmd.save_spikes(TEST_DIR, "spks.dat")

        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, "spks.dat")))

        os.remove(os.path.join(TEST_DIR, "spks.dat"))

    def test_get_voltages_and_rv(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        cstmd.step(1, [(0, 0, 1)])

        self.assertEqual(cstmd.get_voltages().shape, (40, 9))
        m, n, h = cstmd.get_recovery_variables()
        self.assertEqual(m.shape, (40, 9))
        self.assertEqual(n.shape, (40, 9))
        self.assertEqual(h.shape, (40, 9))

    def test_plot_firing_rate(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        cstmd.step(1, [(0, 0, 1)])
        cstmd.plot_firing_rate(show=False, save_path=os.path.join(TEST_DIR, "fr.png"))

        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, "fr.png")))
        os.remove(os.path.join(TEST_DIR, "fr.png"))

    def test_plot_compartments(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)

        cstmd.step(1, [(0, 0, 1)])

        cstmd.plot_compartments([0, 1, 2],
                                show=False,
                                save_path=os.path.join(TEST_DIR, "c.png"),
                                names=["Compartment1", "Compartment2"])

        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, "c.png")))
        os.remove(os.path.join(TEST_DIR, "c.png"))

    def test_plot_synapses(self):
        cstmd = Cstmd1(1.0, 40, [2, 2], 20, "test",
                       self.morphology_parameters,
                       self.sim_parameters)
        cstmd.step(1, [(0, 0, 1)])

        cstmd.plot_synapse_compartments(TEST_DIR)
        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, "synapse_plt_3_6.svg")))
        self.assertTrue(os.path.isfile(os.path.join(TEST_DIR, "synapse_plt_8_5.svg")))

        os.remove(os.path.join(TEST_DIR, "synapse_plt_3_6.svg"))
        os.remove(os.path.join(TEST_DIR, "synapse_plt_8_5.svg"))
