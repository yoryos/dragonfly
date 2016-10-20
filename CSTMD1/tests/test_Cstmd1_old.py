from CSTMD1.Cstmd1 import Cstmd1
from mock import patch
import unittest
import os

TEST_DIR = os.path.dirname(__file__)

class TestCstmd1(unittest.TestCase):
    def setUp(self):
        # Try key combinations of each debug mode and dummy file
        #
        self.cstmd1_00 = Cstmd1(morphology_path=TEST_DIR,
                                morphology_prefix="testdummySWC_",
                                number_of_neurons=2,
                                dt=0.05,
                                estmd_dim=(1, 1),
                                max_t=10,
                                synapses=True,
                                synapses_f_name="testdummySynapses.dat",
                                soma_electrodes=True,
                                debug_mode=0)

        self.cstmd1_01 = Cstmd1(morphology_path=TEST_DIR,
                                morphology_prefix="testdummySWC_",
                                number_of_neurons=2,
                                dt=0.05,
                                estmd_dim=(1, 1),
                                max_t=10,
                                synapses=True,
                                synapses_f_name="testdummySynapses.dat",
                                soma_electrodes=True,
                                debug_mode=1)

        self.cstmd1_02 = Cstmd1(morphology_path=TEST_DIR,
                                morphology_prefix="testdummySWC_",
                                number_of_neurons=2,
                                dt=0.05,
                                estmd_dim=(1, 1),
                                max_t=10,
                                synapses=True,
                                synapses_f_name="testdummySynapses.dat",
                                soma_electrodes=True,
                                debug_mode=2)

        self.cstmd1_03 = Cstmd1(morphology_path=TEST_DIR,
                                morphology_prefix="testdummySWC_",
                                number_of_neurons=2,
                                dt=0.05,
                                estmd_dim=(1, 1),
                                max_t=10,
                                synapses=True,
                                synapses_f_name="testdummySynapses.dat",
                                electrodes_f_name='testdummyElectrodes.dat',
                                soma_electrodes=True,
                                debug_mode=2)

        self.cstmd1_04 = Cstmd1(morphology_path=TEST_DIR,
                                morphology_prefix="testdummySWC_",
                                number_of_neurons=2,
                                dt=0.05,
                                estmd_dim=(2, 2),
                                max_t=10,
                                synapses=True,
                                synapses_f_name="testdummySynapses.dat",
                                estmd_map_f_name='testdummyESTMDMapping.dat',
                                soma_electrodes=True,
                                debug_mode=2)

        self.cstmd1_05 = Cstmd1(morphology_path=TEST_DIR,
                                morphology_prefix="testdummySWC_",
                                number_of_neurons=2,
                                dt=0.05,
                                estmd_dim=(2, 2),
                                max_t=10,
                                synapses=True,
                                synapses_f_name="testdummySynapses.dat",
                                estmd_map_f_name='testdummyESTMDMapping.dat',
                                soma_electrodes=True,
                                debug_mode=3)

    def test_constructors(self):
        Cstmd1(morphology_path=TEST_DIR,
               morphology_prefix="testdummySWC_",
               number_of_neurons=2,
               dt=0.05,
               estmd_dim=(1, 1),
               max_t=10,
               soma_electrodes=False,
               number_of_new_electrodes=1)

        with self.assertRaises(AssertionError):
            Cstmd1(morphology_path=TEST_DIR,
                   morphology_prefix="testdummySWC_",
                   number_of_neurons=0,
                   dt=0.05,
                   estmd_dim=(1, 1),
                   max_t=10,
                   soma_electrodes=True)

        with self.assertRaises(IOError):
            Cstmd1(morphology_path="bad_path",
                   morphology_prefix="testdummySWC_",
                   number_of_neurons=2,
                   dt=0.05,
                   estmd_dim=(1, 1),
                   max_t=10,
                   soma_electrodes=True)

        with self.assertRaises(IOError):
            Cstmd1(morphology_path=TEST_DIR,
                   morphology_prefix="testdummySWC_",
                   number_of_neurons=100,
                   dt=0.05,
                   estmd_dim=(1, 1),
                   max_t=10,
                   soma_electrodes=True)

        with self.assertRaises(IOError):
            Cstmd1(morphology_path=TEST_DIR,
                   morphology_prefix="testdummySWC_",
                   number_of_neurons=100,
                   dt=0.05,
                   estmd_dim=(0, 0),
                   max_t=10,
                   soma_electrodes=True)

    def tearDown(self):
        pass

    def test_run_no_estmd(self):
        self.assertTrue(len(self.cstmd1_00.step(0.1)) > 0)
        self.assertTrue(len(self.cstmd1_01.step(0.1)) > 0)
        self.assertTrue(len(self.cstmd1_02.step(0.1)) > 0)
        self.assertTrue(len(self.cstmd1_03.step(0.1)) > 0)
        self.assertTrue(len(self.cstmd1_04.step(0.1)) > 0)
        self.assertTrue(len(self.cstmd1_05.step(0.1)) > 0)
        # Cuda side assert on ensuring time is less than max!

    def test_run_estmd(self):
        self.assertTrue(len(self.cstmd1_00.step(0.1, [(0, 0, 1)])) > 0)
        self.assertTrue(len(self.cstmd1_01.step(0.1, [(0, 0, 1)])) > 0)
        self.assertTrue(len(self.cstmd1_02.step(0.1, [(0, 0, 1)])) > 0)
        self.assertTrue(len(self.cstmd1_03.step(0.1, [(0, 0, 1)])) > 0)
        self.assertTrue(len(self.cstmd1_04.step(0.1, [(0, 0, 1)])) > 0)
        self.assertTrue(len(self.cstmd1_05.step(0.1, [(0, 0, 1)])) > 0)

    def test_save_spikes(self):

        path = os.path.join(TEST_DIR,"testOutput.dat")
        self.cstmd1_00.save_spikes(path)
        self.cstmd1_01.save_spikes(path)
        self.cstmd1_02.save_spikes(path)
        self.cstmd1_03.save_spikes(path)
        self.cstmd1_04.save_spikes(path)
        self.cstmd1_05.save_spikes(path)
        assert os.path.exists(path)
        #This is a useless test need to assert that the file is actually created


    @patch("matplotlib.pyplot.show")
    def test_x(self, mock_show):
        mock_show.return_value = None
        self.cstmd1_00.step(0.1)
        self.cstmd1_00.plot_spikes(0.05)
        self.cstmd1_01.step(0.1)
        self.cstmd1_01.plot_spikes(0.05)
        self.cstmd1_02.step(0.1)
        self.cstmd1_02.plot_spikes(0.05)
        self.cstmd1_03.step(0.1)
        self.cstmd1_03.plot_spikes(0.05)
        self.cstmd1_04.step(0.1)
        self.cstmd1_04.plot_spikes(0.05)
        self.cstmd1_05.step(0.1)
        self.cstmd1_05.plot_spikes(0.05)
