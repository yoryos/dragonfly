__author__ = 'zoelandgraf'

from copy import deepcopy
import math
import unittest
from Helper.Configurer import Configurer
#import stdp.neuron as neuron
#from stdp import simulation
from STDP.stdp import Stdp
import numpy as np
import os

TEST_DIR = os.path.dirname(__file__)

class StdpTests(unittest.TestCase):

    def setUp(self):

       """
        :return:
       """

       configurer = Configurer(os.path.join(TEST_DIR,'config/stdpTestsConfig.ini'))
       config = configurer.config_section_map('Test1')

       self.sample_path = os.path.join(TEST_DIR,'samples/')
       self.figure_path = os.path.join(TEST_DIR,'figures/')
       self.parameters_path = os.path.join(TEST_DIR,'parameters/')
       self.weights_path = os.path.join(TEST_DIR,'weights/')


       self.stdp = Stdp(config['run_id'],
                            config['num_afferents'], config['num_neurons'], config['weights'],
                            config['load_weights'], config['weights_list'], config['weights_path'],
                            config['training'], config['output_type'],
                            config['verbose'],
                            config['historic_weights_path'], config['saving_historic_weights'],
                            config['alpha'], config['theta'], config['a_plus'], config['a_ratio'],
                            config['tau_m'], config['tau_s'], config['t_plus'], config['t_minus'], config['k'], config['k1'], config['k2']
                            )
       self.stdp.test_with_pattern('short_testing_sample','STDP/tests/samples/')

       self.stdp.test_with_pattern('short_testing_sample','STDP/tests/samples/')
       self.constructors = []
       for t in ['Test2', 'Test3']:
            configurer = Configurer(os.path.join(TEST_DIR,'config/stdpTestsConfig.ini'))
            config = configurer.config_section_map(t)

            self.constructors.append(Stdp(config['run_id'],
                                 config['num_afferents'], config['num_neurons'], config['weights'],
                                 config['load_weights'], config['weights_list'], config['weights_path'],
                                 config['training'], config['output_type'],
                                 config['verbose'],
                                 config['historic_weights_path'], config['saving_historic_weights'],
                                 config['alpha'], config['theta'], config['a_plus'], config['a_ratio'],
                                 config['tau_m'], config['tau_s'], config['t_plus'], config['t_minus'], config['k'], config['k1'], config['k2']
                                 ))


       config = configurer.config_section_map('Test4')

       self.stdp2 = Stdp(config['run_id'],
                             config['num_afferents'], config['num_neurons'], config['weights'],
                             config['load_weights'], config['weights_list'], config['weights_path'],
                             config['training'], config['output_type'],
                             config['verbose'],
                             config['historic_weights_path'], config['saving_historic_weights'],
                             config['alpha'], config['theta'], config['a_plus'], config['a_ratio'],
                             config['tau_m'], config['tau_s'], config['t_plus'], config['t_minus'], config['k'], config['k1'], config['k2']
                             )
       self.stdp2.test_with_pattern('short_testing_sample',self.sample_path)

       for stdp in self.constructors:
           stdp.test_with_pattern('short_testing_sample',self.sample_path)
       return


    def test_bad_weights_path(self):

        configurer = Configurer(os.path.join(TEST_DIR,'config/stdpTestsConfig.ini'))
        config = configurer.config_section_map('Test5')

        self.assertRaises(IOError, lambda: Stdp(config['run_id'],
                             config['num_afferents'], config['num_neurons'], config['weights'],
                             config['load_weights'], config['weights_list'], config['weights_path'],
                             config['training'], config['output_type'],
                             config['verbose'],
                             config['historic_weights_path'], config['saving_historic_weights'],
                             config['alpha'], config['theta'], config['a_plus'], config['a_ratio'],
                             config['tau_m'], config['tau_s'], config['t_plus'], config['t_minus'], config['k'], config['k1'], config['k2']
                             ) )

    def test_test_with_pattern(self):

        """
        """
        for stdp in self.constructors:

            stdp.test_with_pattern('short_testing_sample_no_extras',self.sample_path)
            stdp.test_with_pattern('short_testing_sample',self.sample_path)



        return


    def test_produce_spike_array(self):

        """
        """


        spike_indices = np.array([1,3,4,5,9])

        good_spikes = np.array([0., 1., 0., 1., 1., 1., 0., 0., 0., 1.])

        for stdp in self.constructors:
            spikes = stdp.produce_spike_array(spike_indices)

            self.failUnless(np.array_equal(spikes, good_spikes))
        return

    def test_step_with_training(self):
        """
        Checks that stdp.step
        returns the correct output
        :return:
        """

        spike_indices = np.array([1,3,4,5,9])

        for stdp in self.constructors:
            output_spikes = stdp.step(spike_indices)
            self.assertTrue(len(output_spikes) == 4)
            np.testing.assert_array_equal(output_spikes,np.array([False,False,False,False]))
        return

    def test_step_without_training(self):

        num_afferents = 10

        spike_indices = np.array([1,3,4,5,9])

        for stdp in self.constructors:
            output_spikes = stdp.step(spike_indices)
            self.failUnless(len(output_spikes) == 4)
            np.testing.assert_array_equal(output_spikes, np.array([False,False,False,False]))
        return

    def test_step_without_training_withHistoricWeights(self):

        num_afferents = 10

        spike_indices = np.array([1,3,4,5,9])
        for stdp in self.constructors:
            stdp.sampling_interval = 3
            for n in stdp.neurons:
                n.historic_weights = np.random.normal(0.475, 0.1, (num_afferents, 1))

        for stdp in self.constructors:
            output_spikes = stdp.step(spike_indices)
            self.failUnless(len(output_spikes) == 4)
            np.testing.assert_array_equal(output_spikes,np.array([False,False,False,False]))
        return

    # def test_step_with_more_afferents(self):
    #
    #
    #     for stdp in self.constructors:
    #         self.Stdp_test = Stdp(1000,1)
    #
    #         for j in xrange(1):
    #             #test_pattern = Stdp_test.sim.spike_trains[:,j]
    #             test_pattern = np.ones((1000))
    #             num_afferents = test_pattern.shape[0]
    #             test_indices = []
    #
    #             for i in xrange(len(test_pattern)):
    #                 if test_pattern[i] == 1:
    #                     test_indices.append(i)
    #
    #                     spike_out  = self.Stdp_test.step(test_indices, 1000)
    #                     self.failUnless(True in spike_out)
    #     return


    def test_step_with_time_delta(self):

        for stdp in self.constructors:
            for n in stdp.neurons:

                n.spike_times = [-1,-1]

                spike_indices = np.array([1,3,4,5,9])
                spikes = stdp.produce_spike_array(spike_indices)
        return


    def test_calculate_metric(self):


        metric = self.stdp.calculate_metric(50)
        print metric
        self.assertTrue((metric[0] >= 0.0) and (metric[0] <= 1.0))
        return

    def test_calculate_metric2(self):


        metric = self.stdp.calculate_metric(50,pybo_mode = True)
        print metric
        self.assertTrue((metric >= 0.0) and (metric <= 1.0))
        return

    def test_plot_membrane_potential(self):

        sim_len = self.stdp.spike_trains.shape[1]
        for j in xrange(sim_len):
            test_pattern = deepcopy(self.stdp.spike_trains[:,j])
            test_indices = []

            #transformation of pattern_input into indices (simulating cstmd1 output)
            for i in xrange(len(test_pattern)):
                if test_pattern[i] == 1:
                    test_indices.append(i)

            self.stdp.step(test_indices,output = bool)

            
        self.stdp.plot_membrane_potential(verbose = False, parameter_config_num = 1, path = self.figure_path)
        return

    def test_plot_membrane_potential2(self):

        sim_len = self.stdp.spike_trains.shape[1]
        for j in xrange(sim_len):
            test_pattern = deepcopy(self.stdp.spike_trains[:,j])
            test_indices = []

            #transformation of pattern_input into indices (simulating cstmd1 output)
            for i in xrange(len(test_pattern)):
                if test_pattern[i] == 1:
                    test_indices.append(i)

            self.stdp.step(test_indices,output = bool)


        self.stdp.plot_membrane_potential(verbose = True, parameter_config_num = 1, path = self.figure_path)
        return


    def test_save_current_weights(self):

        for stdp in self.constructors:
            stdp.save_current_weights(self.weights_path)


    def test_save_parameters(self):

        for stdp in self.constructors:
            stdp.save_parameters(self.parameters_path)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
