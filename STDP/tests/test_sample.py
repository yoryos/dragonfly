__author__ = 'juancarlosfarah'

from Helper.Configurer import Configurer
import unittest
import math
from copy import deepcopy

import os
import STDP.sample as sg
import numpy as np

TEST_DIR = os.path.dirname(__file__)

class SampleGeneratorTests(unittest.TestCase):

    def setUp(self):
        """
        Creates a sample to test on.
        :return: None.
        """
        configurer = Configurer(os.path.join(TEST_DIR,'config/sampleTestsConfig.ini'))
        self.config = configurer.config_section_map('Test1')

        sample_directory = os.path.join(TEST_DIR,self.config['sample_directory'])
        # Generate Sample
        sample_duration = 1000
        num_neurons = 500
        self.sample = sg.Sample(self.config['sample_duration'], self.config['start_time'],
                                self.config['end_time'],sample_directory,rate_max = self.config['rate_max'],
                            reading_pattern = self.config['reading_pattern'],pattern_filename = self.config['pattern_filename'],                            pattern_path = self.config['pattern_path'] )

    def test_pattern_duration(self):
        """
        Tests if the pattern duration property matches the length
        of the first pattern in the patterns property.
        :return: None.
        """

        # Create and insert patterns.
        self.sample.generate_sample()
        self.sample.generate_patterns()
        self.sample.insert_patterns()

        target_duration = self.sample.pattern_duration
        #print  self.sample.patterns[0].shape

        real_duration = self.config['end_time'] - self.config['start_time']
        print ""
        print "Pattern Duration"
        print "================"
        print "Expected: ", target_duration
        print "Actual:   ", real_duration
        print ""

        self.failUnless(target_duration == real_duration)

    def test_avg_hz(self):
        """
        Tests if the sample has the appropriate average firing rate.
        :return: None.
        """

        # Set up.
        dt = 0.001
        bin_size = 10
        target_rate = 54.0
        target_sigma = 2.0
        threshold = 100
        firing_rates = []

        self.sample.generate_sample()
        spike_trains = self.sample.spike_trains
        num_neurons = self.sample.num_neurons
        actual_rate = 0

        # Sample rate over time bins.
        for i in range(0, spike_trains.shape[1] - bin_size, bin_size):
            total_spikes = np.sum(np.sum(spike_trains[:, i: i + bin_size]))
            actual_rate = total_spikes / (num_neurons * bin_size) / dt
            firing_rates.append(actual_rate)

        # Calculate mean and standard deviation.
        mean_rate = np.mean(firing_rates)
        actual_sigma = np.std(firing_rates)

        # Print Mean and Standard Deviation
        print ""
        print "Mean Firing Rate without Noise"
        print "================================"
        print "Expected: ", target_rate
        print "Actual:   ", actual_rate
        print ""
        print "Standard Deviation"
        print "=================="
        print "Expected: ", target_sigma
        print "Actual:   ", actual_sigma
        print ""

        # Ensure that the actual figures are within the threshold.
        self.failUnless(math.fabs(target_rate - mean_rate) < threshold)
        self.failUnless(math.fabs(target_sigma - actual_sigma) < threshold)
        return

    def test_avg_hz_with_noise(self):
        """
        Tests the average firing rate of the sample with noise.
        :return: None.
        """

        # Set up.
        dt = 0.001
        bin_size = 10
        target_rate = 64.0
        target_sigma = 2.0
        threshold = 100
        firing_rates = []

        self.sample.generate_sample()
        spike_trains = self.sample.spike_trains
        num_neurons = self.sample.num_neurons
        actual_rate = 0

        # Add noise.
        self.sample.add_noise()

        # Sample rate over time bins.
        for i in range(0, spike_trains.shape[1] - bin_size, bin_size):
            total_spikes = np.sum(np.sum(spike_trains[:, i: i + bin_size]))
            actual_rate = total_spikes / (num_neurons * bin_size) / dt
            firing_rates.append(actual_rate)

        # Calculate mean and standard deviation.
        mean_rate = np.mean(firing_rates)
        actual_sigma = np.std(firing_rates)

        # Print Mean and Standard Deviation
        print ""
        print "Mean Firing Rate with Noise"
        print "==========================="
        print "Expected: ", target_rate
        print "Actual:   ", actual_rate
        print ""
        print "Standard Deviation"
        print "=================="
        print "Expected: ", target_sigma
        print "Actual:   ", actual_sigma

        # Ensure that the actual figures are within the threshold.
        self.failUnless(math.fabs(target_rate - mean_rate) < threshold)
        self.failUnless(math.fabs(target_sigma - actual_sigma) < threshold)
        return

    def spacing(self):
        """

        :return:
        """
        return

    def test_empty_pattern(self):
        """
        Test that empty pattern check is correct.
        :return: None.
        """
        self.sample.generate_patterns()
        self.failUnless(np.sum(self.sample.spike_trains) == 0)
        return

    def test_save(self):
        sample = self.sample
        sample.generate_sample()
        sample.generate_patterns()
        sample.filename == None
        extension = ".npz"
        filename = "/samples/{}_{}_{}_{}_{}_{}_{}".format(sample.num_patterns,
                                                         sample.num_neurons,
                                                         sample.duration,
                                                         sample.pattern_duration,
                                                         sample.rep_ratio,
                                                         sample.inv_ratio,
                                                         sample.noise)

        path = TEST_DIR + "/samples"

        print '1: ', filename
        sample.save(path)
        self.failUnless(os.path.exists(TEST_DIR + filename  + extension))

        # Now save with custom filename.
        filename = "test_save"

        # sample.filename = filename
        # sample.save(path)
        # self.failUnless(os.path.exists(path + "/" + filename +  extension))

     # def test_custom_pattern(self):

     #    # Load sample custom pattern.
     #    p = np.load(TEST_DIR + "/test/custom_pattern_500_50.npz")
     #    st = p['spike_trains']
     #    ps = [st]
     #    d = 1000
     #    num_neurons = 500
     #    self.sample = sg.Sample(d, patterns = ps, num_neurons = num_neurons, rep_ratio = 0.25)
     #    sample = self.sample

     #    # Expected values.
     #    expected_patterns = deepcopy(ps)
     #    expected_num_patterns = 1
     #    expected_pattern_duration = 50
     #    expected_num_buckets = math.floor(d / 50)
     #    expected_free_buckets = np.arange(expected_num_buckets - 1)

     #    # Assertions.
     #    for i in range(len(expected_patterns)):
     #        self.failUnless((expected_patterns[i] == sample.patterns[i]).all())
     #    self.failUnless(expected_num_patterns == len(sample.patterns))
     #    self.failUnless(expected_pattern_duration == sample.pattern_duration)
     #    self.failUnless(expected_num_buckets == sample.num_buckets)
     #    self.failUnless((expected_free_buckets == sample.free_buckets).all())

     #    # Append another pattern and run again.
     #    sample.load_pattern(st)
     #    expected_num_patterns = 2
     #    expected_patterns.append(st)

     #    # Assertions.
     #    for i in range(len(expected_patterns)):
     #        self.failUnless((expected_patterns[i] == sample.patterns[i]).all())
     #    self.failUnless(expected_num_patterns == len(sample.patterns))

    def tearDown(self):
        """
        Resets the sample for the tests.
        :return:
        """
        self.sample = None


def main():
    unittest.main()

if __name__ == '__main__':
    main()
