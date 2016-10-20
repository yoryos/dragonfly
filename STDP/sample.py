
__author__ = 'Juan Carlos Farah,' \
             'Panagiotis Almpouras'
__authoremail__ = 'juancarlos.farah14@imperial.ac.uk,' \
                  'panagiotis.almpouras12@imperial.ac.uk'
import os
import math
import sys
import matplotlib.pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


"""
Constants
=========
"""

SEED = 1                        # Seed for the random generator.

# Set seed.
np.random.seed(SEED)


class Sample(object):
    """
    Generates sample input spike trains.
    """
    def __init__(self, duration, start_time = 0, end_time = 50, sample_directory = None, rate_min = 0.0, rate_max = 90.0, patterns=None, num_patterns = 1,
                 num_neurons=1000, rep_ratio=0.20, inv_ratio=0.5, filename=None,
                 description=None, reading_pattern = False, pattern_filename = None, pattern_path = None):

        self.duration = duration
        self.start_time = start_time
        self.end_time = end_time
        self.pattern_duration = self.end_time - self.start_time
        self.num_neurons = num_neurons
        self.filename = filename
        self.num_patterns = num_patterns
        self.sample_directory = sample_directory
        self.patterns = []
        # self.description = description

        # Initialise containers
        self.spike_trains = np.zeros((num_neurons, self.pattern_duration), dtype=np.float)
        self.start_positions = []


        #handle reading pattern from file

        self.reading_pattern = reading_pattern
        self.pattern_filename = pattern_filename
        self.pattern_path = pattern_path


        # Handle custom patterns.
        if reading_pattern == False:
            if patterns is None:
                self.patterns = []
                self.num_patterns = 0

            else:
                if self.num_patterns > 1:
                    self.patterns = patterns
                    self.num_patterns = len(patterns)
                    self.pattern_duration = patterns[0].shape[1]
                else:
                    self.patterns = []
                    self.patterns.append(patterns)
                    self.pattern_duration = patterns.shape[1]

        # Duration of the spike pattern and buckets.
        self.num_buckets = math.floor(self.duration / self.pattern_duration)
        self.free_buckets = np.arange(self.num_buckets - 1)

        self.dt = 0.001             # Time step in seconds.
        self.rep_ratio = rep_ratio  # Ratio of pattern in the overall sample.
        self.inv_ratio = inv_ratio  # Ratio of afferents in the pattern.
        self.noise = 10.0           # Noise in Hz.

        self.r_min = rate_min            # Minimum firing rate in Hz.
        self.r_max = rate_max     #90.0           # Maximum firing rate in Hz.
        self.s_min = -1800.0        # Minimum negative rate of change in Hz/s.
        self.s_max = 1800.0         # Maximum positive rate of change in Hz/s.
        self.ds_min = -360.0        # Maximum change of rate of change in Hz/s.
        self.ds_max = 360.0         # Maximum change of rate of change in Hz/s.

    def generate_spike_train(self):
        """
        Generates spike train for one neuron.
        :return:
        """

        # Container for spike train.
        spike_train = np.zeros(self.duration)

        # Set initial rate of change.
        s = np.random.uniform(self.s_min, self.s_max)
        r = np.random.uniform(self.r_min, self.r_max)

        for i in range(0, self.duration):

            # Calculate probability of giving a spike at given time step.
            p = r * self.dt

            # Ensure that all afferent spikes at
            # least once every given pattern length.
            if i >= self.pattern_duration:
                spike_sum = np.sum(spike_train[i - self.pattern_duration: i])
            else:
                spike_sum = 1

            if spike_sum < 1:
                spike_train[i] = 1

            # Fire if p is > random number between 0 and 1.
            elif p > np.random.uniform(0, 1):
                spike_train[i] = 1

            # Calculate change in r, apply and clip.
            dr = s * self.dt
            r += dr
            r = min(self.r_max, max(r, self.r_min))

            # Calculate rate of change and clip.
            ds = np.random.uniform(self.ds_min, self.ds_max)
            s += ds
            s = min(self.s_max, max(self.s_min, s))

        return spike_train

    def generate_spike_trains(self):
        """
        Generates spike trains for all the afferents in the sample.
        :return:
        """

        # Container for spike trains.
        spike_trains = np.zeros((self.num_neurons, self.duration))

        for i in range(0, self.num_neurons):
            spike_train = self.generate_spike_train()
            spike_trains[i, :] = spike_train

            # Track progress
            progress = (i / float(self.num_neurons)) * 100
            sys.stdout.write("Generating spike trains: %d%% \r" % progress)
            sys.stdout.flush()

        self.spike_trains = spike_trains

    def read_patterns(self):
        """
        Loads a pattern from a file.
        :return: Pattern as numpy array
        """
        print 'reading patterns from file....'
        path = self.pattern_path
        # print 'len: ', len(self.pattern_filename)
        # for i in xrange(len(self.pattern_filename)):
        full_path = path + self.pattern_filename
        pattern = np.loadtxt(full_path)
        # pattern = pattern.transpose()
        #cho;pping it up
        pattern = pattern[:,self.start_time:self.end_time]
        self.patterns.append(pattern)
        #print len(self.patterns)


    def generate_pattern(self):
        """
        Generates a repeating pattern.
        :return: Pattern as a numpy array.
        """
        print 'Generating a pattern' 
        # Number of neurons involved in the pattern.
        num_neurons = self.num_neurons * self.inv_ratio

        # Identify a pattern of given length.
        start = np.random.randint(0, self.duration - self.pattern_duration)
        end = start + self.pattern_duration

        # Extract the pattern and save.
        # TODO: Ensure that it is not too similar to other patterns.
        pattern = deepcopy(self.spike_trains[:num_neurons, start: end])
        self.patterns.append(pattern)

        return pattern

    def generate_sample(self):
        """
        Generates the sample.
        :return:
        """

        # Generate background spike trains.
        self.generate_spike_trains()

    def add_noise(self):
        """
        Add noise to all spike trains.
        :return: None
        """
        for i in range(self.num_neurons):
            spike_train = deepcopy(self.spike_trains[i, :])

            # Get indices without spikes.
            indices = [j for j, dt in enumerate(spike_train) if dt == 0]

            # Add spikes to indices randomly with given probability.
            p = self.noise * self.dt
            for index in indices:
                if np.random.uniform(0, 1) < p:
                    spike_train[index] = 1

            self.spike_trains[i, :] = spike_train

    def insert_patterns(self):
        """
        Inserts patterns into the background noise.
        :return: None.
        """
        print 'inserting ', len(self.patterns), 'patterns'
        for pattern in self.patterns:
            # Get the start positions for the pattern to be inserted.
            starts = self.generate_start_positions()

            # Insert the pattern at start positions.
            num_neurons_in_pattern = self.num_neurons * self.inv_ratio
            for left in starts:
                right = left + self.pattern_duration
                # self.spike_trains[:num_neurons_in_pattern, left: right] = pattern
                self.spike_trains[:, left: right] = pattern

            # Save start positions for this pattern.   ???So this means that start position for each pattern will be adjacent in this array, How do we know which start positions belong to which patter?????
            self.start_positions.append(starts)

    def generate_patterns(self):
        """
        Generate patterns based on current spike trains.
        :param num_patterns: Number of patterns to generate.
        :return: Patterns generated.
        """


        # Duration of the spike pattern and buckets.
        self.num_buckets = math.floor(self.duration / self.pattern_duration)
        self.free_buckets = np.arange(self.num_buckets - 1)

        # TODO: Handle error more gracefully.
        if np.sum(self.spike_trains) == 0:
            print "WARNING! Generating empty pattern. " \
                  "Please generate spike trains first."


        for i in range(self.num_patterns):
            print "Generating pattern..."
            # Generate pattern from spike trains.

            if self.reading_pattern:
                self.read_patterns()

            else:
                self.generate_pattern()

        return self.patterns

    def load_pattern(self, pattern):
        """
        Loads a custom pattern to the generator.
        :param pattern: Pattern to load.
        :return: Array of patterns.
        """
        # TODO: Do some error checking.
        self.patterns.append(deepcopy(pattern))
        return self.patterns

    def generate_start_positions(self):
        """
        Gets the start position of a repeating pattern within the noise.
        :return: A list of the column indexes where the pattern starts.
        """
        effective_factor = 1.25
        num_buckets = len(self.free_buckets)
        positions = np.random.uniform(0, 1, num_buckets)
        start_positions = []
        count = 0
        effective_ratio = self.rep_ratio * effective_factor
        buckets_to_delete = []

        # Select buckets for this pattern.
        while count < num_buckets:

            # Mark as bucket and skip next bucket.
            if positions[count] < effective_ratio:
                pos = self.free_buckets[count] * self.pattern_duration
                buckets_to_delete.append(count)
                start_positions.append(pos)
                count += 2
            else:
                count += 1

        # Remove newly-occupied buckets.
        self.free_buckets = np.delete(self.free_buckets, buckets_to_delete)


        return start_positions

    """
    def generate_pattern(num_neurons, bg_len, pattern_len=50, seed=SEED):
        # Create the spike trains using a Poisson distribution.
        # Returns a dictionary with the spike trains and the time steps
        # where the pattern begins.
        #
        # :param num_neurons: Number of neurons.
        # :param bg_len: Length in ms of observation period.
        # :param pattern_len: Length in ms of repeating pattern.
        # :param seed: Seed to use for random generation.
        # :return: Pseudo-random-generated observation matrix.

        # Set seed.
        np.random.seed(seed)

        # Ensure that pattern is always shorter than total lengths.
        if pattern_len > bg_len:
            pattern_len = bg_len - 1

        # Create a num_neurons * bg_len matrix that contains
        # values uniformly distributed between 0 and 1.
        vt = np.random.uniform(0, 1, (num_neurons, bg_len))

        spikes = deepcopy(vt)

        # When probability is lower afferent does not spike.
        spikes[vt > F_PROB] = 0

        # When probability is lower afferent spikes.
        spikes[vt < F_PROB] = 1

        # Identify a pattern of given length.
        start = np.random.randint(0, bg_len - pattern_len)

        # Make only half of the neurons display the pattern.
        pattern = deepcopy(spikes[NUM_NEURONS / 2:, start: start + pattern_len])

        # Ensure that all afferents spike at least once in the pattern.
        for i in range(0, pattern.shape[0]):
            spike_sum = np.sum(pattern[i, :])
            if spike_sum < 1:
                rand_col = np.random.randint(0, pattern_len)
                pattern[i, rand_col] = 1

        # Calculate number of times that pattern will be repeated.
        reps = math.floor((bg_len * REPETITION_RATIO) / pattern_len)

        # Get the start positions for the pattern to be inserted.
        start_positions = get_start_positions(pattern_len, bg_len, reps)
        start_positions.sort()

        # Insert the pattern at start positions.
        for left in start_positions:
            right = left + pattern_len
            spikes[NUM_NEURONS / 2:, left: right] = pattern

        rvalue = dict()
        rvalue['spikes'] = spikes
        rvalue['start_positions'] = start_positions

        return rvalue

    def load_sample(filename):
        #TODO: erase this function
        f = open(filename)

        spike_trains = np.array([])
        start_positions = map(int, (f.readline()).split())
        lines = f.read().split('\n')
        num_neurons = len(lines)
        count = 0.0
        for line in lines:
            if line != [] and line != ['\n'] and line != '':
                if spike_trains.size == 0:
                    spike_trains = map(int, line.split())
                    spike_trains = np.reshape(spike_trains, (1, len(spike_trains)))
                else:
                    spike_trains = np.vstack((spike_trains, map(int, line.split())))

            progress = (count / num_neurons) * 100
            sys.stdout.write("Loading spike trains: %d%% \r" % progress)
            sys.stdout.flush()
            count += 1
        f.close()

        # Package everything nicely.
        rvalue = dict()
        rvalue['spike_trains'] = spike_trains
        rvalue['start_positions'] = start_positions

        return rvalue

    def save_sample(filename, sample):
        #TODO: Erase this function
        #Old version
        start_positions = sample['start_positions']
        spike_trains = sample['spike_trains']
        f = open(filename, 'w')
        for start in range(len(start_positions)):
            f.write('%d ' % start_positions[start])
        f.write('\n')

        num_neurons = spike_trains.shape[0]
        for row in range(num_neurons):
            for col in range(spike_trains.shape[1]):
                f.write('%0.1d ' % spike_trains[row, col])
            f.write('\n')
            progress = (row / float(num_neurons)) * 100
            sys.stdout.write("Saving spike trains: %d%% \r" % progress)
            sys.stdout.flush()
        f.write('\n\n')
        f.close()
    """

    def save(self):


        if self.filename is None:
            
            filename = path + "/{}_{}_{}_{}_{}_{}_{}".format(self.num_patterns,
                                                             self.num_neurons,
                                                             self.duration,
                                                             self.pattern_duration,
                                                             self.rep_ratio,
                                                             self.inv_ratio,
                                                             self.noise)
            print filename
        else:
            filename = os.path.join(self.sample_directory,self.filename)

        np.savez(filename,
                 start_positions=self.start_positions,
                 spike_trains=self.spike_trains,
                 pattern_duration=self.pattern_duration
                )

'''
if __name__ == '__main__':
    bg = np.load("samples/5_neur_100_elecs_15000_runtime.npz")
    st = bg['spike_trains']
    ps = [st[:, 150:200]]
    sg = Sample(15000, ps, num_neurons=500, rep_ratio=0.25, inv_ratio=1,
                filename="trial_inv_100pc")
    sg.spike_trains = st
    sg.insert_patterns()
    sg.add_noise()
    sg.save()
    # sample = generate_sample(NUM_NEURONS, TOTAL_MS, PATTERN_MS)
    # spike_trains = sample['spike_trains']
    # mpl.imshow(spike_trains[0:2000, 0:2000],
    #            interpolation='nearest',
    #            cmap=mpl.cm.Greys)
    # mpl.title('Spike Trains')
    # mpl.ylabel('# Afferent')
    # mpl.xlabel('Time (ms)')
    # mpl.show()
'''
