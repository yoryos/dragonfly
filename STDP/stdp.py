from Helper.Configurer import Configurer
import pickle
import math
import time
import datetime
import numpy as np
import pylab
from matplotlib.patches import Rectangle
import os

import neuron
from Helper.BrainModule import BrainModule






class Stdp(BrainModule):
    """
    Class for a stdp set up with n neurons and num_afferents afferents
    If training is set to true, the weights of the synapses will be changed according to stdp
    """

    def __init__(self, run_id = None,
                 num_afferents = 1000, num_neurons=4, weights = None,
                 load_weights=False, weights_list = None, weights_path = None,
                 training=False, output_type=bool,
                 verbose = True,
                 historic_weights_path=None, saving_historic_weights=False,
                 alpha=0.25, theta=200, a_plus=0.03125, a_ratio=0.95,
                 tau_m=10.0, tau_s=2.5, t_plus=17.8, t_minus=26.7, k=2.1222619, k1=2, k2=4
                 ):

        BrainModule.__init__(self, run_id)

        self.verbose = verbose
        self.count = 0  # internal clock
        self.t_min = 0  # Start time in ms.
        self.t_step = 1  # Time step in ms.


        self.num_afferents = num_afferents
        self.num_neurons = num_neurons
        self.neurons = []
        self.training = training

        self.saving_historic_weights = saving_historic_weights
        self.historic_output = np.array([], dtype=output_type)


        # this allows to load different weights for different neurons
        if load_weights == True:
            if weights_list != None:
                if self.verbose:
                    print 'loading weights from weights list for each neurons'
                for weights_path in weights_list:
                    weights = np.loadtxt(weights_path, ndmin = 2)
                    self.add_neuron(num_afferents, alpha, theta, a_plus, a_ratio,
                                    tau_m, tau_s, t_plus, t_minus, k, k1, k2, weights)
            else:
                if weights_path == None:
                    print 'aborting, there is no path given to load weights'
                    raise IOError
                weights = np.loadtxt(weights_path, ndmin = 2)
                if self.verbose:
                    print 'attempting to load ', len(weights), ' weights \n from path \n', weights_path, '\n for ', self.num_neurons, ' neurons.'
                self.set_up(num_afferents, alpha, theta, a_plus, a_ratio,
                            tau_m, tau_s, t_plus, t_minus, k, k1, k2, weights)

        else:
            if self.verbose:
                print 'using random weights!'
            self.set_up(num_afferents, alpha, theta, a_plus, a_ratio,
                        tau_m, tau_s, t_plus, t_minus, k, k1, k2, None)


        if saving_historic_weights:
            self.historic_weights_path = historic_weights_path  # path to save weights
            self.historic_weights_filename = "weights_of"  # filename to save weights

        self.parameters = {"theta": theta,
                            "alpha": alpha,
                            "a_plus": a_plus,
                            "a_ratio": a_ratio,
                            "t_plus": t_plus,
                            "t_minus": t_minus
                            }

    # setting up the neuron net
    def set_up(self, num_afferents, alpha, theta, a_plus, a_ratio,
               tau_m, tau_s, t_plus, t_minus, k, k1, k2, weights):

        """
        :param: neuron relevant constants
        :return: neuron.
        """

        # set container to save output
        self.historic_output = np.resize(self.historic_output, (1, self.num_neurons))
        # Reset neurons.
        for i in range(len(self.neurons)):
            # Clear spike times container.
            self.neurons[i].spike_times = []
        self.neurons = []
        # add neurons
        for i in xrange(self.num_neurons):
            # using: a_plus, a_ratio and theta with below values:
            if weights != None:
                if self.verbose:
                    print 'using saved weights......'
                self.add_neuron(num_afferents, alpha, theta, a_plus, a_ratio,
                                tau_m, tau_s, t_plus, t_minus, k, k1, k2, weights)
            else:
                self.add_neuron(num_afferents, alpha, theta, a_plus, a_ratio,
                                tau_m, tau_s, t_plus, t_minus, k, k1, k2, weights)
        # connect the neurons
        for i in xrange(self.num_neurons):
            for j in xrange(i, self.num_neurons - 1):
                self.neurons[i].connect(self.neurons[j])


    # Adds neuron to simulation.
    def add_neuron(self, num_afferents, alpha, theta, a_plus, a_ratio,
                   tau_m, tau_s, t_plus, t_minus, k, k1, k2, weights):
        """
        :param: neuron relevant constants
        :return: neuron.
        """
        n = neuron.Neuron(self.num_afferents,
                          alpha, theta, a_plus, a_ratio,
                          tau_m, tau_s, t_plus, t_minus, k, k1, k2, weights)
        self.neurons.append(n)
        return n

    # produces a spike train from indices
    def produce_spike_array(self, spike_indices):
        """
        :param: indices of incomng spikes
        :return: spike train.
        """
        # initialize spikes
        spikes = np.zeros((self.num_afferents))

        # create spike train using spike indices
        for i in xrange(len(spikes)):
            if i in spike_indices:
                spikes[i] = 1

        return np.array(spikes)

    # step function
    #function which performs a step in the stdp procedure: the spike indices of a spike train of one dt is fed into the stdp neurons.
    def step(self, spike_indices, output = bool):
        # create a spike train
        spikes = self.produce_spike_array(spike_indices)

        # create container for spike output for each stdp neuron
        output_spikes = []

        # get the current timestep using the length of saved membrane potentials of one of the neurons
        ms = self.count

        # do the same thing for all stdp neurons
        for n in self.neurons:

            # Shape spikes into numpy column array
            spikes = np.reshape(spikes, (self.num_afferents, 1))

            # Update time delta. Getting the time difference to the last spike
            if len(n.spike_times) > 0:
                n.time_delta = ms - n.spike_times[-1]

            # Update EPSP inputs.
            n.update_epsps(spikes)

            # Send inhibitory signal to sibling neurons.
            if n.time_delta == 0:
                n.ipsps = np.array([])
                for s in n.siblings:
                    s.update_ipsps(ms)

            # Calculate membrane potential.
            p = n.calculate_membrane_potential(ms)

            # Post the potential to the next ms.
            n.potential = np.resize(n.potential, (len(n.potential) + 1))
            n.potential[ms + 1] = p

            # Update LTP window width.
            n.update_ltp_window_width(ms)

            # Record weights at this point only if running with flag
            if self.saving_historic_weights and self.training:
                if n.historic_weights.size == 0:
                    n.historic_weights = self.neurons[0].current_weights

                else:
                    n.historic_weights = np.hstack((n.historic_weights,
                                                    n.current_weights))

            # shape spikes back for update_weights
            spikes = np.reshape(spikes, (self.num_afferents))

            # Update weights.
            if self.training:
                n.update_weights(spikes, ms)

            spike_monitor = False
            if output == bool:
                output_spike = False
            else:
                output_spike = 0.0
            # If threshold has been met and more than 1 ms has elapsed
            # since the last post-synaptic spike, schedule a spike.
            if p >= n.theta and (n.time_delta > 1 or n.time_delta is None):
                spike_monitor = True
                if output == bool:
                    output_spike = True
                else:
                    output_spike = p

            if spike_monitor == True:
                n.spike_times.append(ms + 1)

            # append spike info of neuron to output_spikes
            output_spikes.append(output_spike)

        # increase timestep count of stdp class-this remembers what ms was for the next input
        self.count += 1
        # save output_spike
        self.historic_output = np.insert(self.historic_output, self.historic_output.shape[0], output_spikes, axis=0)
        return np.array(output_spikes)

    # function to save historic weights to file
    def save_historic_weights(self, path):
        # save historic weights to files
        for i in xrange(len(self.neurons)):
            full_path = os.path.join(path, ('historic_weights_neuron' + np.str(i) + ".dat"))
            np.savetxt(full_path, self.neurons[i].historic_weights)

    # function to save the current weights to file
    def save_current_weights(self,path):
        weights = []
        for i in xrange(len(self.neurons)):
            weights.append(self.neurons[i].current_weights)
        full_path = os.path.join(path, ("current_stpd_weights.dat"))
        np.savetxt(full_path, weights)
        if self.verbose:
            print 'weights saved at: ', full_path

    def save_output_spikes(self, directory=None, name="stdp_spikes.dat", run_id_prefix=False):
        self.save_numpy_array(self.historic_output, directory, name, fmt="%i")

    # functions to run a simulation with a test pattern
    def test_with_pattern(self, filename, folder):

        # add filename and folder to parameter dict
        self.parameters["sample_filename"] = filename
        self.parameters["sample_folder"] = folder
        # load sample file to test the step function
        self.load_file(filename, folder)
        self.sample_loc = folder + filename

    def load_file(self, filename, folder, extension=".npz"):
        """
        Loads a file containing sample spike trains.
        :param filename: Name of file with spike trains.
        :param folder: Folder containing sample files.
        :param extension: Filename extension.
        :return: None.
        """
        path = folder + filename + extension
        sample = np.load(path)
        ###IS THIS REDUNDANT?####!!!!!!!!!!
        self.spike_trains = sample['spike_trains']

        if 'start_positions' in sample:
            self.start_positions = sample['start_positions']
        else:
            self.start_positions = []

        if 'pattern_duration' in sample:
            self.pattern_duration = sample['pattern_duration']
        else:
            if self.verbose:
                print 'using default pattern duration of 50 ms'
            self.pattern_duration = 50

        # self.num_afferents = self.spike_trains.shape[0]

        self.duration = self.spike_trains.shape[1]

        self.sampling_interval = math.ceil(self.duration / 5)

    # def load(self, sample):
    #     """
    #     Loads a sample.
    #     :param sample: Sample
    #     :return: None.
    #     """
    #     self.spike_trains = sample.spike_trains
    #     self.start_positions = sample.start_positions
    #     self.pattern_duration = sample.pattern_duration
    #     self.duration = self.spike_trains.shape[1]
    #     self.sampling_interval = math.ceil(self.duration / 5)

    #function to plot the saved membrane potential
    def plot_membrane_potential(self, verbose, parameter_config_num = None , path = None, _duration = None):

        start = self.t_min
        if _duration == None:
            end = self.duration
        else:
            end = _duration

        # Container for time.
        time = np.arange(start, end, 1, dtype=np.int32)

        colors = ["#E6E6E6", "#CCFFCC", "#FFCC99", "#CCFFFF", "#FFFFCC"]

        # Boundaries.
        min_y = self.neurons[0].theta * -0.5
        max_y = self.neurons[0].theta * 2.25

        # Prepare the pattern plot.
        if self.start_positions != None:
            for i in range(len(self.start_positions)):
                color = colors[i % len(colors)]
                for j in self.start_positions[i]:
                    pylab.gca().add_patch(Rectangle((j, min_y),
                                                    self.pattern_duration,
                                                    max_y + math.fabs(min_y),
                                                    facecolor=color,
                                                    edgecolor=color))

        # Plot membrane potential for each neuron.
        for n in self.neurons:

            pylab.plot(time[start:end], n.potential[start:end])
            pylab.ylim(min_y, max_y)
            pylab.tick_params(axis='x', labelsize=27)
            pylab.tick_params(axis='y', labelsize=27)

        # Prepare and display plot.
        pylab.xlabel('Time (ms)', fontsize = 30)
        pylab.ylabel('Membrane Potential (V)', fontsize = 30)
        pylab.title('STDP distinguishing between two distinct patterns (linear and oscillating)', fontsize = 30)
        if verbose:
            pylab.show()
            pylab.gcf().clear()
        else:
            pylab.savefig(path+'figure_of_' + str(parameter_config_num) + 'th_parameter_combination')
            pylab.gcf().clear()

    # metric function to calculate the effectiveness of STDP.
    def calculate_metric(self, time_end, time_start=0, pybo_mode = False):
        """
        param: time range over which to calculate the metric, start and end
        return: metric calculation array (one metric for each neuron)
        """

        # can only be run when there is a pattern sample, as the start_positions
        # and the pattern_duration are required
        if self.start_positions == None or self.pattern_duration == None:
            print 'Cannot compute metric without known pattern_start positions'
            return 0

        if len(self.neurons) != len(self.start_positions):
            print 'number of neurons and number of patterns in sample not the same'
            return 0

        beta = 1
        # container for metric
        metric = np.zeros(len(self.neurons))
        # increase positives when spike happens within pattern
        for j in xrange(len(self.neurons)):
            # adjust amount of patterns there are according to time slice
            start_positions = []
            for i in self.start_positions[j]:
                if i > time_start and i < time_end:
                    start_positions.append(i)
            # adjust amount of spikes there are according to time slice
            spike_times = []
            for i in self.neurons[j].spike_times:
                if i > time_start and i < time_end:
                    spike_times.append(i)
            m_positives = 0
            for spike_time in spike_times:

                #check if spike is within any of the patterns
                for el in start_positions:
                    if el <= spike_time <= (el+self.pattern_duration):
                        m_positives += 1

            m_fnegatives = len(start_positions) - m_positives

            m_fpositives = len(spike_times) - m_positives

            #get the ratio of (positives * (1 + beta2))/total spike_times
            denominator = (beta**2 * m_fnegatives + m_fpositives + (m_positives * (1 + beta**2)))
            if denominator != 0:
                metric[j] = (m_positives * (1 + beta**2))/denominator
            else:
                metric[j]=0.0


        if pybo_mode:
            return metric[0]
        else:
            return metric

    def save_parameters(self, path):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        filename = 'stdp_run_' + 'run_id_' + str(self.run_id) + '_' + st
        full_path = path + filename
        pickle.dump(self.parameters, open(full_path,"wb"))
        if self.verbose:
            print 'saved parameters pickled at: ', full_path
