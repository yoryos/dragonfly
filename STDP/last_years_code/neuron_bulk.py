__author__ = 'Juan Carlos Farah,' \
             'Panagiotis Almpouras,' \
             'Erik Grabljevec'
__authoremail__ = 'juancarlos.farah14@imperial.ac.uk,' \
                  'panagiotis.almpouras12@imperial.ac.uk,' \
                  'erik.grabljevec14@imperial.ac.uk'

import math
from copy import deepcopy

import numpy as np
import pylab
import matplotlib.pyplot as plt


# Set Seed
# np.random.seed(1)


class Neuron:
    """
    Simulates a leaky integrate-and-fire (LIF) neuron following the
    Gerstner's Spike Response Model (SRM). As a basis we are using
    the parameters from Masquelier et al. (2008).
    """

    # Defaults
    THETA = 50
    A_PLUS = 0.03125
    # Q1: was 0.92, should A_Ratio be 0.85 maybe?
    A_RATIO = 0.85

    def __init__(self,
                 num_afferents,
                 a_plus=A_PLUS,
                 a_ratio=A_RATIO,
                 theta=THETA,
                 weights=None):
        self.dt = 1                          # Discrete time step in ms.
        self.num_afferents = num_afferents   # Number of afferents.
        self.tau_m = 10.0                    # Membrane time constant in ms.
        self.tau_s = 2.5                     # Synapse time constant in ms.
        self.t_plus = 16.8                   # LTP modification constant in ms.
        self.t_minus = 27.7                  # LTD modification constant in ms.This was initially 33.7
        self.t_window = int(self.tau_m * 7)  # Max time that spike affects EPSP.
        self.weight_max = 1                  # Maximum weight value.
        self.weight_min = 0                  # Minimum weight value.
        self.theta = theta                   # Threshold in arbitrary units.
        self.alpha = 0.25                    # Multiplicative constant for IPSP.
        self.k = 2.1222619                   # Multiplicative constant for EPSP.
        self.k1 = 2                          # Constant for positive pulse.
        self.k2 = 4                          # Constant for after-potential.

        # LTP and LTD learning rates. NOTE: a_minus_ratio should be 0.85.
        self.a_plus = a_plus
        self.a_minus = a_ratio * self.a_plus
        self.ltp_window = int(7 * self.t_plus)    # LTP learning window.
        self.ltd_window = int(7 * self.t_minus)   # LTD learning window.

        # Spike information.
        self.time_delta = None               # Time since last spike in ms.
        self.spike_times = []                # Container to save spike times.
        self.potential = []                  # Tracks membrane potential.
        self.ipsps = np.array([])            # Stores deltas for IPSP.

        #adding spike history for afferents (list of lists of spikes for each afferent)

        self.aff_spike_history = np.zeros((self.num_afferents,1))

        self.siblings = []

        # Save historic weights in this container. Note that all weights will
        # be saved only if the simulation is running with the save_weights flag.
        self.historic_weights = np.array([])

        #Q3: Weight distributions will be saved.
        self.weight_distributions = []

        # Effective width of the LTP window given spike pattern.
        self.effective_ltp_window = self.ltp_window

        # Initialise weights.
        np.random.seed(1)
        if weights is None:
            self.current_weights = np.random.normal(0.475, 0.1, (num_afferents, 1))
            self.current_weights[self.current_weights < self.weight_min] = self.weight_min
            self.current_weights[self.current_weights > self.weight_max] = self.weight_max
        else:
            self.current_weights = weights


        # Initialise epsilons.
        self.epsilons = self.calculate_epsilons()

        #Q4: essentially, initialize all synapses?
        # Initialise synapses eligible to undergo LTD.
        self.synapses_for_ltd = np.ones((self.num_afferents, 1), dtype=np.int32)

        # Initialise EPSP inputs container. (one for each afferent)
        self.epsps = np.zeros((self.num_afferents, 1))

    #this creates a vector of epsilon values from time of spike until t_window threshhold in steps of 1ms
    def calculate_epsilons(self):
        """
        Returns a vector with epsilon values relevant for EPSP.
        :return: Vector of epsilon values in time window.
        """
        epsilons = np.ndarray((self.t_window, 1), dtype=float)
        for i in range(0, self.t_window):
            delta = self.t_window - (i + 1)  #getting the diff between time and time of presynaptic spike t - t_spike
            epsilons[i, 0] = self.calculate_epsilon(delta)

        return epsilons

    #this calculates epsilon for one specific time after presynaptic spike time
    def calculate_epsilon(self, delta):
        """
        Returns the value of the epsilon kernel given a time delta.
        :param delta: Time since last spike.
        :return:
        """
        hss = self.calculate_heavyside_step(delta)
        left_exp = math.exp(-delta / self.tau_m)
        right_exp = math.exp(-delta / self.tau_s)
        epsilon = self.k * (left_exp - right_exp) * hss
        return epsilon


    def update_weights(self, spike_trains, ms):
        """
        Updates the weights according to STDP.
        :return: Updated weight vector.
        """
        ar = spike_trains[:,ms]
        print 'sum of spike_train: ', np.sum(ar)
        # Avoid updating weights without time delta.
        if self.time_delta is None:
            return self.current_weights

        # Get number of neurons.
        num_afferents = spike_trains.shape[0]
        ltp_window_start = ms - self.effective_ltp_window
        print 'ltp_window_start: ', ltp_window_start, 'ms : ', ms, 'width: ', ms - ltp_window_start


        #OLD version CHANGE TO THIS WHEN USING SIMULATION INSTEAD OF STDP
        spikes = deepcopy(spike_trains[:, ltp_window_start: ms + 1])

        stop1 = raw_input('press any key to see the spikes sum of each afferent')
        print np.sum(spikes,axis=0)
        stop2 = raw_input('press any key to see the length of memory')
        print spikes.shape


        # If post-synaptic neuron has just fired, calculate time delta and
        # LTP for each afferent, then adjust all weights within the time window.
        if self.time_delta == 0:
            for i in range(0, self.num_afferents):
                spike_lag = self.calculate_time_delta(spikes[i, :])
                weight_delta = self.calculate_ltp(spike_lag)

                # Add weight delta and clip so that it's not > maximum.
                self.current_weights[i] = min(self.weight_max,
                                              self.current_weights[i] + weight_delta)

            # Q6: Reset synapses eligible for LTD.????????
            self.synapses_for_ltd = np.ones((self.num_afferents, 1),
                                            dtype=np.float)

        # Otherwise calculate LTD for all neurons that have fired,
        # if post-synaptic neuron has fired within the time window.
        elif 0 < self.time_delta < self.ltd_window:

            # Only consider last ms in spike trains.
            spikes = np.reshape(deepcopy(spikes[:, -1]), (num_afferents, 1))

            # Get LTD change for pre-synaptic neurons that
            # just spiked and have not been weighed yet.
            neurons_to_weigh = np.multiply(spikes, self.synapses_for_ltd)
            weight_delta = self.calculate_ltd(self.time_delta) * neurons_to_weigh

            # Add weight delta and clip so that they are not < minimum.
            self.current_weights += weight_delta
            self.current_weights[self.current_weights < self.weight_min] = self.weight_min
            self.synapses_for_ltd = np.multiply(self.synapses_for_ltd,
                                                np.logical_not(spikes))

        return self.current_weights


    def update_ltp_window_width(self, ms):

        # Number of spikes.
        num_spikes = len(self.spike_times)

        # Get effective width of LTP window.
        if num_spikes == 0:
            self.effective_ltp_window = min(ms, self.ltp_window)
        elif num_spikes == 1:
            if ms == self.spike_times[0]:
                self.effective_ltp_window = min(ms, self.ltp_window)
            elif ms > self.spike_times[0]:
                self.effective_ltp_window = min(ms - self.spike_times[0] + 1, self.ltp_window)
        else:
            if ms == self.spike_times[-1]:  #if ms corresponds to the last spike
                #This sets the effective ltp window to the difference between the last two spike times, but why?
                self.effective_ltp_window = min(self.spike_times[-1] - self.spike_times[-2], self.ltp_window)
                #if ms is greate than the last spike time....
            elif ms > self.spike_times[num_spikes - 1]:
                self.effective_ltp_window = min(ms - self.spike_times[-1], self.ltp_window)
        # print 'updated ltp_window_width at ms: ', ms, 'is: ', self.effective_ltp_window

    def calculate_time_delta(self, spike_train):
        """
        Given an afferent, return its last spike time relative to now.
        :param spike_train: Array of spike values for afferent.
        :return: Time delta of spike time.
        """
        start = 0
        end = len(spike_train) - 1

        # Find the nearest spike in the spike train.
        for ms in range(end, start - 1, -self.dt):
            if spike_train[ms] == 1:
		#note that this order might be to obtain a negative time delta
                return ms - end


        # If there are no spikes return a value outside the learning
        # window which means this neuron will be ignored for STDP.
        return (self.ltp_window + 1) * -1

    def calculate_ltp(self, time_delta):
        """
        Calculate weight change according to LTP.
        Note that the input delta to LTP has to be <= 0.
        :param time_delta: Delay between post-synaptic and afferent spike.
        :return: Change in weight.
        """

        # Input delta to LTP has to be <= 0.
        #so, time delta is calculates as time of presynaptic spike - time of postsynaptic spike
        #if this delta is positive, then the spiking occurred in the wrong order, so this synapse should not
        #be considered for LTP
        if time_delta > 0:
            print "ERROR! Input to LTP function needs to be less than " \
                  "or equal to zero. Please double check your function calls."
            # Use raise value error.
            exit(1)

        # Only consider deltas within the learning window.
        if time_delta is None or math.fabs(time_delta) > self.ltp_window:
            return 0

        return self.a_plus * math.exp(time_delta / self.t_plus)

    def calculate_ltd(self, time_delta):
        """
        Calculate weight change according to LTD.
        Note that the input delta to LTP has to be > 0.
        :param time_delta: Delay between post-synaptic and afferent spike.
        :return: Change in weight.
        """

        # Input delta to LTP has to be > 0.
        if time_delta <= 0:
            print "ERROR! Input to LTD function needs to be greater " \
                  "than zero. Please double check your function calls."
            # Use raise value error.
            exit(1)

        # Only consider deltas within the learning window.
        if time_delta is None or math.fabs(time_delta) > self.ltd_window:
            return 0

        return -self.a_minus * math.exp(-time_delta / self.t_minus)

    def calculate_heavyside_step(self, delta):
        """
        Calculate value of heavyside step for a given time delta.
        :param delta: Time difference.
        :return:
        """
        if delta >= 0:
            return 1
        else:
            return 0

    def update_epsps(self, spikes):
        """
        Given vectors of spikes and weights update the EPSP contributions.
        :param spikes: Vector of current spikes.
        :return:
        """
        #IS THIS WHERE THE TOTAL MEMBRANE POTENTIAL IS CALCULATED?

        # Flush EPSP inputs if neuron just spiked.
        if self.time_delta == 0:
            self.epsps = np.zeros((self.num_afferents, 1))
        else:
            weighted = np.multiply(spikes, self.current_weights)
	    #WHAT exactly does the hstack here do?
            self.epsps = np.hstack((self.epsps, weighted))

            # Length of new EPSP window.
            width = self.epsps.shape[1]

            # Q8: Clamp neuron to learning window.??????????
            if width > self.t_window:
                window_start = width - self.t_window
                self.epsps = self.epsps[:, window_start:]

        return self.epsps

    def sum_epsps(self):
        """
        Sum the EPSP contribution of all afferents.
        :return:
        """

        # Only consider as many epsilons as we have input.?????????????
        input_width = self.epsps.shape[1]
        # print 'input width is: ', input_width
        epsilon_height = self.epsilons.shape[0]
        start = epsilon_height - input_width

        # Get weighted EPSP contributions.
        weighted = np.matrix(self.epsps) * np.matrix(self.epsilons[start:])

        # Return sum of weighted contributions.
        return np.sum(weighted)

    def update_ipsps(self, ms):
        """
        Add the current time delta to the input values for IPSP calculation.
        :param ms: Current time in ms.
        :return: None.
        """
        # TODO: Figure out clamping for efficiency.
        self.ipsps = np.append(self.ipsps, ms)

    def sum_ipsps(self, ms):
        """
        Sum the IPSP contribution of all connected neurons.
        :return: Sum of IPSP values.
        """
        # Don't calculate if IPSP is empty.
        if self.ipsps.size == 0:
            return 0

        # Return sum of weighted contributions.
        ms = np.repeat(ms, len(self.ipsps))
        time_deltas = ms - self.ipsps
        f = np.vectorize(self.calculate_mu)
        ipsp_values = f(time_deltas)
        return np.sum(ipsp_values)

    def calculate_psp(self, time_delta=0, debugging=False):
        """
        Calculate the effect of a post-synaptic spike on the potential.
        :param time_delta: Time since last spike of post-synaptic neuron.
        :return: Value of the effect.
        """
        # If time delta is not initialised or irrelevant, return 0.
        if debugging:
            time_delta = time_delta
        else:
            if self.time_delta is None or self.time_delta > self.t_window:
                return 0
            else:
                time_delta = self.time_delta

        # Otherwise perform normal calculations.
        hss = self.calculate_heavyside_step(time_delta)
        left_exp = math.exp(-time_delta / self.tau_m)
        right_exp = math.exp(-time_delta / self.tau_s)
        v = self.k1 * left_exp - self.k2 * (left_exp - right_exp)
        return self.theta * v * hss

    def calculate_membrane_potential(self, ms):
        """
        Return the current membrane potential.
        :return: Membrane potential.
        """
        psp = self.calculate_psp()
        # print 'psp is: ', psp
        epsp_sum = self.sum_epsps()
        # print 'epsp_sum is: ', epsp_sum
        ipsp_sum = self.sum_ipsps(ms)
        # print 'ipsp_sum is: ', ipsp_sum

        membrane_pot = psp + epsp_sum + ipsp_sum
        #print 'membrane potential is:', membrane_pot
        return psp + epsp_sum + ipsp_sum

    def calculate_mu(self, delta):
        """
        Calculate the value of the mu kernel.
        :return: Value of mu.
        """
        if delta is None or delta > self.t_window:
            return 0
        epsilon = self.calculate_epsilon(delta)
        mu = -self.alpha * self.theta * epsilon
        return mu

    def plot_eta(self, t_min=0):
        """
        Plots the values of the eta kernel for a given range.
        :return: Void.
        """
        # Calculate PSPs for given range.
        window = self.t_window
        lead = 10
        p = [0] * lead
        time = [i for i in range(-lead, 0)] + [j for j in range(t_min, window)]
        for ms in range(t_min, window):
            psp = self.calculate_psp(ms, True)
            p.append(psp)

        # Plot values.
        pylab.plot(time, p)
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Post-Synaptic Potential')
        pylab.title('Eta Kernel')
        pylab.show()

    def plot_epsilon(self, t_min=0):
        """
        Plots the values of the epsilon kernel.
        :return: Void.
        """
        epsilons = self.calculate_epsilons()
        pylab.plot(range(self.t_window - 1, t_min - 1, -1), epsilons)
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Epsilon')
        pylab.title('EPSP Epsilon Kernel')
        pylab.show()

    def plot_ltp(self, show=True, return_fig=False):
        """
        Plots the values of LTP over the learning window.
        :return: Void.
        """
        # Set plot parameters.
        start = 0
        end = int(self.ltp_window) * -1

        # Containers for y and x values, respectively.
        ltps = []
        time_deltas = np.arange(end, start, 1, dtype=np.int32)

        # Get value for LTP.
        for ms in time_deltas:
            # Input to LTP has to be negative.
            ltp = self.calculate_ltp(ms)
            ltps.append(ltp)

        # Plot values.
        fig = pylab.plot(time_deltas, ltps)
        if show:
            pylab.xlabel('Time Delta (ms)')
            pylab.ylabel('Weight Change')
            pylab.title('Weight Change from LTP')
            pylab.show()

        if return_fig:
            return fig

    def plot_ltd(self, show=True, return_fig=False):
        """
        Plots the values of LTD over the learning window.
        :return: Void.
        """
        # Set plot parameters.
        start = 1
        end = int(self.ltd_window)

        # Containers for y and x values, respectively.
        ltds = []
        time_deltas = np.arange(start, end, 1, dtype=np.int32)

        # Get value for LTD.
        for ms in range(start, end, 1):
            ltd = self.calculate_ltd(ms)
            ltds.append(ltd)

        # Plot values.
        fig = pylab.plot(time_deltas[start:end], ltds[start:end])

        if show:
            pylab.xlabel('Time Delta (ms)')
            pylab.ylabel('Weight Change')
            pylab.title('Weight Change from LTD')
            pylab.show()

        if return_fig:
            return fig

    """
    def plot_weight_distribution(self, ms, rows=1, cols=1,
                                 current_frame=1, bin_size=1,
                                 show = True, return_fig = False):


        Plots the distribution of the values of the weights.
        :param rows: Number of rows in the plot.
        :param cols: Number of columns in the plot.
        :param current_frame: Current frame for the plot.
        :param bin_size: Size of bins when generating the histogram.
        :return: Void.

        # Plot a histogram of the values.
        p = pylab.subplot(rows, cols, current_frame)

        # Only add title to first plot.
        if current_frame == 1:
            pylab.title('Weight Distribution Over Time')

        p.axes.get_xaxis().set_visible(False)
        p.axes.get_yaxis().set_ticks([])
        label = str(ms / 1000) + 's'
        bins = int(len(self.current_weights) / bin_size)
        print 'BINS ARE:', bins
        pylab.ylabel(label)
        fig = p.hist(self.current_weights, bins=bins)

        # Only show if plot is complete.
        if rows * cols == current_frame:
            pylab.xlabel("Weight Value")
            p.axes.get_xaxis().set_visible(True)
            if show:
                pylab.show()

        if return_fig:
            return fig

    def save_weight_distributions(self):
        Saves current weight distribution.
        :return: None.
        dist = np.histogram(self.current_weights, range=(0, 1))[0]
        self.weight_distributions.append(dist.tolist())
    """

    def connect(self, neuron):
        """
        Connects two neurons.
        :param neuron: Neuron to connect to.
        :return: None.
        """
        self.siblings.append(neuron)
        neuron.siblings.append(self)

    def plot_stdp(self, show=True):
        """
        Plot STDP from both LTD and LTP.
        :return: Void.
        """
        self.plot_ltd(False)
        self.plot_ltp(False)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        pylab.xlabel('Time Delta (ms)')
        pylab.ylabel('Weight Change from STDP')
        pylab.xlim(-1 * self.ltp_window, self.ltd_window)
        pylab.title('Effect of STDP on Synaptic Weights')
        if show:
            pylab.show()

if __name__ == "__main__" and __package__ is None:
    neuron = Neuron(500)
    neuron.plot_stdp()
