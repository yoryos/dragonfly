"""
Cstmd1

Description

__author__: Dragonfly Project 2016 - Imperial College London
            ({anc15, cps15, dk2015, gk513, lm1015,zl4215}@imperial.ac.uk)

"""

import cPickle as pickle
import os
import pprint

import matplotlib.pyplot as plt
import numpy as np
from cstmd1Cython import Cstmd1Sim
from scipy import signal

from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
from Helper.BrainModule import BrainModule


class Cstmd1(BrainModule):
    v = None
    m = None
    n = None
    h = None
    dirty = False
    T = 0

    def __init__(self, global_dt, speed_up, estmd_dim, buffer_size, run_id, morph_param, sim_parameters,
                 cuda_device=0, debug_mode=0, enable_spike_dump=False, preload_path = None, save_bare_nc_path = None):
        """

        Args:
            global_dt (float): time step for simulator (ms)
            speed_up (int): number of times to run cuda step per python step
            estmd_dim (Tuple(int,int)): tuple of estmd pixel array dimensions, (width, height)
            buffer_size (int): max time, used to reserve GPU memoru
            run_id (str): id of trial run, used as prefix when saving files
            morphology_parameters (dict): morphology parameters used to construct model of CSTMD1
            sim_parameters (dict): simulation parameters used by Cuda simulator
            debug_mode (int): 1-6, sets debugging output level
        """
        BrainModule.__init__(self, run_id)

        print "=====Attempting to build CSTMD1 Module====="
        print "Morphology parameters: "
        pprint.pprint(morph_param)
        print
        print "Cuda simulation parameters: "
        pprint.pprint(sim_parameters)
        print

        self.global_dt = global_dt
        self.dt = float(global_dt) / speed_up
        print "GPU time-step = " + str(self.dt) + "ms"
        self.verbose_debug = debug_mode == 3
        self.spike_history = []
        self.spike_rate_data = None
        self.save_bare_nc_path = save_bare_nc_path

        self.enable_spike_dump = enable_spike_dump
        print "Mass spike dump enabled?", self.enable_spike_dump
        self.number_of_neurons = morph_param["number_of_neurons"]
        self.enable_synapses = morph_param["synapses"]
        self.morphology_path = morph_param["morphology_path"]
        self.moph_parameters = morph_param
        self.sim_parameters = sim_parameters

        # Build the neurons
        self.__build_neuron_model(preload_path = preload_path,
                                  number_of_neurons=self.number_of_neurons,
                                  morphology_path=self.morphology_path,
                                  morphology_prefix=morph_param["morphology_prefix"],
                                  homogenise=morph_param['homogenise'])

        # Get synapses
        if self.enable_synapses:
            self.__build_synapses(synapses_f_name=morph_param['synapses_file_name'],
                                  n=morph_param['number_of_synapses'],
                                  min_distance=morph_param['minimum_synapse_distance'])

        # Get electrodes
        self.__build_electrodes(electrodes_f_name=morph_param["electrodes_f_name"],
                                soma=morph_param['soma_electrodes'],
                                n=morph_param['number_of_electrodes'],
                                random=morph_param['random_electrodes'])

        # Compute estmd mapping
        self.__build_estmd_mapping(estmd_dim=estmd_dim,
                                   estmd_map_f_name=morph_param['estmd_map_f_name'],
                                   topological=morph_param['topological'])

        if self.verbose_debug:
            if self.enable_synapses:
                print "Synapses", self.synapses
            print "Morphology", self.morphology
            print "Electrodes", self.electrodes

        print "--> Attempting to build gpu simulator"
        print self.morphology.shape
        self.simulator = Cstmd1Sim(self.morphology, self.dt, debug_mode, buffer_size, parameters=self.sim_parameters,
                                   device=cuda_device)
        print "Finished building gpu simulator "

        if self.enable_synapses:
            max_conductance = np.float32(sim_parameters["synapse_max_conductance"])
            print "--> Attempting to load synapses onto gpu simulator with g_max = " + str(max_conductance)
            self.simulator.load_synapses(self.synapses, max_conductance)
            print "Finished loading synapses onto gpu simulator "

        print "--> Attempting to load electrodes onto gpu simulator"
        if self.enable_spike_dump:
            self.simulator.load_electrodes(self.all_compartments)
        else:
            self.simulator.load_electrodes(self.electrodes)
        print "Finished loading electrodes onto gpu simulator "

        print "=====Constructed CSTMD1 Module with " + str(self.number_of_neurons) + " neurons===== "

    def __build_estmd_mapping(self, estmd_dim, estmd_map_f_name, topological=False):

        print "--> Attempting to get estmd mapping"

        if estmd_map_f_name is not None:
            estmd_map_path = os.path.join(self.morphology_path, estmd_map_f_name)
            print "Getting estmd mapping from " + estmd_map_path
            self.neuron_collection.import_estmd1_mapping_from_file(path=estmd_map_path, pixel_array_shape=estmd_dim)

        self.estmd_mapping = self.neuron_collection.cstmd1_sim_get_estmd1_mapping(height=estmd_dim[1],
                                                                                  width=estmd_dim[0],
                                                                                  generate_new=estmd_map_f_name is None,
                                                                                  random=True,
                                                                                  topological=topological)

        if estmd_map_f_name is None:
            mapping = np.array([(x, y, v) for (x, y), v in sorted(self.estmd_mapping.iteritems())])
            if topological:
                new_mapping = self.run_id + "_topological_height:" + str(estmd_dim[1]) + "x_width:" + str(
                        estmd_dim[0]) + "_estmd_mapping.dat"
            else:
                new_mapping = self.run_id + "_height:" + str(estmd_dim[1]) + "x_width:" + str(
                        estmd_dim[0]) + "_estmd_mapping.dat"
            np.savetxt(os.path.join(self.morphology_path, new_mapping), mapping, "%i")

        print "Finished generating estmd mapping "

    def __build_electrodes(self, electrodes_f_name, soma, n, random):

        print "--> Attempting to get electrodes"

        if electrodes_f_name is not None:
            electrode_path = os.path.join(self.morphology_path, electrodes_f_name)
            print "Getting electrodes from " + electrode_path
            self.neuron_collection.import_electrodes_from_file(path=electrode_path)

        if electrodes_f_name is None:
            assert soma is not None or n is not None

        self.electrodes = self.neuron_collection.cstmd1_sim_get_electrodes(generate_new=electrodes_f_name is None,
                                                                           soma=soma,
                                                                           random=random,
                                                                           number=n)
        if self.enable_spike_dump:
            self.all_compartments = np.array(self.neuron_collection.get_compartment_idxs(
                    self.neuron_collection.total_compartments()), dtype=np.int32)
            print "Will record all compartments but only connect electrodes to STDP"
        self.n_electrodes = len(self.electrodes)

        print "Finished adding " + str(self.n_electrodes) + " electrodes "

    def __build_synapses(self, synapses_f_name, n, min_distance):

        print "--> Attempting to get synapses"
        if synapses_f_name is not None:
            synapse_path = os.path.join(self.morphology_path, synapses_f_name)
            print "Getting synapses from " + synapse_path
            self.neuron_collection.import_synapses_from_file(path=synapse_path)
            self.synapses = self.neuron_collection.cstmd1_sim_get_synapses()
        else:
            print "Getting new synapses, this may take a while (number " + str(n) + " distance " + str(
                    min_distance) + ")"
            assert n is not None and min_distance is not None, \
                "To generate new synapses need to provide number and min distance"
            self.synapses = self.neuron_collection.cstmd1_sim_get_synapses(generate_new=True,
                                                                           number_between_adjacent=n,
                                                                           min_distance=min_distance)

            nsp = os.path.join(self.morphology_path, self.run_id + "_" + str(n) +
                               "_" + str(min_distance) + "_synapses.dat")
            np.savetxt(nsp, self.synapses, "%i")
            print "Saved new synapses to " + nsp

        print "Finished adding " + str(self.synapses.shape[0]) + " synapses "

    def __build_neuron_model(self, preload_path, number_of_neurons, morphology_path, morphology_prefix,
                             homogenise=False):
        """
        Args:
            number_of_neurons (int): number of neurons in the model
            morphology_path (str): path to directory containing morphology files
            morphology_prefix (str): prefix of morphology swc file
            homogenise (bool): standardise the lengths of the compartments

        Returns: False if could not find all required swc files
        """

        if preload_path:
            try:
                self.neuron_collection = pickle.load(open(preload_path))
            except:
                print "Failed to preload neuron collection. Generating new neuron file"
                preload_path = None

        if preload_path is None:
            print "--> Attempting to build " + str(number_of_neurons) + " neurons"
            self.neuron_collection = NeuronCollection()

            for i in xrange(number_of_neurons):
                mcn = MultiCompartmentalNeuron()
                path = os.path.join(morphology_path, morphology_prefix + str(i) + '.swc')
                try:
                    mcn.construct_from_SWC(path, [-10, 0, 0])
                    print "Constructed neuron " + str(i) + " from " + path
                except(IOError):
                    print "Could not construct neuron " + str(i)
                    return False
                if homogenise:
                    print "Homogenising lengths for neuron " + str(i)
                    mcn.homogenise_lengths(0.1)

                self.neuron_collection.add_neuron(neuron=mcn)

        if homogenise:
            l = self.neuron_collection.cstmd1_sim_get_median_length() / (10e6)
            print "Overriding parameter: median length of compartment {:.3e} m".format(l)
            self.sim_parameters["l"] = l

        self.morphology = self.neuron_collection.cstmd1_sim_get_electical_connections()
        if self.save_bare_nc_path is not None:
            self.neuron_collection.save_to_file(self.save_bare_nc_path)
        print "Neuron model"
        print self.neuron_collection
        print "Finished constructing " + str(self.neuron_collection.number_of_neurons()) + " neurons "
        return True

    def step(self, step_size, estmd_input=None):
        """
        Step the simulator through step_size using zero-order hold of
        estmd_input as the stimulus.

        Args:
            step_size (int): time to run the simulator for (ms)
            estmd_input (List[(int,int,float)]): estmd pixel array, non zero
                stimulus values in the form of a list of (x,y,value != 0)

        Raises:

        Return:
            ndarray[bool]: each element indicates whether the electrode
            connected compartment has spikes
        """

        if estmd_input and estmd_input is not None:
            estmd_mapped_input = self.map_estmd_input(estmd_input=estmd_input)
            if self.verbose_debug:
                print "Mapped input"
                print estmd_mapped_input
            self.simulator.load_estmd_currents(estmd_mapped_input[:, 0].astype(np.int32),
                                               estmd_mapped_input[:, 1].astype(np.float32))

        self.T += step_size
        status, result = self.simulator.run(self.T)
        if not status:
            print "Failed to run simulator"
            return False, []

        self.spike_history.append(result)
        self.dirty = True

        if self.enable_spike_dump:
            result = np.array(result)[self.electrodes]

        return True, np.where(result)[0]

    def map_estmd_input(self, estmd_input):
        """
        Map the estmd input indexed by (x,y) to estmd input indexed by (compartment.idx)

        Args:
            estmd_input (List[(int,int,float)]): estmd pixel array, non zero
                stimulus values in the form of a list of (y,x,value != 0)

        Raises:
            KeyError: if the any coordinate (x,y) has not been mapped onto
                a compartment

        Returns:
            ndarray: estmd input mapped to compartments (compartment.idx, stimulus)
        """
        return np.array([(self.estmd_mapping[(x, y)], stimulus) for y, x, stimulus in estmd_input])

    def save_parameters(self, directory=None, run_id_prefix=False):

        self.save_dictionary(self.sim_parameters, directory, "CSTMD_Simulation_Parameters.dat", run_id_prefix)
        self.save_dictionary(self.moph_parameters, directory, "CSTMD_Morphology_Parameters.dat", run_id_prefix)

    def save_morphology(self, directory=None, run_id_prefix=False):

        if hasattr(self.neuron_collection, "electrodes"):
            self.save_numpy_array(self.neuron_collection.electrodes, directory, "electrodes", npz=True,
                                  run_id_prefix=run_id_prefix)
        if hasattr(self.neuron_collection, "synapses"):
            self.save_numpy_array(np.array(self.neuron_collection.synapses, dtype=np.int32),
                                  directory, "synapses", npz=True, run_id_prefix=run_id_prefix)

    def save_graphs(self, directory=None, run_id_prefix=False):
        if directory is None:
            directory = self.get_output_directory("CSTMD1_Graphs")
        self.plot_spikes(show=False, save_path=os.path.join(directory, "spike_plot.svg"))

        self.plot_spikes(show=False, save_path=os.path.join(directory, "spike_plot.pdf"))
        if 50 <= len(self.electrodes) <= 1500 and not self.enable_spike_dump:
            self.plot_spikes(show=False, save_path=os.path.join(directory, "spike_plot_expanded.svg"), expand = True)
        self.plot_firing_rate(False, save_path=os.path.join(directory, "spike_rate_plot.svg"))

    def save_spike_rate_data(self, directory=None, name="cstmd_spike_rates.dat", run_id_prefix=False):
        """
        Save electrode spike rate data

        Args:
            directory (str): directory to save to, default to current
            name (str): name of the file
            run_id_prefix (bool): prefix the name with the current run id
        """
        self.save_numpy_array(self.spike_rate_data, directory, name, run_id_prefix=run_id_prefix)

    def save_spikes(self, directory=".", name="cstmd_spikes.dat", run_id_prefix=False, time_range=None,transpose=False):
        """
        Save electrode spike record

        Args:
            directory (str): directory to save to, default to current
            name (str): name of the file
            run_id_prefix (bool): prefix the name with the current run id
        """
        data = np.array(self.spike_history)

        if time_range is not None:
            min, max = time_range
            data = data[min:max, :]

        self.save_numpy_array(data, directory, name, fmt="%i", run_id_prefix=run_id_prefix,transpose=transpose)

    def save_voltages(self, directory=None, name="cstmd_voltages.dat", run_id_prefix=False):
        """
        Args:
            directory (str): directory to save to, default to current
            name (str): name of the file
            run_id_prefix (bool): prefix the name with the current run id
        """

        self.save_numpy_array(self.get_voltages(), directory, name, run_id_prefix=run_id_prefix)

    # def print_spikes(self):

    #     print
    #     spikes = np.array(self.spike_history)
    #     print 'Step\t| Spikes'
    #     print 'idx\t|', self.electrodes
    #     for i, entry in enumerate(spikes):
    #         print i, '\t|', entry
    #     print

    def get_voltages(self):
        """
        Gets the voltages from CUDA device
        Indexed like [time,compartment]

        Returns:
            ndarray: voltages at each compartment and timestep
        """
        if self.v is None or self.dirty is True:
            v = self.simulator.get_voltages()
            n_compartments = self.neuron_collection.total_compartments()
            self.v = np.array(v).reshape([len(v) / n_compartments, n_compartments])

        self.dirty = False
        t = int(self.T / self.dt)
        return self.v[:t, :]

    def get_recovery_variables(self):
        """
        Gets the recovery variables from CUDA device
        Each array is indexed [time,compartment]

        Returns:
            (ndarray,ndarray,ndarray): the 3 recovery variables
        """
        if self.m is None or self.dirty is True:
            m, n, h = self.simulator.get_recovery_variables()
            n_compartments = self.neuron_collection.total_compartments()
            self.m = np.array(m).reshape([len(m) / n_compartments, n_compartments])
            self.n = np.array(n).reshape([len(n) / n_compartments, n_compartments])
            self.h = np.array(h).reshape([len(h) / n_compartments, n_compartments])

        self.dirty = False
        t = int(self.T / self.dt)
        return self.m[:t, :], self.n[:t, :], self.h[:t, :]

    # def print_voltages(self, variable='v'):
    #     """
    #     Print out the output of get_voltages with some formatting
    #     """
    #     if variable == 'v' and (self.v is None or self.dirty is True):
    #         print "Retrieving voltages"
    #         self.get_voltages()
    #     elif self.dirty is True:
    #         print "Retrieving recovery variables"
    #         self.get_recovery_variables()

    #     if variable == 'v':
    #         data = self.v
    #     elif variable == 'm':
    #         data = self.m
    #     elif variable == 'n':
    #         data = self.n
    #     elif variable == 'h':
    #         data = self.h

    #     print 'Step\t| Voltage at neuron'
    #     for i in xrange(len(data)):
    #         if not i % self.morphology.shape[0]:
    #             if i != 0:
    #                 print
    #             print i / self.morphology.shape[0], '\t|',

    #         print '{:<10}'.format(round(data[i], 4)),

    #     print

    #     self.dirty = False
    #     return

    def raster(self, event_times_list, color='k'):
        """
        Creates a raster plot
        Parameters
        ----------
        event_times_list : iterable
                           a list of event time iterables
        color : string
                color of vlines
        Returns
        -------
        ax : an axis containing the raster plot
        """
        ax = plt.gca()
        for ith, trial in enumerate(event_times_list):
            plt.vlines(trial, ith + .5, ith + 1.5, color=color)
        plt.ylim(.5, len(event_times_list) + .5)
        return ax

    def plot_spikes(self, show=False, save_path=None, expand = False):
        """
        Plot the spikes for each of the electrodes

        Args:
            expand ():
            save_path ():
            show ():
        """
        spikes = np.array(self.spike_history)
        spike_time, e_idx = np.where(spikes)
        spike_time = spike_time.astype('float32')
        spike_time *= self.global_dt
        spike_time_pair = zip(e_idx,spike_time)
        spike_time_pair.sort()
        spike_time_pair = np.array(spike_time_pair)
        spike_time_pair = list(np.split(spike_time_pair, np.where(np.diff(spike_time_pair[:,0]))[0]+1))

        if self.enable_spike_dump:
            n = len(self.all_compartments)
        else:
            n = len(self.electrodes)

        s = []
        for i in xrange(n):
            s1 = [t[:,1] for t in spike_time_pair if t[0,0] == i]
            s.append(s1)

        fig = plt.figure()
        ax = self.raster(s)

        if n < 50 or expand:
            ax.set_yticks(np.arange(1, n + 1))
            if self.enable_spike_dump:
                ax.set_yticklabels(tuple(self.all_compartments))
            else:
                ax.set_yticklabels(tuple(self.electrodes))
        else:
            ax.set_yticklabels([])

        ax.set_ylabel('Electrode IDX')
        ax.set_xlabel('Time (msec)')
        ax.set_title('CSTMD Electrode Spikes for ' + str(n) + ' compartments')

        if not show and expand:
            if n > 40:
                w,h = fig.get_size_inches()
                h *= n / 40
                fig.set_size_inches(w,h)

        if save_path is not None:
            #fig.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            print "Saved Cstmd spike train to " + save_path
            plt.gcf().clear()
        if show:
            plt.show()

    def rate_filter(self, data, low_pass=False, tau=30.0):

        m = 8 * tau / self.global_dt
        r = np.linspace(-(m - 1), 0, m)
        b = np.exp(r / tau)

        b_reversed = b[::-1]
        out = signal.lfilter(b_reversed, [1.0], data) * 1000
        normalising = []
        for i in xrange(len(out)):
            l = min(i, int(m)) + 1
            out[i] /= b_reversed[:l].sum()
            normalising.append(b_reversed[:l].sum())
        if low_pass:
            b, a = signal.butter(2, self.global_dt / 6, 'low')
            return signal.lfilter(b, a, out)
        else:
            return out

    def calculate_spiking_rate_exp(self, low_pass=True, tau=20.0):

        spikes = np.array(self.spike_history)

        time_steps, noElectrodes = spikes.shape

        if self.verbose_debug:
            print "Time steps", self.T, "ms"
            print "Time steps", time_steps, "steps"
            print "Total spikes", spikes.sum()

        # spike_rate_data example:
        # Time Electrode1 Electrode2 .....
        #    0     1.53       3.67    .....
        self.spike_rate_data = np.zeros((time_steps, len(self.electrodes) + 1))

        for j in xrange(time_steps):
            self.spike_rate_data[j, 0] = j * self.global_dt

        for i in xrange(len(self.electrodes)):
            self.spike_rate_data[:, i + 1] = self.rate_filter(spikes[:, i], low_pass=low_pass, tau=tau)

        return self.spike_rate_data

    def plot_firing_rate(self, show=True, save_path=None, low_pass=True, tau=30.0):

        if self.enable_spike_dump:
            print "WARNING spike dump enabled so will not save spike rate plots as there would too many"
            return

        spike_rate_data = self.calculate_spiking_rate_exp(low_pass=True, tau=30.0)
        T, n_electrodes = spike_rate_data.shape

        fig = plt.figure()
        y_label = "F_Rate Hz"
        w_count = 1
        if not show:
            if (n_electrodes - 1) > 5:
                w,h = fig.get_size_inches()
                h *= (n_electrodes - 1)/ 20
                w_count = 4
                w *= w_count
                fig.set_size_inches(w,h)
                y_label = ""

        for i in xrange(1, n_electrodes):
            ax = plt.subplot(np.ceil(float(n_electrodes - 1)/w_count), w_count, i)
            if i <= w_count:
                ax.set_title("Electrode Firing Rate /Hz")
            ax.plot(spike_rate_data[:, 0], spike_rate_data[:, i])
            ax.yaxis.grid(True)
            ax.set_ylabel(y_label + str(self.electrodes[i-1]))

        plt.xlabel('Time (msec)')
        if save_path is not None:
            fig.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.gcf().clear()
            print "Saved Cstmd spike rate to " + save_path

        if show:
            plt.show()

    def plot_compartments(self, compartments, show=True, save_path=None, names=None):
        """
            Plot problematic comparment and its neighbours
            Kwargs:
                no_adjacent_comparmtents : how many neighbours to plot
            Returns:
        """
        print 'Copying voltages and recovery variables for debugging...'
        # Note if they have already been copied then the following functions won't
        # do a copy
        v = self.get_voltages()

        # Check for infinities
        x, y = np.where(np.isinf(np.array(v)))
        print "Bad compartments found ", y
        _, naning = np.where(np.isnan(np.array(v)))
        print "Nans found:", naning
        # If none found, yay and set tmax to whole range
        if len(x) == 0:
            t_max = v.shape[0]
        else:
            t_max = x[0] - 1
        print "Voltage shape", v.shape

        m, n, h = self.get_recovery_variables()
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
        neighbours = self.neuron_collection.get_neighbour_idx(compartments[1])

        ax1.set_title('Compartments ' + str(compartments))
        ax1.set_ylabel('Voltage /mV')
        ax2.set_ylabel('m')
        ax3.set_ylabel('n')
        ax4.set_ylabel('h')
        ax4.set_xlabel('Step')
        ax5.axis('off')
        for index, i in enumerate(np.append(compartments, neighbours)):

            p_v = v[:t_max, i]
            p_m = m[:t_max, i]
            p_n = n[:t_max, i]
            p_h = h[:t_max, i]

            if index < len(names):
                label = names[index]
            else:
                label = "c_idx:" + str(i)

            ax1.plot(p_v, label=label)
            ax2.plot(p_m, label=label)
            ax3.plot(p_n, label=label)
            ax4.plot(p_h, label=label)
        ax1.set_ylim(-50, 200)
        ax2.set_ylim(0, 1)
        ax3.set_ylim(0, 1)
        ax4.set_ylim(0, 1)

        handles, labels = ax1.get_legend_handles_labels()
        f.legend(handles, labels, loc='lower center', ncol=3, borderaxespad=2)

        if save_path is not None:
            #fig.tight_layout()
            plt.savefig(save_path)
            print "Saved figure to " + save_path
        if show:
            plt.show()
        if not show:
            plt.clf()

    def plot_synapse_compartments(self, directory=None):

        if directory is None:
            directory = self.get_output_directory("Synapse_Plots")
        print "Outputting synapse plots to:", directory

        if self.synapses is None:
            print "No synapses to get"

        for pair in self.synapses:
            self.plot_compartments(pair, False,
                                   os.path.join(directory, "synapse_plt_" + str(pair[0]) + "_" + str(pair[1]) + ".svg"),
                                   ['pre', 'post'])

    # def plot_problematic_compartments(self, no_adjacent_comparmtents=1, show=True, save_path=None):
    #     """
    #         Plot problematic comparment and its neighbours
    #         Kwargs:
    #             no_adjacent_comparmtents : how many neighbours to plot
    #         Returns:
    #     """
    #     print 'Copying voltages and recovery variables for debugging...'
    #     # Note if they have already been copied then the following functions won't
    #     # do a copy
    #     v = self.get_voltages()
    #     m, n, h = self.get_recovery_variables()
    #     # find the inf
    #     x, y = np.where(np.isinf(np.array(v)))
    #     print 'First 3 problematic comparments:'
    #     print x[:3]
    #     print 'First 3 problematic times:'
    #     print y[:3]
    #     plt.figure()
    #     target = y[0]
    #     print 'Plotting the ', no_adjacent_comparmtents, 'comparments around ', target
    #     f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    #     for i in range(target - no_adjacent_comparmtents, target + no_adjacent_comparmtents + 1):
    #         p_v = v[:(x[0] - 1), i]
    #         p_m = m[:(x[0] - 1), i]
    #         p_n = n[:(x[0] - 1), i]
    #         p_h = h[:(x[0] - 1), i]
    #         ax1.plot(p_v)
    #         ax1.set_title('voltage')

    #         ax2.plot(p_m)
    #         ax2.set_title('m')

    #         ax3.plot(p_n)
    #         ax3.set_title('n')

    #         ax4.plot(p_h)
    #         ax4.set_title('h')

    #         ax1.set_ylim(-50, 200)
    #         ax2.set_ylim(0, 1)
    #         ax3.set_ylim(0, 1)
    #         ax4.set_ylim(0, 1)

    #         print 'Minimux and maximum values at compartment: ', i
    #         print 'v', round(p_v.min(), 2), round(p_v.max(), 2)
    #         print 'm', round(p_m.min(), 2), round(p_m.max(), 2)
    #         print 'n', round(p_n.min(), 2), round(p_n.max(), 2)
    #         print 'h', round(p_h.min(), 2), round(p_h.max(), 2)
    #         print
    #         # plt.plot(data,label=str(i))
    #         # plt.ylim(-50,200)
    #     # plt.legend(loc=2)
    #     if save_path is not None:
    #         plt.savefig(save_path)
    #         plt.gcf().clear()
    #     if show:
    #         plt.show()
