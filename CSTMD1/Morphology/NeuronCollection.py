'''
NeuronCollection

Data structure for holding collections of neurons

__author__: Dragonfly Project 2016 - Imperial College London ({anc15, cps15, dk2015, gk513, lm1015,
zl4215}@imperial.ac.uk)
'''

import random as r
from itertools import product
import os
from Helper.Object_Array import Pixel_Array
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from matplotlib import collections  as mc
from progressbar import Percentage, Bar, ETA, ProgressBar
from scipy.linalg import block_diag

from CSTMD1Visualiser import CSTMD1Visualiser
from Helper.Vectors import Vector_2D
from MultiCompartmentalNeuron import MultiCompartmentalNeuron


class NeuronCollection(object):
    """ A container for holding a collection of neurons. Allows the generation
    of synapses and matrices required for the CSTMD1 simulator

    Attributes:
        neurons (List[MultiCompartmentalNeuron]): list of the neurons which
            make up the connection
        synapse_groups (List[(int,int,List[Compartment])]: list of synpases
            groupings where each entry is a synapse set between two neurons
            given by their idx
    """

    neurons = None
    synapse_groups = None


    def __init__(self):
        """
        NeuronCollection constructor, initialised neurons and synapse_groups to empty lists
        """
        self.neurons = []
        self.synapse_groups = []
        self.debug = False

    def save_to_file(self, path):

        try:
            fileObj = open(path, 'wb')
        except IOError:
            return False

        pickle.dump(self, fileObj, -1)
        fileObj.close()
        print "Saved to " + path
        return True

    def add_neuron(self, neuron):
        """
        Add a neuron to the neuron collection.
        !!!Neurons should be added in the correct order. i.e adjacent order.
        The neuron will be set to be adjacent to the previous neuron added

        Args:
            neuron (MultiCompartmentalNeuron): neuron to add

        Notes:
            The neuron's idx will be set to the number of neurons in the
            collection.
        """

        neuron.idx = self.number_of_neurons()

        self.neurons.append(neuron)

        neuron.adjacent = None
        if (self.number_of_neurons() > 1):
            self.neurons[-2].adjacent = neuron
        if self.debug:
            print str(neuron) + " added "

    def get_spikes_from_compartments(self, compartments_idx):

        assert hasattr(self, "electrodes"), "No electrodes"
        comp = self.get_compartment(self.electrodes[0])
        assert hasattr(comp, "spike_record")

        spikes = np.empty((len(comp.spike_record), len(compartments_idx)))

        for i, index in enumerate(compartments_idx):
            c = self.get_compartment(index)
            assert hasattr(c, "spike_record"), "Trying to get spikes from compartment without data"
            spikes[:, i] = c.spike_record.transpose()

        return spikes

    def load_bulk_electrode_data(self, voltage=None, spikes=None):
        """
        Load voltage

        Args:
            voltage (ndarray): voltage array, mxn m = max time step and
                n = number of compartments
        """

        assert voltage is not None or spikes is not None, "No data given"

        assert hasattr(self, "electrodes"), "No electrodes to map data to"

        if spikes is not None:
            assert spikes.shape[1] == len(self.electrodes)

        if voltage is not None:
            assert voltage.shape[1] == len(self.electrodes)

        electrodes = self.map_electrodes_to_compartments()
        print "Loading spikes for " + str(len(electrodes)) + " electrodes"

        for i, electrode in enumerate(electrodes):
            if voltage is not None:
                electrode.voltage_record = voltage[:, i]
            if spikes is not None:
                electrode.spike_record = spikes[:, i]

    def load_electrode_data(self, spikes):
        """

        Args:
            spikes ():

        Returns:

        """

        assert hasattr(self, "electrodes"), "No electrodes to map data to"

        assert len(spikes) == len(self.electrodes)

        for i, electrode in enumerate(self.map_electrodes_to_compartments()):
            if hasattr(electrode, "spike_record"):
                electrode.spike_record = np.append(electrode.spike_record, spikes[i])
            else:
                electrode.spike_record = np.array([spikes[i]])

    def load_voltage_from_file(self, path):
        """
        Load voltage data from file

        Args:
            path (str): path to the voltage file

        Raises:
            IOError: if path is invalid
        """
        voltage = np.loadtxt(path)
        self.load_bulk_electrode_data(voltage=voltage)

    def load_spikes_from_file(self, path):
        """
        Load spike train from file

        Args:
            path (str): path to the spike file

        Raises:
            IOError: if path is invalid
        """

        spikes = np.loadtxt(path, dtype=bool)
        self.load_bulk_electrode_data(spikes=spikes)

    def number_of_neurons(self):
        """
        Get the number of neurons in the collection

        Returns:
            int: number of neurons
        """
        return len(self.neurons)

    def __checkIdx(self, idx):
        """
        Check that the collection contains a neuron with the given idx

        Args:
            idx (int): index of the neuron to check

        Returns:
            bool: if collection contains neuron with given idx
        """

        if (idx < 0 or idx > self.number_of_neurons() - 1):
            print str(idx) + " is not a valid idx for this compartment"
            return False

        return True

    def import_synapses_from_file(self, path):
        """
        Import synapses from file

        Args:
            path (str): path to the synapse file

        Raises:
            IOError: if path is invalid
        """
        synapses = np.loadtxt(path, int, delimiter=" ", ndmin=2);

        start = self.collection_compartment_Idx_offset()
        self.synapses = synapses
        self.synapse_groups = []

        pairs = [(i,j) for (i,j) in product(xrange(self.number_of_neurons()),
                                            xrange(self.number_of_neurons())) if i !=j]

        for i,j in pairs:
            self.synapse_groups.append([i, j, []])

        for (pre, post) in synapses:

            pre_compartment = self.get_compartment(pre)
            post_compartment = self.get_compartment(post)
            pre_compartment_neuron = pre_compartment.neuron.idx
            post_compartment_neuron = post_compartment.neuron.idx

            entry_idx = [(i[0],i[1]) for i in self.synapse_groups].index((pre_compartment_neuron,
                                                                          post_compartment_neuron))

            self.synapse_groups[entry_idx][2].append((pre_compartment,post_compartment))

    def generate_synapses(self, idx1, idx2, number_of_synapses, max_distance):
        """
        Generate synapses between two neurons with given idx. Synapse is
        generated if the middle of the compartments are within max_distance of
        one another. If more than number_of_synapses are found, a random sample
        if taken

        Args:
            idx1 (int): Index of neuron 1
            idx2 (int): Index of neuron 2
            number_of_synapses (int): Number of synapses to generate
            max_distance (float): Max distance below which to a synapse is
                generated

        Raises
            AssertionError: if idx1 or idx2 is invalid

        Returns:
            List[(MultiCompartmentalNeuron,MultiCompartmentalNeuron)] list of
            synapses, with the form of a list of tuples (pre,post)
        """

        assert self.__checkIdx(idx1)
        assert self.__checkIdx(idx2)

        up = [idx1,idx2,[]]
        down = [idx2,idx1,[]]

        neuron = self.neurons[idx1]
        adjacent_neuron = self.neurons[idx2]

        widgets = ['Calculating Synapses from ' + str(idx1) + ' to ' + str(idx2) + ' ', Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=(len(neuron.compartments))).start()

        for compartment in pbar(neuron.compartments):
            for opposingCompartment in adjacent_neuron.compartments:
                if compartment.midpoint().distance(opposingCompartment.midpoint()) <= max_distance:
                    if r.random() < 0.5:
                        up[2].append((compartment,opposingCompartment))
                    else:
                        down[2].append((opposingCompartment,compartment))

        found = len(up[2]) + len(down[2])

        print "Found " + str(found) + " synapses between neuron " + str(neuron.idx) + " and " + str(adjacent_neuron.idx)
        print str(len(up[2])) + " up and " + str(len(down[2])) + " down "

        if found < number_of_synapses:
            print "Reduce maxDistance as only found " + str(found) + " synapses"
            self.synapse_groups.append(up)
            self.synapse_groups.append(down)
            return up, down

        else:
            n_up = int(np.ceil((float(len(up[2])) / found) * number_of_synapses))
            n_down = int(np.floor((float(len(down[2])) / found) * number_of_synapses))
            up[2] = r.sample(up[2], n_up)
            down[2] = r.sample(down[2], n_down)

        found = len(up[2]) + len(down[2])

        print "Reduced to " + str(found) + " synapses ",
        print str(len(up[2])) + " up and " + str(len(down[2])) + " down "
        self.synapse_groups.append(up)
        self.synapse_groups.append(down)

        return up, down

    def cstmd1_sim_get_synapses(self, generate_new=False, number_between_adjacent=None, min_distance=None):
        """
        Get the synapses in the form (pre.idx,post.idx) where the idx are
        altered such that every compartment in the NeuronCollection is unique.

        Notes:
            If neuron0 has 10 compartments 0..9 and neuron1 has compartments
            0..9 then neuron1's first compartment have idx 10..19

        Args:
            generate_new (bool): Generate new synapses
            number_between_adjacent (int): number of synapses to generate
                between adjacent neurons
            min_distance (float): minimum distance to consider a synapse

        Raises:
            AssertionError: if generate_new is false and no synapses are found

        Returns:
            ndarray[int,int]: array of synapses (pre.idx,post.idx)

        """

        start = self.collection_compartment_Idx_offset()
        synapses = []

        if not generate_new:
            assert len(self.synapses) != 0, "No synapses found, generate new"

        if generate_new and number_between_adjacent is not None and min_distance is not None:
            for neuron in self.neurons:
                # if neuron.adjacent is not None:
                for neuron_adj in self.neurons:
                    if neuron is neuron_adj:
                        continue
                    if hasattr(neuron, "synapses_done_for"):
                        if neuron_adj in neuron.synapses_done_for:
                            continue
                    self.generate_synapses(neuron.idx, neuron_adj.idx, number_between_adjacent,min_distance)
                    if not hasattr(neuron_adj, "synapses_done_for"):
                        neuron_adj.synapses_done_for = [neuron]
                    else:
                        neuron_adj.synapses_done_for.append(neuron)

            self.synapses = []
            for (_, _, synapseGroup) in self.synapse_groups:
                for synapse in synapseGroup:
                    pre, post = synapse
                    self.synapses.append((self.get_global_idx(pre), self.get_global_idx(post)))

        return np.array(self.synapses, dtype=np.int32)

    def cstmd1_sim_get_median_length(self):

        radii_lengths = self.cstmd1_sim_get_radii_lengths()
        return np.median(radii_lengths[0, :], axis=0)

    def cstmd1_sim_get_radii_lengths(self):
        """
        Get compartment data as columns of numpy array

        Returns:
            ndarray[float,float]: array with first row being the compartment
            lengths and the second column the compartment radii
        """
        radii = np.empty((0), dtype=float)
        length = np.empty((0), dtype=float)
        for neuron in self.neurons:
            l, r = neuron.compartment_data()
            radii = np.concatenate((radii, r))
            length = np.concatenate((length, l))

        return np.vstack((length, radii))

    def cstmd1_sim_get_electrodes(self, generate_new=False, soma=False, random=False, number=0):
        """
        Get a number of electrodes

        Args:
            generate_new (bool): get new electrodes
            number (int): number of electrodes to get
            soma (bool): get electrodes at the soma
            random (bool): get random electrodes, otherwise they are linearly dispersed

        Raises:
            AssertionError: if there are not enough compartments to get unique
                electrodes, or generate_new if false and no synapses found

        Returns:
            ndarray[np.int32]: array of compartment idx
        """

        if generate_new:
            if soma:
                print "Getting soma electrodes"
                self.electrodes = self.collection_compartment_Idx_offset()[:-1].astype(np.int32)
            else:
                print "Getting " + str(number),
                if random:
                    print "random",
                print "electrodes",

                self.electrodes = np.array(self.get_compartment_idxs(number, random), dtype=np.int32)
                print " - got " + str(len(self.electrodes))
        else:
            assert hasattr(self, 'electrodes'), "No electrodes found, generate new"

        return self.electrodes

    def get_compartment(self, n):
        """

        Args:
            n (int): compartment global idx

        Raises:
            AssertionError: n is out of bounds

        Returns:
            Compartment
        """
        start = self.collection_compartment_Idx_offset()

        assert n < start[-1], "n is out of range"

        neuron_id = np.argmin(n >= start) - 1
        return self.neurons[neuron_id].compartments[n - start[neuron_id]]

    def get_compartment_neuron(self, idx):
        start = self.collection_compartment_Idx_offset()
        return np.argmin(idx >= start) - 1

    def get_neighbour_idx(self, compartment_idx):

        compartment = self.get_compartment(compartment_idx)
        neighbours = []
        for c in [compartment.parent] + compartment.children + compartment.siblings:
            if c is not None:
                neighbours.append(self.get_global_idx(c))

        return np.array(neighbours).astype(int)

    def get_global_idx(self, compartment):

        starts = self.collection_compartment_Idx_offset()
        neuron_index = self.neurons.index(compartment.neuron)
        return starts[neuron_index] + compartment.idx

    def map_electrodes_to_compartments(self):
        """
        Map electrode idx to compartments

        Returns:
            List[Compartments]: List of compartments mapped from electrode ids

        """
        if not hasattr(self, "electrodes"):
            return False

        compartments = []
        for i in self.electrodes:
            compartments.append(self.get_compartment(i))

        return compartments

    def import_electrodes_from_file(self, path):
        """
        Import synapses from file

        Args:
            path (str): path to the electrode file

        Raises:
            IOError: if path is invalid
        """
        self.electrodes = np.loadtxt(path).astype(np.int32)

    def get_compartment_idxs(self, number, random=False):
        """
        Get a number of compartments ids.

        Args:
            number (int): number of compartment idx to get
            random (bool): get random compartments

        Raises:
            AssertionError: if there are not enough compartments

        Returns:
            List[int]: List of compartment idx
        """
        total_compartments = self.total_compartments()

        assert number <= total_compartments, "Not enough compartments for mapping"

        if random:
            compartment_idxs = r.sample(xrange(total_compartments), number)
        else:
            n_compartments_per_neuron = [n.number_of_compartments() for n in self.neurons]
            n_electrodes_per_neuron = [int(float(number * n) / sum(n_compartments_per_neuron)) for n in
                                       n_compartments_per_neuron]
            n_electrodes_per_neuron[0] += number - sum(n_electrodes_per_neuron)
            print "(compartments per neuron", n_electrodes_per_neuron, ")",
            neuron_elec_compartments = []
            for n, neuron in zip(n_electrodes_per_neuron, self.neurons):
                step = int(neuron.number_of_compartments() / float(n))
                neuron_elec_compartments += [neuron.compartments[j] for j in xrange(0, step * n, step)]

            index_start = self.collection_compartment_Idx_offset()
            compartment_idxs = [index_start[c.neuron.idx] + c.idx for c in neuron_elec_compartments]

        return compartment_idxs

    def import_estmd1_mapping_from_file(self, path, pixel_array_shape):
        """
        Import estmd mapping

        Args:
            path (str): path to the mapping file
            pixel_array_shape (int,int): width, height array shape

        Raises:
            IOError: if path is invalid
            AssertionError: if file does not contain mapping for pixels
        """
        mapping = np.loadtxt(path)

        assert mapping.shape[0] == pixel_array_shape[0] * pixel_array_shape[1], "Wrong number of mappings"

        self.mapping = dict(zip(zip(mapping[:, 0], mapping[:, 1]), mapping[:, 2]))

    def cstmd1_sim_get_estmd1_mapping(self, height, width, generate_new=False, random=False, topological=False):
        """
        Get a mapping from the estmd pixel array with height and width, to the
        compartments

        Args:
            height (int): height of the pixel array
            width(int): width of the pixel array

        Raises:
            AssertionError: if there are not enough compartments to create a
                1:1 mapping, or generate_new is false and no preloaded mapping

        Returns:
            dictionary[(int,int):int]: dictionary keys by coordinate (x,y)
            (mapping to the compartment)
        """

        assert height * width <= self.total_compartments()

        if generate_new:
            print "Getting new mapping for pixels array " + str(height) + " high and " + str(width) + " wide",
            if topological:
                print "(topological)"
                return self.topological_mapping(height, width, randomize_order=random)

            else:
                if random:
                    print "(random)",
                else:
                    print "(linearly dispersed)",
                compartment_idxs = self.get_compartment_idxs(height * width, random)
                print "- got " + str(len(compartment_idxs))
                return dict(zip(list(product(xrange(width), xrange(height))), compartment_idxs))
        else:
            assert hasattr(self, 'mapping'), "No mapping found, generate new"
            return self.mapping

    def get_all_compartments(self, include_axon=False):
        compartments = []
        for neuron in self.neurons:
            compartments += [c for c in neuron.compartments if (include_axon or (not c.axon_comp))]
        return compartments

    def pixel_midpoints(self, height, width, compartments=None, projection_axis="x", include_axon = False):

        if compartments is None:
            compartments = self.get_all_compartments(include_axon)

        max_x = max(compartments, key=lambda x: x.midpoint().project(projection_axis).x).midpoint().project(
            projection_axis).x
        min_x = min(compartments, key=lambda x: x.midpoint().project(projection_axis).x).midpoint().project(
            projection_axis).x
        max_y = max(compartments, key=lambda x: x.midpoint().project(projection_axis).y).midpoint().project(
            projection_axis).y
        min_y = min(compartments, key=lambda x: x.midpoint().project(projection_axis).y).midpoint().project(
            projection_axis).y
        # print max_x, min_x, max_y, min_y
        x_grid_interval = (max_x - min_x) / width
        y_grid_interval = (max_y - min_y) / height

        x_int = np.linspace(min_x + x_grid_interval / 2, max_x - x_grid_interval / 2, width)
        y_int = np.linspace(min_y + y_grid_interval / 2, max_y - y_grid_interval / 2, height)

        indices = list(product(xrange(width), reversed(xrange(height))))
        pixels = [Vector_2D(x, y) for x, y in product(x_int, y_int)]

        for i, pixel in enumerate(pixels):
            pixel.idx = indices[i]

        return pixels

    def topological_mapping(self, height, width, randomize_order=False, plot=False, axis = "x", n = None):

        compartments = self.get_all_compartments(False)

        """
        z
        ^
        |
        |
        --> y
        """

        pixels = self.pixel_midpoints(height, width, compartments, projection_axis=axis)

        if randomize_order:
            r.shuffle(pixels)

        pixel_to_compartment = []
        pixel_idx_to_compartment_idx = {}

        total_error = 0

        widgets = ['Topological Mapping: ', Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA()]

        pbar = ProgressBar(widgets=widgets, maxval=(len(pixels))).start()

        for pixel in pbar(pixels):
            i = 0
            if n is not None:
                candidates = [c for c in compartments if (not hasattr(c, 'taken') and c.steps_to_root(True) > n)]
            else:
                candidates = [c for c in compartments if not hasattr(c, 'taken')]
            j = 0
            while len(candidates) == 0:
                i += 10
                if n is not None:
                    candidates = [c for c in compartments if (not hasattr(c, 'taken') and c.steps_to_root(True) > (n-i))]
                else:
                    candidates = [c for c in compartments if not hasattr(c, 'taken')]
                j += 1
                if j > 5:
                    print i, pixel, "Warning cannot find compartment"
                if i == n:
                    print "Warning no compartments left"
                    raise Exception

            closest_compartment = min(candidates,key=lambda c: pixel.distance(c.midpoint().project(axis)))

            midpoint = closest_compartment.midpoint().project(axis)
            d = pixel.distance(midpoint)

            pixel_to_compartment.append((pixel, closest_compartment))
            pixel_idx_to_compartment_idx[pixel.idx] = self.get_global_idx(closest_compartment)

            total_error += d

            closest_compartment.taken = True
            print str(pixel), pixel.idx, closest_compartment.steps_to_root(True)



        print "Mapped " + str(len(pixels)) + " pixels with total offset between pixel and compartments of " + str(
            total_error)

        if plot:
            self.plot_plan(pixels=pixels, pixel_mapping=pixel_to_compartment, full=False)

        return pixel_idx_to_compartment_idx

    def total_compartments(self):
        """
        Get total number of compartments in the collection

        Returns:
            int: number of compartments
        """

        return self.collection_compartment_Idx_offset()[-1]

    def cstmd1_sim_get_electical_connections(self):
        """
        Get the electrical connections between compartments in neuron collection.

        See Also:
            MultiCompartmentalNeuron.connection_matrix()

        Example:
            >>> mcn1 = MultiCompartmentalNeuron()
            >>> mcn2 = MultiCompartmentalNeuron()
            ...
            >>> c1 = mcn1.connection_matrix()
            >>> c2 = mcn1.connection_matrix()
            >>> nc = NeuronCollection()
            >>> nc.add_neuron(mcn1)
            >>> nc.add_neuron(mcn2)
            >>> nc.cstmd1_sim_get_electical_connections()
            [[c1 0],[0 c2]]

        Returns:
            ndarray[np.int32]: Matrix of electrical connections
        """
        if hasattr(self,"con"):
            print self.con.shape
            print self.total_compartments()
            if self.con.shape[0] == self.con.shape[1] == self.total_compartments():
                return self.con
        else:
            print "Generating new matrix"

        connections = []

        for neuron in self.neurons:
            connections.append(neuron.connection_matrix())

        connections = block_diag(*connections).astype(np.int32)

        if self.debug:
            print "Connection shape: " + str(connections.shape)
            print "Saving connections"
        self.con = connections

        return connections

    def collection_compartment_Idx_offset(self):
        """
        Find the cumulative sums of the number of compartments in each of the
        neurons in the collection

        Returns:
            ndarray[int]: cumulative sum of the number of compartments
        """
        start = [0]

        for neuron in self.neurons:
            start.append(neuron.number_of_compartments())

        return np.cumsum(start)

    def __str__(self):

        desc = "[Neuron Collection " + str(self.number_of_neurons()) + " neurons: \n"
        for n in self.neurons:
            desc += (str(n) + '\n')
        desc += "]"
        return desc

    def plot(self, synapses=False, expand=50, animate_spikes=False, time_step=50):
        """
        Static 3D plot of neurons in collection
        """
        self.plot_object = CSTMD1Visualiser(self, synapses=synapses, plot_static_electrodes=False, expand=expand,
                                            spikes=animate_spikes, enable_run=True)

        self.plot_object.run(animate_spikes, time_step)

    def plot_estmd_mapping_from_file(self, width, height, map_file):

        compartments = []
        for neuron in self.neurons:
            compartments += [c for c in neuron.compartments if not c.axon_comp]

        pixel_midpoints = self.pixel_midpoints(height, width, compartments)
        pixels = list(product(xrange(width), reversed(xrange(height))))
        pixel_midpoints_mapping = dict(zip(pixels, pixel_midpoints))
        # print pixel_midpoints_mapping

        mappings = np.loadtxt(map_file, int)

        pixel_compartment_pairs = []
        for mapping in mappings:
            c = self.get_compartment(mapping[2])
            pixel_c = (mapping[0], mapping[1])
            pixel = pixel_midpoints_mapping[pixel_c]
            pixel_compartment_pairs.append((pixel, c))

        return self.plot_plan(pixels=pixel_midpoints, pixel_mapping=pixel_compartment_pairs, full=False, lock_aspect
        =False)

    def plot_plan(self, pixels=None, pixel_mapping=None, full=True, lock_aspect=True, num = None):

        lc = edges = None
        if pixel_mapping is not None:
            edges = [[(p.x, p.y), (c.midpoint().y, c.midpoint().z)] for p, c in pixel_mapping]

        if edges is not None:
            lc = mc.LineCollection(edges, colors='k', linewidths=1)

        ax = []
        n = self.number_of_neurons()
        w = 1
        if full:
            w = 3
        proj = plt.subplot2grid((n + 1, w), (0, 0), rowspan=n, colspan=1)
        colors = cm.rainbow(np.linspace(0, 1, n))
        labels = []
        a = []
        for i, (neuron, col) in enumerate(zip(reversed(self.neurons), colors)):

            if full:
                ax.append((plt.subplot2grid((n, w), (i, 1), rowspan=1, colspan=1),
                           plt.subplot2grid((n, w), (i, 2), rowspan=1, colspan=1)))
            if n is not None:
                coordinates = np.array([c.midpoint().to_list() for c in neuron.compartments if not c.axon_comp and
                                    c.steps_to_root() > num])
            else:
                coordinates = np.array([c.midpoint().to_list() for c in neuron.compartments if not c.axon_comp])

            axon_coordinates = np.array([c.midpoint().to_list() for c in neuron.compartments if c.axon_comp])

            if len(coordinates) > 0:
                a.append(proj.scatter(coordinates[:, 1], coordinates[:, 2], s=10, c=col, alpha = 0.5))

            labels.append("Neuron " + str(n-i-1) + " compartment midpoints")

            # proj.scatter(axon_coordinates[:, 1], axon_coordinates[:, 2], s=30, c='k', marker='x')

            if full:
                if len(coordinates) > 0:
                    ax[i][0].scatter(coordinates[:, 0], coordinates[:, 1], s=3)
                if len(axon_coordinates) > 0:
                    ax[i][0].scatter(axon_coordinates[:, 0], axon_coordinates[:, 1], s=10, marker='x')
                ax[i][0].set_xlabel("x")
                ax[i][0].set_ylabel("y")
                ax[i][0].set_title("Neuron " + str(n - i - 1) + "x/y plane")
                if len(coordinates) > 0:
                    ax[i][1].scatter(coordinates[:, 0], coordinates[:, 2], s=3)
                if len(axon_coordinates) > 0:
                    ax[i][1].scatter(axon_coordinates[:, 0], axon_coordinates[:, 2], s=10, marker='x')
                ax[i][1].set_xlabel("x")
                ax[i][1].set_ylabel("z")
                ax[i][1].set_title("Neuron " + str(n - i - 1) + "x/z plane")

        if pixels is not None:
            colors = cm.Accent(np.linspace(0, 1, n))
            for i,neuron in enumerate(self.neurons):
                pixels_for_neuron = [p for p,c in pixel_mapping if c.neuron is neuron]
                pixel_x = [p.x for p in pixels_for_neuron]
                pixel_y = [p.y for p in pixels_for_neuron]
                a.append(proj.scatter(pixel_x, pixel_y, s=15, marker='s', c=colors[i], edgecolors
                = 'k', alpha = 0.8))
                labels.append("Pixel locations")

        if lc is not None:
            a.append(proj.add_collection(lc))
            labels.append("Pixel-compartment mapping")

        proj.set_xlabel("y / micrometers", fontsize = 15)
        proj.set_ylabel("z / micrometers", fontsize = 15)


        f = plt.gcf()
        f.legend(reversed(a), reversed(labels), loc='lower center', ncol=4, borderaxespad=4,
                 scatterpoints=1, fontsize = 13, columnspacing = 0, handletextpad = 0.2)
        proj.set_title("Dendrite tree y/z projection", fontsize = 13)
        # proj.set_xlim([-480,480])
        # proj.set_ylim([-25, 125])

        if (not full) and lock_aspect:
            plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        return proj

    def alternative_mapping(self, width, height, projection_axis, middle_dims, tl_tr_bl_br_m, force_n=0):

        compartments = self.get_all_compartments()
        max_x = max(compartments, key=lambda x: x.midpoint().project(projection_axis).x).midpoint().project(projection_axis).x
        min_x = min(compartments, key=lambda x: x.midpoint().project(projection_axis).x).midpoint().project(projection_axis).x
        max_y = max(compartments, key=lambda x: x.midpoint().project(projection_axis).y).midpoint().project(projection_axis).y
        min_y = min(compartments, key=lambda x: x.midpoint().project(projection_axis).y).midpoint().project(projection_axis).y

        x_grid_interval = (max_x - min_x) / width
        y_grid_interval = (max_y - min_y) / height

        x_int = np.linspace(min_x + x_grid_interval / 2, max_x - x_grid_interval / 2, width)
        y_int = np.linspace(min_y + y_grid_interval / 2, max_y - y_grid_interval / 2, height)[::-1]

        indices = list(product(xrange(width),xrange(height)))
        indices.sort(key = lambda c:c[1])
        pixels = np.array([[Vector_2D(i,j) for i in x_int] for j in y_int])
        pixel_array = Pixel_Array(pixels,middle_dims[0],middle_dims[1])

        c=0
        for row in pixels:
            for pixel in row:
                pixel.idx = indices[c]
                c+=1
        # print pixels
        # print "Here1"
        # print pixel_array
        # print "Here2"
        tl_p,tr_p,bl_p,br_p,m_p = pixel_array.get_quadrants()
        # print tl_p
        # print tr_p

        quadrant_pixels = [Pixel_Array.flatten(tl_p),
                           Pixel_Array.flatten(tr_p),
                           Pixel_Array.flatten(bl_p),
                           Pixel_Array.flatten(br_p),
                           Pixel_Array.flatten(m_p)]
        mapping = {}

        for quadrant_neuron, pixels_in_q in zip(tl_tr_bl_br_m,quadrant_pixels):
            print "About to try to map ", quadrant_neuron, "to " + str(len(pixels_in_q)) + " pixels"
            # print pixels_in_q
            single_mapping = quadrant_neuron.topological_map(pixels_in_q,force_n)
            for pixel,compartment in single_mapping:
                mapping[pixel.idx] = self.get_global_idx(compartment)

        return mapping