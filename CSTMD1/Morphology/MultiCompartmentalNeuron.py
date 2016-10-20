'''
MultiCompartmentalNeuron

Data structure for multi-compartmental neuron.

__author__: Dragonfly Project 2016 - Imperial College London ({anc15, cps15, dk2015, gk513, lm1015,
zl4215}@imperial.ac.uk)

'''

import matplotlib.pyplot as py
import numpy as np

from Compartment import Compartment
from Helper.Vectors import Vector_3D
from progressbar import Percentage, Bar, ETA, ProgressBar

class MultiCompartmentalNeuron(object):
    """A collection of compartments forming a tree

    Attributes:
        compartments(List[Compartment]): list of compartments that make up the MCN
        idx(int): unique index of the MCN
        soma(Optional[Compartment]): compartment which is the soma of the MCN

    """
    compartments = None
    idx = None
    soma = None

    def __init__(self, idx=None):

        """
        MultiCompartmentalNeuron constructor

        Args:
            idx (Optional[int]): unique identifier for the MCN
        """
        self.compartments = []
        self.idx = idx

    def shift(self, vector):

        for compartment in self.compartments:
            if not hasattr(compartment.start, "shifted"):
                compartment.start += vector
                compartment.start.shifted = True
            if not hasattr(compartment.end, "shifted"):
                compartment.end += vector
                compartment.end.shifted = True

        for compartment in self.compartments:
            if hasattr(compartment.start, "shifted"):
                delattr(compartment.start, "shifted")
            if hasattr(compartment.end, "shifted"):
                delattr(compartment.end, "shifted")


    def number_of_compartments(self):
        """
        Get the number of compartments in the MCN

        Returns:
            int: number of compartments in the neuron
        """
        return len(self.compartments)

    def __add_compartment(self, compartment):
        """
        Add a compartment to the neuron, sets the compartment idx to the number of neurons in the MCN

        Raises:
            AssertionError: if the compartment is already in the MCN

        Args:
            compartment (Compartment): compartment to add to the neuron
        """

        assert compartment not in self.compartments

        compartment.idx = self.number_of_compartments()
        self.compartments.append(compartment)

    def construct_from_SWC(self, path, soma_offset, axon_n = None):
        """
        Construct a neuron from a SWC data file, will connect neurons together to form the node/edge tree given

        Args:
            path (str): path to SWC file
            soma_offset (List[float]): [soma_x, soma_y, soma_z], the soma start offset relative to the origin

        Raises:
            AssertionError: if there is a problem adding parent-child relationships, i.e. the SWC file is malformed
            IOError: if path is invalid

        Example:

            >>> mcn = MultiCompartmentalNeuron()
            # Consider SWC file 'example.txt' such as:
            # inode R X Y Z D/2 idpar
            # 1 1 0.0 0.0 0.0 0.5 -1
            # 2 1 10.0 10.0 0.0 0.5 1
            # 3 1 20.0 0.0 0.0 0.5 2
            # 4 1 20.0 20.0 0.0 0.5 2
            >>> mcn.construct_from_SWC(example.txt, [-10,0,0])
            #        [-10,0,0]
            #            | (neuron0)
            #         [0,0,0]
            #            | (neuron1)
            #         [10,0,0]
            #(neuron2)|      |(neuron3)
            #     [20,0,0] [20,0,0]

        """

        sx, sy, sz = soma_offset

        try:
            data = np.loadtxt(path)
        except IOError as e:
            print 'Could not load SWC file, it may not exist ' + path
            raise IOError

        for i,node in enumerate(data):

            parent = int(node[6])

            end_coordinate = Vector_3D(node[2], node[3], node[4])

            if parent > 0:
                connected_node = data[node[6] - 1]
                start_coordinate = Vector_3D(connected_node[2], connected_node[3], connected_node[4])
            else:
                start_coordinate = Vector_3D(node[2] + sx, node[3] + sy, node[4] + sz)

            comp = Compartment(self.number_of_compartments(), start_coordinate, end_coordinate)

            comp.neuron = self

            if parent > 0:
                comp.add_parent_relationship(self.compartments[parent - 1])
            else:
                self.soma = comp

            self.__add_compartment(comp)

            if axon_n is not None and comp.steps_to_root() <= axon_n:
                comp.axon_comp = True


    def connection_matrix(self):
        """
        Find the connection weighting matrix for electrical connections

        Raises:
            AssertionError: if the tree is corrupted, i.e. a compartment listed in the tree cannot be found in the
            MCN compartment list

        Returns:
            ndarray: connection matrix for electrical connections, if MCN has n compartments, then nxn array
        """

        S = np.zeros((self.number_of_compartments(), self.number_of_compartments()), dtype=int)

        for compartment in self.compartments:
            for connected_to in [compartment.parent] + compartment.children + compartment.siblings:
                if connected_to is None:
                    continue

                assert (connected_to in self.compartments)

                S[compartment.idx, compartment.idx] += 1
                S[compartment.idx, connected_to.idx] -= 1

        return S


    def __str__(self):
        return "[idx: " + str(self.idx) + ", compartments: " + str(self.number_of_compartments()) + ", somaID: " + str(
                self.soma.idx) + ": " + str(self.soma.start) + " " + str(self.soma.end) + "]"

    def compartment_data(self):
        """
        Get compartment data

        Returns:
            ndarray[float], ndarray[float]: lengths and radii of the compartments
        """

        lengths = []
        radii = []

        for compartment in self.compartments:
            lengths.append(compartment.length())
            radii.append(compartment.radius)

        return np.array(lengths), np.array(radii)

    def plot_compartment_data(self, plot_lengths=True, plot_radii=False, block = True, normed = False):
        """
        Plot a histogram of the compartment lengths and radii

        Args:
            plot_lengths (bool): plot lengths
            plot_radii (bool): plot radii
        """
        lengths, radii = self.compartment_data()

        i, plots = 1, 0
        if plot_lengths:
            plots += 1
        if plot_radii:
            plots += 1

        py.figure()

        if plot_lengths:
            py.subplot(plots, 1, 1)
            py.hist(lengths, np.linspace(0, max(lengths), 100),normed=normed)
            py.title("Median Length = %0.2f"% self.median_length() + " Length stdev = %0.2f"%self.length_stdev())
            py.ylabel("Number of compartments")
            py.xlabel("Lengths")
            i += 1
        print i
        if plot_radii:
            py.subplot(plots, 1, i)
            py.hist(radii, np.linspace(0, max(radii), 100), normed=normed)
            py.ylabel("Number of compartments")
            py.xlabel("Radii")

        if plot_radii or plot_lengths:
            py.show(block = block)

    def generate_radii(self, weighting, soma_radius):
        """
        Generate radii for the mcn, where the radius is decreases the further away from the soma the compartment is
        location
        Args:
            weighting (float): the decay rate of the radii
            soma_radius (float): the radius of the soma

        Example:
            Given tree: c->[a,b]

            >>> generate_radii(2, 1)
            >>> c.radius
            1
            >>> b.radius
            0.5
            >>> a.radius
            0.5
        """
        self.soma.radius = float(soma_radius)

        for compartment in self.compartments:
            if compartment is not self.soma:
                compartment.radius = float(soma_radius) / (weighting * compartment.steps_to_root())

    def homogenise_lengths(self, offset = None, low = None, high = None):
        """
        Homogenise the lengths of the compartments in the MCN. Does not affect the branching nodes of the graph.
        Only compartments that can join with their child are candidates to merge. All compartments are candidates to
        split.

        Args:
            low (float): Low soft bound on compartment length
            high (float): High soft bound on compartment length
            offset (Optional[float]): if specified then low = median_length - offset * length_stdev and high =
                median_length + offset * length_stdev

        See Also:
            Compartment.can_join()
            Compartment.join_with_child()
            Compartment.split()
        """

        if offset is not None:
            median = self.median_length()
            std = self.length_stdev()
            low = median - offset * std
            high = median + offset * std

        average = (high + low) / 2

        if offset is None:
            assert low is not None and high is not None, "If offset from median is not given, bounds must be given"

        for compartment in self.compartments:
            l = compartment.length()

            #If the length of the compartment is smaller than low, and if joining with child puts the length closer
            # to the average THEN JOIN
            while compartment.can_join() and l < low and abs(l + compartment.children[0].length() - average) < abs(l-average):

                to_remove = compartment.join_with_child()
                self.compartments.remove(to_remove)
                l = compartment.length()

            #If the length of the compartment is larger than high, and if splitting the compartment in two puts the
            # length of the resulting compartments closer to the average then SPLIT
            while l > high and abs(l / 2 - average) < abs(l - average):
                new_compartment = compartment.split(self.number_of_compartments())
                new_compartment.neuron = self
                self.compartments.append(new_compartment)
                l = compartment.length()

        self.rebase_idx()

    def rebase_idx(self):
        """
        Rebase the compartments idx. Such that the index of the compartment is the position of the compartment in the
        compartment list
        """
        self.compartments.sort(key = lambda c: c.steps_to_root())

        for compartment in self.compartments:

            new_i = self.compartments.index(compartment)
            old_i = compartment.idx
            if new_i != old_i:
                compartment.idx = new_i

    def median_length(self):
        """
        Calculate the median length of the compartments

        Returns:
            float: median length
        """

        length, _ = self.compartment_data()
        return np.median(length)

    def length_stdev(self):
        """
        Calculate the standard deviation of the lengths of the compartments

        Returns:
            float: standard deviation of the lengths
        """

        length, _ = self.compartment_data()
        return length.std()


    def topological_map(self, pixels, n = 100, axis = "x"):

        i = 0
        pixel_to_compartment = []

        widgets = ["Neuron " + str(self.idx) + " ", Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA()]

        pbar = ProgressBar(widgets=widgets, maxval=(len(pixels))).start()
        steps = []
        for pixel in pbar(pixels):

            candidates = [c for c in self.compartments if (not hasattr(c, 'taken') and c.steps_to_root(True) > n)]
            j = 0
            while len(candidates) == 0:
                    i += 10
                    candidates = [c for c in self.compartments if (not hasattr(c, 'taken') and c.steps_to_root(True) > (n-i))]
                    j += 1
                    if j > 5:
                        print i, pixel, "Warning cannot find compartment"
                    if i == n:
                        print "Warning no compartments left"
                        raise Exception
            closest_compartment = min(candidates,key=lambda c: pixel.distance(c.midpoint().project(axis)))

            steps.append(closest_compartment.steps_to_root(True))
            pixel_to_compartment.append((pixel, closest_compartment))
            closest_compartment.taken = True

        pbar.finish()
        print
        print "Mean steps to root",np.mean(steps)
        print "Min steps to root",np.min(steps)
        print "Max steps to root",np.max(steps)
        return pixel_to_compartment