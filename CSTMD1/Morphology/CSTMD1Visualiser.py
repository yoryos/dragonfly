import sys

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import os
from Visualiser.VisualiserComponent import VisualiserComponent

class CSTMD1Visualiser(VisualiserComponent):
    index = 0

    compartment_plots = None
    synapse_plots = None
    electrode_plot = None

    # Synapse colors [Up, Down]
    synapse_colors = [(204 / 255, 229 / 255, 255 / 255, .5), (204 / 255, 255 / 255, 204 / 255, .5)]
    compartment_colors = pg.glColor(pg.mkColor('w'))  # k(0.5, 0.5, 0.5, 0.8)

    electrode_base_color = pg.glColor(pg.mkColor('b'))[:-1] + (0.8,)
    electrode_base_size = 10
    electrode_spike_color = pg.glColor(pg.mkColor('r'))[:-1] + (0.8,)
    electrode_spike_size = 15

    def __init__(self,
                 neuron_collection,
                 synapses=False,
                 plot_static_electrodes=False,
                 expand=50,
                 spikes=False,
                 enable_run=True,
                 white=False):
        """
        NeuronVisualiser constructor, sets compartment and synapse plots to empty lists

        """
        VisualiserComponent.__init__(self)
        self.enable_run = enable_run

        if enable_run:
            self.application = QtGui.QApplication([])
        self.neuron_collection = neuron_collection
        self.white = white
        self.window = QtGui.QGroupBox()
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.window.setContentsMargins(0, 0, 0, 0)
        self.window.setLayout(self.layout)
        self.name = "CSTMD1"
        self.title = QtGui.QLabel("CSTMD1: " + str(neuron_collection.number_of_neurons()) + " neurons (" +
                                  str(neuron_collection.total_compartments()) + " compartments)")

        self.title.setContentsMargins(0, 0, 0, 0)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setMinimumHeight(20)

        self.data_window = gl.GLViewWidget()
        self.data_window.pan(-650, 0, 0)
        self.data_window.setCameraPosition(distance=1500)
        self.data_window.setWindowTitle('CSTMD1 Visualiser')
        self.data_window.addItem(gl.GLGridItem())

        self.layout.addWidget(self.title)
        self.layout.addWidget(self.data_window, 1)
        self.layout.setStretch(1, 2)

        if self.white:
            self.data_window.setBackgroundColor('w')  # (255,255,255)
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            self.compartment_colors = []
            self.compartment_colors = pg.glColor(pg.mkColor('k'))

        if not plot_static_electrodes:
            self.electrode_base_size = 0

        self.__load_compartments(neuron_collection)

        if synapses:
            self.__load_synapses(neuron_collection)

        if spikes or plot_static_electrodes:
            self.__load_electrodes(neuron_collection, spikes)

        if spikes:
            self.n_steps = self.electrode_spikes[0].shape[0]

        if expand != 0:
            self.expand(expand, synapses, spikes or plot_static_electrodes)

        self.__plotAxis()

        if synapses:
            self.synapse_plots = []
            self.__plot_synapses()
            if self.white:
                for p in self.synapse_plots:
                    p.setGLOptions('translucent')

        if spikes or plot_static_electrodes:
            self.electrode_plot = None
            self.__plot_electrodes()
            if self.white:
                self.electrode_plot.setGLOptions('translucent')

        self.compartment_plots = []
        self.__plot_compartments()
        if white:
            for p in self.compartment_plots:
                p.setGLOptions('translucent')

    def __load_synapses(self, neuron_collection):
        """
        Load synapses

        Args:
            synapses (List[(int,ndarray,int,ndarray)]): synapse coordinates
        Notes:
            synapses should be in the format
                [(idx1, [midpoint coordinates 1], idx2, [midpoint coordinates 2]) ...]

        """

        self.synapse_coordinates = []
        if neuron_collection.synapse_groups is not None:
            for (i, j, synapseGroup) in neuron_collection.synapse_groups:
                synapsesUp = []
                synapsesDown = []
                if neuron_collection.neurons[i].soma.start.z > neuron_collection.neurons[j].soma.start.z:
                    for synapse in synapseGroup:
                        pre, post = synapse
                        synapsesDown.append(post.midpoint().to_list() + pre.midpoint().to_list())
                else:
                    for synapse in synapseGroup:
                        pre, post = synapse
                        synapsesUp.append(pre.midpoint().to_list() + post.midpoint().to_list())

                self.synapse_coordinates.append((i, np.array(synapsesUp), j, np.array(synapsesDown)))

    def __load_compartments(self, neuron_collection):
        """
        Load compartments

        Args:
            compartments (List[(int,ndarray)]): compartments

        Notes:
            compartments should be in the format:
            ([(neuron.idx, neuron_coordinates)]) where neuron_coordinates is in the format:
            [start.x, start.y, start.z, end.x, end.y, end.z]
        """

        self.compartments = []
        for neuron in neuron_collection.neurons:
            ncoords = []
            for compartment in neuron.compartments:
                ncoords.append(compartment.start.to_list() + compartment.end.to_list())
            self.compartments.append((neuron.idx, np.array(ncoords)))

    def __load_electrodes(self, neuron_collection, spikes=False):
        """

        Args:
            neuron_collection ():

        Returns:

        """

        assert hasattr(neuron_collection, "electrodes"), "No electrodes to load"

        electrode_compartments = neuron_collection.map_electrodes_to_compartments()

        self.electrodes = []

        if spikes:
            self.electrode_spikes = []

        for compartment in electrode_compartments:
            self.electrodes.append((compartment.neuron.idx, compartment.midpoint().to_list()))
            if spikes:
                assert hasattr(compartment, "spike_record"), "Spike record not found for electrode " + str(
                        compartment) + str(compartment.neuron)
                self.electrode_spikes.append(compartment.spike_record)

        if spikes:
            self.electrode_spikes = np.array(self.electrode_spikes)
            self.electrode_color = np.zeros((len(self.electrodes), 4))
            self.electrode_size = np.zeros(len(self.electrodes))

    def __plotAxis(self):
        """
        Plot Axis
        """
        axes = gl.GLAxisItem(QtGui.QVector3D(100, 100, 100))
        self.data_window.addItem(axes)

    def __plot_synapses(self):
        """
        Plot synapses
        """

        for i, (_, s1, _, s2) in enumerate(self.synapse_coordinates):
            k = i * 2

            if (s1.size != 0):
                con_shape = s1.shape
                new_plt = gl.GLLinePlotItem(pos=s1.reshape((con_shape[0] * 2,con_shape[1] / 2)),
                                                            color=self.synapse_colors[0],
                                                            mode='lines')
                self.synapse_plots.insert(k,new_plt)

                self.data_window.addItem(new_plt)
            k += 1

            if s2.size != 0:
                con_shape = s2.shape
                new_plt = gl.GLLinePlotItem(pos=s2.reshape((con_shape[0] * 2,con_shape[1] / 2)),
                                                               color=self.synapse_colors[1],
                                                               mode='lines')
                self.synapse_plots.insert(k, new_plt)

                self.data_window.addItem(new_plt)

    def __plot_compartments(self):
        """
        Plot compartments
        """
        for i, (_, comps) in enumerate(self.compartments):
            con_shape = comps.shape

            self.compartment_plots.insert(i, gl.GLLinePlotItem(pos=comps.reshape((con_shape[0] * 2,
                                                                                  con_shape[1] / 2)),
                                                               color=self.compartment_colors,
                                                               mode='lines'))


            self.data_window.addItem(self.compartment_plots[i])

    def __plot_electrodes(self):
        """

        """

        electrode_coordinates = [xyz for _, xyz in self.electrodes]

        self.electrode_plot = gl.GLScatterPlotItem(pos=np.array(electrode_coordinates),
                                                   color=self.electrode_base_color,
                                                   size=self.electrode_base_size)
        self.data_window.addItem(self.electrode_plot)

    def update(self):
        """
        Step the plot through either the voltage or spike timeseries
        """

        self.electrode_color[:] = self.electrode_base_color
        self.electrode_color[self.electrode_spikes[:, self.index]] = self.electrode_spike_color

        self.electrode_size[:] = self.electrode_base_size
        self.electrode_size[self.electrode_spikes[:, self.index]] = self.electrode_spike_size

        self.electrode_plot.setData(color=self.electrode_color, size=self.electrode_size)

        self.index += 1

        self.index = self.index % self.n_steps
        return True

    def expand(self, gap, synapses, electrodes=False):
        """
        Expand the gap between the neurons

        Args:
            gap (float): the size of the expansion
            synapses (bool): expand synapses
        """
        print "Expanding neurons with gap " + str(gap)
        soma_z = list(set([c.soma.start.z for c in self.neuron_collection.neurons]))
        soma_z.sort()
        print soma_z
        for i, comps in self.compartments:
            print "Expanding neuron " + str(i)
            ammount = soma_z.index(self.neuron_collection.neurons[i].soma.start.z)
            comps[:, np.array([False, False, True, False, False, True])] += gap * ammount

        if synapses:
            print "Expanding " + str(len(self.synapse_coordinates)) + " synapses"

            for (i1, s1, i2, s2) in self.synapse_coordinates:
                amount_1 = soma_z.index(self.neuron_collection.neurons[i1].soma.start.z)
                amount_2 = soma_z.index(self.neuron_collection.neurons[i2].soma.start.z)
                if s1.size != 0:
                    s1[:, np.array([False, False, False, False, False, True])] += amount_2 * gap
                    s1[:, np.array([False, False, True, False, False, False])] += amount_1 * gap
                if s2.size != 0:
                    s2[:, np.array([False, False, True, False, False, False])] += amount_2 * gap
                    s2[:, np.array([False, False, False, False, False, True])] += amount_1* gap

        if electrodes:
            print "Expanding Electrodes"
            for neuron_id, electrode_coordinates in self.electrodes:
                ammount = soma_z.index(self.neuron_collection.neurons[neuron_id].soma.start.z)
                electrode_coordinates[2] += ammount * gap

        print "Finished Expanded"

    def save(self, name = None, dir = ".", index = None, fmt = ".jpeg"):

        if name is None:
            name = self.name
        if index is not None:
            name += ("_" + str(index))

        name += fmt
        path = os.path.join(dir,name)

        self.data_window.grabFrameBuffer().save(path, quality=100)
        print "Saved to " + path
        return 1

    def reset_to_start(self):
        self.index = 0

    def run(self, animate=False, time_step=50):

        if not self.enable_run:
            print "Cannot run visualiser as has not been setup to run in isolation"
            return False

        if animate:
            timer = QtCore.QTimer()
            timer.timeout.connect(self.update)
            timer.start(time_step)

        print "Starting visualisation"
        self.save_button = QtGui.QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.layout.addWidget(self.save_button)

        self.window.showMaximized()
        self.window.show()
        sys.exit(self.application.exec_())

    @property
    def steps(self):
        return self.n_steps