from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
from CSTMD1.Morphology.CSTMD1Visualiser import CSTMD1Visualiser
from Visualiser.VideoVisualiser import VideoVisualiser
from Visualiser.StdpVisualiser import StdpVisualiser
from Visualiser.DragonflyVisualiser import DragonflyVisualiser
import numpy as np
import os
import sys
from pyqtgraph.Qt import QtCore, QtGui

cstmd1_number_of_neurons = 5
cstmd1_morphology_folder = "CSTMD1/Morphology/DATA"
cstmd1_morphology_prefix = "cstmd1_"
cstmd1_synapse_file_name = "cps15_2016-04-11_21:58:31_100_20.0_synapses.dat"
cstmd1_spike_file = "Videos/cps15_2016-04-12_23:23:59_spikes.dat"
environment_output_file = "Videos/cps15_2016-04-12_23:23:59_env_out.avi"
estmd_output_file = "Videos/cps15_2016-04-12_23:23:59_estmdOut.avi"
environment_output_file = "Videos/environment_output.avi"
estmd_output_file = "Videos/estmd_output.avi"

neuron_collection = NeuronCollection()
for i in xrange(cstmd1_number_of_neurons):
    mcn = MultiCompartmentalNeuron()
    mcn.construct_from_SWC(os.path.join(cstmd1_morphology_folder, cstmd1_morphology_prefix + str(i) + '.swc'), [-10, 0, 0], 9)
    mcn.homogenise_lengths(0.1)
    neuron_collection.add_neuron(mcn)

neuron_collection.import_synapses_from_file(os.path.join(cstmd1_morphology_folder, cstmd1_synapse_file_name))
neuron_collection.cstmd1_sim_get_electrodes(True, number=neuron_collection.total_compartments())
neuron_collection.load_spikes_from_file(cstmd1_spike_file)

app = QtGui.QApplication([])

CSTMD1 = CSTMD1Visualiser(neuron_collection, True, False, 100, True, False, False)
Environment = VideoVisualiser("Environment", environment_output_file, True, 1)
ESTMD = VideoVisualiser("ESTMD", estmd_output_file, False, 1)

stdp_afferent_data = neuron_collection.get_spikes_from_compartments(neuron_collection.get_compartment_idxs(1000))
stdp_dummy_out_data = np.random.randint(0, 2, stdp_afferent_data.shape[0] * 4).reshape([700, 4])
STDP = StdpVisualiser(stdp_afferent_data, stdp_dummy_out_data)

vis = DragonflyVisualiser(1.0, [Environment, ESTMD, CSTMD1, STDP], num_cols=2)
vis.show()
vis.run()

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_()
