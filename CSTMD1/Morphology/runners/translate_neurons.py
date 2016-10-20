from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
from Helper.Vectors import Vector_3D
import numpy as np
import sys
import os
import copy
from Visualiser.VideoVisualiser import VideoVisualiser
from pyqtgraph.Qt import QtCore, QtGui
from Visualiser.DragonflyVisualiser import DragonflyVisualiser
from CSTMD1.Morphology.CSTMD1Visualiser import CSTMD1Visualiser


number_of_neurons = 5
morphology_path = "CSTMD1/Morphology/DATA/"
morphology_prefix = "cstmd1_"

neuron_collection = NeuronCollection()
neuron_collection.debug = True

shifts = [None,None,Vector_3D(0,-350,-50),Vector_3D(0,-350,-50), Vector_3D(0,-175,-75-12.5)]
shifts = [None,None,Vector_3D(0,-700,-50),Vector_3D(0,-700,-50), Vector_3D(0,-350,-75-12.5)]
for (i,shift) in zip(*(xrange(number_of_neurons),shifts)):
    mcn = MultiCompartmentalNeuron()
    # Need to do error checking to see if file could be opened correctly
    mcn.construct_from_SWC(morphology_path + morphology_prefix + str(i) + '.swc', [-10, 0, 0], 9)
    mcn.homogenise_lengths(offset=0.1)
    if shift is not None:
        mcn.shift(shift)
    neuron_collection.add_neuron(mcn)

direc = "CSTMD1/Morphology/runners"
cstmd_path = os.path.join(direc,"cstmd_spikes.dat")
synapses = os.path.join(direc,"synapses1000.dat")
neuron_collection.cstmd1_sim_get_electrodes(True,number=neuron_collection.total_compartments())
neuron_collection.load_spikes_from_file(cstmd_path)
neuron_collection.import_synapses_from_file(synapses)

app = QtGui.QApplication([])
cstmd = CSTMD1Visualiser(neuron_collection,True,False,200,True,False)
# env_path = os.path.join(direc,"environment_output.avi")
# env = VideoVisualiser("Environment",env_path)
# estmd_path = os.path.join(direc,"estmd_output.avi")
# # estmd = VideoVisualiser("ESTMD", estmd_path)
# c2 = CSTMD1Visualiser(neuron_collection,True,False,100,True,False)
# c3 = CSTMD1Visualiser(neuron_collection,True,False,100,True,False)
# c4 = CSTMD1Visualiser(neuron_collection,True,False,100,True,False)

dv = DragonflyVisualiser(1.0,[cstmd],num_cols=2)
dv.show()
dv.run()

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_()

#
# neuron_locs = [neuron_collection.neurons[3],
#                neuron_collection.neurons[1],
#                neuron_collection.neurons[2],
#                neuron_collection.neurons[0],
#                neuron_collection.neurons[4]]

# estmd_mapping = neuron_collection.alternative_mapping(64,48,"x",(36,28),neuron_locs,100)
# mapping = np.array([(x, y, v) for (x, y), v in sorted(estmd_mapping.iteritems())])
# np.savetxt("mapping_force_100.dat",mapping,"%i")
# neuron_collection.plot_estmd_mapping_from_file(64,48,"mapping_force_100.dat")
#
# synapses = neuron_collection.cstmd1_sim_get_synapses(True,1000,20)
# print neuron_collection.cstmd1_sim_get_synapses()
# print neuron_collection.synapse_groups
# np.savetxt("synapses1000.dat",synapses,"%i")
# neuron_collection.import_synapses_from_file("synapses600.dat")
# print neuron_collection.synapse_groups
# print neuron_collection.cstmd1_sim_get_synapses()
# neuron_collection.plot(synapses=True, expand=100)