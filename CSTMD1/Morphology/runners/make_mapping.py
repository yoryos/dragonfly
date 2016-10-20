from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
import numpy as np
import os
number_of_neurons = 5
morphology_path = "../DATA/"
morphology_prefix = "cstmd1_"

neuron_collection = NeuronCollection()
neuron_collection.debug = True

for i in xrange(number_of_neurons):
    mcn = MultiCompartmentalNeuron()
    # Need to do error checking to see if file could be opened correctly
    mcn.construct_from_SWC(morphology_path + morphology_prefix + str(i) + '.swc', [-10, 0, 0], 9)
    mcn.homogenise_lengths(offset=0.1)
    neuron_collection.add_neuron(mcn)

compartments = neuron_collection.get_all_compartments(include_axon=True)

run_id = "test"
estmd_dim = (64,48)
estmd_mapping = neuron_collection.topological_mapping(48,64,plot = True, n = 110,randomize_order=True)
mapping = np.array([(x, y, v) for (x, y), v in sorted(estmd_mapping.iteritems())])
new_mapping = run_id + "_topological_height:" + str(estmd_dim[1]) + "x_width:" + str(estmd_dim[0]) + "_estmd_mapping.dat"
np.savetxt(os.path.join(".", new_mapping), mapping, "%i")