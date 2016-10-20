"""
__author__:cps15
Script demonstrates how to build a neuron collection
"""

from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
import os
import numpy as np
import matplotlib.pyplot as plt
number_of_neurons = 5
#Must point to valid morphology directory containing .swc file
morphology_path = "../DATA/"
morphology_prefix = "cstmd1_"

neuron_collection = NeuronCollection()
neuron_collection.debug = True

for i in xrange(number_of_neurons):
    mcn = MultiCompartmentalNeuron()
    mcn.construct_from_SWC(os.path.join(morphology_path,morphology_prefix + str(i) + '.swc'), [-10, 0, 0], 9)
    mcn.homogenise_lengths(offset=0.1)
    neuron_collection.add_neuron(mcn)

new = np.loadtxt("../DATA/test_topological_height:48x_width:64_estmd_mapping.dat")

def get_data(data):
    distance = []
    for entry in data:
        compartment = neuron_collection.get_compartment(int(entry[2]))
        distance.append(compartment.steps_to_root())
    return np.array(distance)


n = get_data(new)

plt.figure()
plt.subplot(111)
plt.hist(n, np.linspace(0, max(n), 100))
plt.xlabel("Compartment distance from soma /steps")
plt.ylabel("Number of compartments mapped to ESTMD input")
plt.show()