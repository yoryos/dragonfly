"""
__author__:cps15
Script demonstrates how to build a neuron collection
"""

from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
import os

number_of_neurons = 5
#Must point to valid morphology directory containing .swc file
morphology_path = "CSTMD1/Morphology/DATA/"
morphology_prefix = "cstmd1_"

neuron_collection = NeuronCollection()
neuron_collection.debug = True

for i in xrange(number_of_neurons):
    mcn = MultiCompartmentalNeuron()
    mcn.construct_from_SWC(os.path.join(morphology_path,morphology_prefix + str(i) + '.swc'), [-10, 0, 0], 9)
    mcn.homogenise_lengths(offset=0.1)
    neuron_collection.add_neuron(mcn)