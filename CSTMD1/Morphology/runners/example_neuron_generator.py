"""
__author__:cps15
Script demonstrates how to generate a .swc file
"""

from CSTMD1.Morphology.NeuronGenerator import NeuronGenerator

output_dir = '/vol/project/2015/530/g1553011/DATA/CSTMD1/CSTMD_Morph/',
# uses trees octave package, need to know path
trees_path = '/homes/cps15/dragonfly/CSTMD1/Morphology/trees'
number_of_neurons = 5

ng = NeuronGenerator()
ng.generate_neuron_morphologies(number_of_neurons,
                                1500,
                                file_prefix="cstmd1_",
                                directory_path = output_dir,
                                trees_path = trees_path,
                                safe = True)
