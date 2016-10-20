"""
__author__:cps15
Script demonstrates homogenising the lengths of the compartment within a neuron
"""


from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
import matplotlib.pyplot as plt
import numpy as np

#Must point to a valid .swc file
morphology_file = '../DATA/cstmd1_0.swc'

mcn = MultiCompartmentalNeuron()
mcn.construct_from_SWC(morphology_file, [0, -10, 0])

mcn.generate_radii(2, 10)

lengths1,_ = mcn.compartment_data()

m = mcn.median_length()
s = mcn.length_stdev()
mcn.homogenise_lengths(offset=0.1)

lengths2,_ = mcn.compartment_data()

ax1 = plt.subplot(211)
plt.hist(lengths1, np.linspace(0, max(lengths1), 100))
plt.subplot(212,sharex=ax1)
plt.hist(lengths2, np.linspace(0, max(lengths2), 100))
plt.show()