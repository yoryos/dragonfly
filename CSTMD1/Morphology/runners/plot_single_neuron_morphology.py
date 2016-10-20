"""
__author__:cps15
Script to plot morphology of a single neuron
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron

morphology_path = "../DATA/"
morphology_prefix = "cstmd1_"

mcn = MultiCompartmentalNeuron()

mcn.construct_from_SWC(morphology_path + morphology_prefix + '0.swc', [-10, 0, 0], 9)
mcn.homogenise_lengths(offset=0.1)

coordinates = np.array([c.midpoint().to_list() for c in mcn.compartments if not c.axon_comp])
axon_coordinates = np.array([c.midpoint().to_list() for c in mcn.compartments if c.axon_comp])

s = 5

ax1 = plt.subplot2grid((1, 5), (0, 0), rowspan=1, colspan=2, projection='3d')
ax1.scatter(xs=coordinates[:, 0], ys=coordinates[:, 1], zs=coordinates[:, 2], c='b', marker='o', s=5)
ax1.scatter(xs=axon_coordinates[:, 0], ys=axon_coordinates[:, 1], zs=axon_coordinates[:, 2], c='r', marker='s')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.set_xlabel("x /micrometres", fontsize=15)
ax1.set_ylabel("y /micrometres", fontsize=15)
ax1.set_zlabel("z /micrometres", fontsize=15)
ax1.legend(['Dendrite Cloud', 'Axon'], loc=2)
plt.title("CSTMD1 Neuron Compartment Centres", fontsize=15)

plt.subplot2grid((1, 5), (0, 2), rowspan=1, colspan=1)
a1 = plt.scatter(coordinates[:, 1], coordinates[:, 0], s=s, c='b', alpha=0.5, marker='o')
a1_b = plt.scatter(axon_coordinates[:, 1], axon_coordinates[:, 0], s=s, c='r', alpha=0.5, marker='s')
plt.xlabel("y /micrometres")
plt.ylabel("x /micrometres")
plt.gca().invert_yaxis()

plt.subplot2grid((1, 5), (0, 3), rowspan=1, colspan=1)
a2 = plt.scatter(coordinates[:, 1], coordinates[:, 2], s=s, c='b', alpha=0.5)
a2_b = plt.scatter(axon_coordinates[:, 1], axon_coordinates[:, 2], s=s, c='r', alpha=0.5, marker='s')
plt.xlabel("y /micrometres")
plt.ylabel("z /micrometres")

plt.subplot2grid((1, 5), (0, 4), rowspan=1, colspan=1)
a3 = plt.scatter(coordinates[:, 0], coordinates[:, 2], s=s, c='b', alpha=0.5)
a3_b = plt.scatter(axon_coordinates[:, 0], axon_coordinates[:, 2], s=s, c='r', alpha=0.5, marker='s')
plt.gca().invert_yaxis()
plt.xlabel("x /micrometres")
plt.ylabel("z /micrometres")
plt.show()
