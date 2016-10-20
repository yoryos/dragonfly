"""
__author__:cps15
Script for plotting electrode spikes using a raster plot.
"""

import matplotlib.pyplot as plt
import numpy as np


def raster(event_times_list, color='k'):
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax


def plot_spikes(n, global_dt, spike_history):
    spikes = np.array(spike_history)
    spike_time, e_idx = np.where(spikes)
    spike_time = spike_time.astype('float32')
    spike_time *= global_dt
    spike_time_pair = zip(e_idx, spike_time)
    spike_time_pair.sort()
    spike_time_pair = np.array(spike_time_pair)
    spike_time_pair = list(np.split(spike_time_pair, np.where(np.diff(spike_time_pair[:, 0]))[0] + 1))

    s = []
    for i in xrange(n):
        s1 = [t[:, 1] for t in spike_time_pair if t[0, 0] == i]
        s.append(s1)

    ax = raster(s)

    ax.set_ylabel('Electrode IDX')
    ax.set_xlabel('Time (msec)')
    ax.set_title('CSTMD Electrode Spikes for ' + str(n) + ' compartments')

    return plt.gca()

import os
spike_file = "cstmd_spikes_corners_fast.dat"
spike_file = "cstmd_spikes_corners_strong.dat"
direc = "DATA"
cstmd_path = os.path.join(direc,"cstmd_spikes.dat")
# spike_file = "cstmd_spikes_center_fast.dat"
data = np.loadtxt(cstmd_path)
ax = plot_spikes(data.shape[1], 1, data)

compartment_count = np.array([0, 1913, 1931, 1928, 1919, 1963])
cc = compartment_count.cumsum()
somas = cc[:-1]

plt.hlines(somas, 0, data.shape[0], colors='k', linestyles='dashed')
for i in xrange(len(somas)):
    plt.text(data.shape[0] + 25, (cc[i] + cc[i + 1]) / 2.0, "Neuron " + str(i),
             ha="center",
             va="center",
             rotation=270,
             size=12.5,
             bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
plt.title("CSTMD1 Compartment Spiking")
plt.show()
