from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os


def calculate_spiking_rate_exp(data, dt, low_pass=True, tau=20.0):
    time_steps, noElectrodes = data.shape
    spike_rate_data = np.zeros((time_steps, noElectrodes))
    time = np.arange(time_steps) * dt

    for i in xrange(noElectrodes):
        spike_rate_data[:, i] = rate_filter(data[:, i], dt, low_pass=low_pass, tau=tau)

    return spike_rate_data, time


def rate_filter(data, dt, low_pass=False, tau=30.0):
    m = 8 * tau / dt
    r = np.linspace(-(m - 1), 0, m)
    b = np.exp(r / tau)

    b_reversed = b[::-1]
    out = signal.lfilter(b_reversed, [1.0], data) * 1000
    normalising = []
    for i in xrange(len(out)):
        l = min(i, int(m)) + 1
        out[i] /= b_reversed[:l].sum()
        normalising.append(b_reversed[:l].sum())
    if low_pass:
        b, a = signal.butter(2, dt / 6, 'low')
        return signal.lfilter(b, a, out)
    else:
        return out


number_of_neurons = 5
morphology_path = "../DATA/"
morphology_prefix = "cstmd1_"

neuron_collection = NeuronCollection()
neuron_collection.debug = True
for i in xrange(number_of_neurons):
        mcn = MultiCompartmentalNeuron()
        mcn.construct_from_SWC(morphology_path + morphology_prefix + str(i) + '.swc', [-10, 0, 0], 9)
        mcn.homogenise_lengths(0.1)
        neuron_collection.add_neuron(mcn)

neuron_collection.cstmd1_sim_get_electrodes(True, number=neuron_collection.total_compartments())
neuron_collection.load_spikes_from_file("Videos/cps15_2016-04-12_23:23:59_spikes.dat")
offsets = neuron_collection.collection_compartment_Idx_offset()

def plot_consol(f_name, start_end_pairs, label):
    data = np.loadtxt(os.path.dirname(__file__) + f_name)
    neuron_spikes_count = np.zeros([data.shape[0], len(start_end_pairs)])
    plt.figure(1)
    for n, (i, j) in enumerate(start_end_pairs):
        neuron_spikes_count[:, n] = np.sum(data[:, i:j], axis=1)
    for i in xrange(neuron_spikes_count.shape[1]):
        plt.subplot(neuron_spikes_count.shape[1], 1, i + 1)
        plt.plot(neuron_spikes_count[:, i])
    rate_w, time_w = calculate_spiking_rate_exp(neuron_spikes_count, 1.0)
    f = plt.figure(2)
    a = None
    for i in xrange(neuron_spikes_count.shape[1]):
        a = plt.subplot(neuron_spikes_count.shape[1] + 1, 1, i + 1)
        if i == 0:
            plt.title("Total Neuron Firing Rate")
        plt.plot(time_w, rate_w[:, i]/1000.0, label = label)
        plt.xlim([0, 600])
        plt.ylim([0, 100])
        if i < neuron_spikes_count.shape[1] - 1:
            plt.gca().set_xticklabels([])
        plt.ylabel("N" + str(neuron_spikes_count.shape[1] - i - 1) +" /kHz")

    return f, a

start_end_pairs = [(offsets[i], offsets[i + 1]) for i in xrange(len(offsets) - 1)]
f,a1 = plot_consol("/../DATA/cstmd_spikes_tb_0.dat", start_end_pairs, "No synapses")
f, a2 = plot_consol("/../DATA/cstmd_spikes_tb_2000_strong_noise.dat", start_end_pairs, "2000 synapses")

h,l = a2.get_legend_handles_labels()

plt.gcf().legend(h, l, loc='lower center', ncol=2, borderaxespad=2)
plt.xlabel("Time /ms")
plt.show()
