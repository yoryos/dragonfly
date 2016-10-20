import numpy as np
import pylab as plt
import math
from Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from Morphology.NeuronCollection import NeuronCollection

alpha_n = np.vectorize(lambda v: 0.01 * (-v + 10) / (np.exp((-v + 10) / 10) - 1) if v != 10 else 0.01)
beta_n = lambda v: 0.125 * np.exp(-v / 80)
n_inf = lambda v: alpha_n(v) / (alpha_n(v) + beta_n(v))

alpha_m = np.vectorize(lambda v: 0.1 * (-v + 25) / (np.exp((-v + 25) / 10) - 1) if v != 25 else 0.1)
beta_m = lambda v: 4 * np.exp(-v / 18)
m_inf = lambda v: alpha_m(v) / (alpha_m(v) + beta_m(v))

alpha_h = lambda v: 0.07 * np.exp(-v / 20)
beta_h = lambda v: 1 / (np.exp((-v + 30) / 10) + 1)
h_inf = lambda v: alpha_h(v) / (alpha_h(v) + beta_h(v))


class CSTMD1_Python_Sim:

    V_rest = np.float32(0.0)
    Cm = np.float32(1.0)
    gbar_Na = np.float32(120.0)
    gbar_K = np.float32(36.0)
    gbar_l = np.float32(0.3)
    E_Na = np.float32(115.0)
    E_K = np.float32(-12.0)
    E_l = np.float32(10.613)

    S = np.float32(7.0)
    RA = np.float32(0.1)
    r = np.float32(2e-4)
    l = np.float32(0.00001)
    Ra = np.float32((RA * l) / (3.14 * r * r))

    tau_gaba = np.float32(100.0)

    g_max = np.float32(0.5)

    E_gaba = np.float32(10.0)
    gain = np.float32(10.0)#8.0

    THRESHOLD = np.float32(30.0)

    def __init__(self, morphology_data, n_compartments, dt, T_max):

        self.index = 0
        self.morphology_data = morphology_data
        self.n_compartments = n_compartments

        self.dt = dt
        self.total_steps = (int)(T_max / dt + 1)

        self.voltage = np.zeros([self.total_steps, n_compartments], dtype=np.float32)
        self.voltage[self.index, :] = self.V_rest
        self.m = np.zeros([self.total_steps, n_compartments], dtype=np.float32)
        self.m[self.index, :] = m_inf(self.V_rest)
        self.n = np.zeros([self.total_steps, n_compartments], dtype=np.float32)
        self.n[self.index, :] = n_inf(self.V_rest)
        self.h = np.zeros([self.total_steps, n_compartments], dtype=np.float32)
        self.h[self.index, :] = h_inf(self.V_rest)

        self.refactor = np.zeros(n_compartments, dtype=bool)
        self.stimulus_current = np.zeros([n_compartments], dtype=np.float32)
        self.spike_record = np.zeros([self.total_steps, n_compartments], dtype=bool)
        self.synapses = False


    def load_electrodes(self, electrodes):
        self.electrodes = electrodes


    def load_synapses(self, synapses):

        self.n_synapses = synapses.shape[0]

        self.synapse_pre_idx = synapses[:, 0]
        self.synapse_post_idx = synapses[:, 1]

        self.gaba_conductance = np.zeros([self.total_steps, self.n_synapses], dtype=np.float32)
        self.post_synaptic_currents = np.zeros(self.n_synapses, dtype=np.float32)

        self.synapse_post_idx_unique = np.unique(self.synapse_post_idx)
        self.n_unique_post = len(self.synapse_post_idx_unique)
        self.post_synaptic_currents_unique = np.zeros(self.n_unique_post, dtype=np.float32)

        self.unique_mapping = np.zeros([self.n_unique_post, self.n_synapses])
        for (i, post_idx_unique) in enumerate(self.synapse_post_idx_unique):
            for (j, post_idx) in enumerate(self.synapse_post_idx):
                if post_idx == post_idx_unique:
                    self.unique_mapping[i, j] += 1

        self.post_synaptic_currents_unique = np.zeros(self.n_unique_post, dtype=np.float32)
        self.synapses = True

    def load_estmd_current(self, estmd_current):
        for (index, current) in estmd_current:
            self.stimulus_current[index] = current

    def step(self, step, estmd_input):
        estmd_input_compartment = self.map_estmd(estmd_input)

        self.load_estmd_current(estmd_input_compartment)
        self.start = self.index

        for i in xrange(int(step / self.dt)):
            print "Internal Step", i
            self.hodgkin_huxley()
            self.propogate_current()

            if self.synapses:
                self.synaptic_current()
            self.estmd_current()

            if self.synapses:
                self.decay_conductances()
                self.reset_conductances()
            self.check_spike()

            #print self.voltage[self.index, :]

            self.index += 1

        self.stimulus_current[:] = 0

        return self.spike_record[self.start:self.index, self.electrodes], self.voltage[self.start:self.index, :]

    def check_spike(self):

        self.spike_record[self.index + 1, np.logical_and(self.voltage[self.index + 1, :] > self.THRESHOLD,
                                                         np.logical_not(self.refactor))] = True

        self.refactor[:] = True
        self.refactor[self.voltage[self.index + 1, :] <= self.THRESHOLD] = False

    def synaptic_current(self):

        self.post_synaptic_currents = np.multiply(self.gaba_conductance[self.index,:], -self.voltage[self.index,
                                                                                                    self.synapse_post_idx] +self.E_gaba)

        self.post_synaptic_currents_unique = self.unique_mapping.dot(self.post_synaptic_currents)
        self.voltage[self.index + 1, self.synapse_post_idx_unique] += self.dt / self.Cm * \
                                                                      self.post_synaptic_currents_unique

    def hodgkin_huxley(self):

        self.n[self.index + 1, :] = self.n[self.index, :] + \
                                    self.dt * ((alpha_n(self.voltage[self.index, :]) * \
                                                (1.0 - self.n[self.index, :])) - beta_n(
                                            self.voltage[self.index, :]) * self.n[self.index, :])
        self.m[self.index + 1, :] = self.m[self.index, :] + \
                                    self.dt * ((alpha_m(self.voltage[self.index, :]) * \
                                                (1.0 - self.m[self.index, :])) - beta_m(
                                            self.voltage[self.index, :]) * self.m[self.index, :])
        self.h[self.index + 1, :] = self.h[self.index, :] + \
                                    self.dt * ((alpha_h(self.voltage[self.index, :]) * \
                                                (1.0 - self.h[self.index, :])) - beta_h(
                                            self.voltage[self.index, :]) * self.h[self.index, :]);

        g_Na = self.gbar_Na * (self.m[self.index, :] ** 3) * self.h[self.index, :]
        g_K = self.gbar_K * (self.n[self.index, :] ** 4)
        g_l = self.gbar_l

        dV = -(g_Na * (self.voltage[self.index, :] - self.E_Na) + g_K * (self.voltage[self.index] - self.E_K) + g_l * (
            self.voltage[self.index, :] - self.E_l))

        self.voltage[self.index + 1, :] = self.voltage[self.index, :] + self.dt * dV / self.Cm;

    def estmd_current(self):

        self.voltage[self.index + 1, :] += self.dt / self.Cm * self.gain * self.stimulus_current

    def reset_conductances(self):

        pre_synaptic_spike = self.spike_record[self.index, self.synapse_pre_idx]
        #print pre_synaptic_spike
        self.gaba_conductance[self.index + 1, pre_synaptic_spike] = self.g_max
        #print self.gaba_conductance[self.index + 1,:]

    def decay_conductances(self):

        self.gaba_conductance[self.index+1,:] = self.gaba_conductance[self.index,:] *(1- self.dt / self.tau_gaba)

    def propogate_current(self):

        self.voltage[self.index + 1, :] += - self.dt / (self.Cm * self.Ra) * self.morphology_data.dot(self.voltage[
                                                                                                  self.index, :])

    def map_estmd(self, estmd_input):

        return np.array([(self.estmd_mapping[(x, y)], stimulus) for y, x, stimulus in estmd_input])


    def load_estmd_mapping(self, map):
        self.estmd_mapping = map
number_of_neurons = 5
morphology_path = "Morphology/DATA/"
morphology_prefix = "cstmd1_"
synapses_f_name = "1000.dat"

neuron_collection = NeuronCollection()
neuron_collection.debug = True

for i in xrange(number_of_neurons):
    mcn = MultiCompartmentalNeuron()
    # Need to do error checking to see if file could be opened correctly
    mcn.construct_from_SWC(morphology_path + morphology_prefix + str(i) + '.swc', [-10, 0, 0], 9)
    mcn.homogenise_lengths(offset=0.1)
    neuron_collection.add_neuron(mcn)
print "Compartments",neuron_collection.total_compartments()
# neuron_collection.import_synapses_from_file(morphology_path + synapses_f_name)
# neuron_collection.import_estmd1_mapping_from_file(morphology_path +
#                                                   "cps15_2016-04-11_21:58:31_topological_height:48x_width:64_estmd_mapping.dat",(64,48))

glob_dt = 1
dt = float(glob_dt)/40
t_max = 100
n_compartments = neuron_collection.total_compartments()
sim = CSTMD1_Python_Sim(neuron_collection.cstmd1_sim_get_electical_connections(),
                        neuron_collection.total_compartments(), dt, t_max)
# sim.load_synapses(neuron_collection.cstmd1_sim_get_synapses())
sim.load_electrodes(neuron_collection.cstmd1_sim_get_electrodes(True, random=False, number=neuron_collection.total_compartments()))
# sim.load_estmd_mapping(neuron_collection.cstmd1_sim_get_estmd1_mapping(64,47,False))
sim.estmd_mapping = {(0,1):2,(1,0):3,(2,4):4}

import time
glob_start = time.time()

for i in xrange(t_max):
    start = time.time()
    print "Global step", i
    if i < t_max / 4:
        sim.step(glob_dt,[(0,1,6),(1,0,4)])
    else:
        sim.step(glob_dt,[(2,4,6)])
    step = time.time() - start
    print step
glob_time = time.time()-glob_start
print glob_time

#
# if np.isnan(sim.voltage).any():
#     print "Failed"
#
# plot_all = True
#
# if plot_all:
#     k = 5
# else:
#     k = 1
#
#
# time = np.linspace(0,t_max,int(t_max/dt) + 1)
# print n_compartments
#
# for i in xrange(n_compartments):
#     if i == 9:
#         ax = plt.subplot(n_compartments,k,5*k + 1)
#         ax.plot(time, sim.voltage[:,i])
#     ax = plt.subplot(n_compartments,k,i*k + 1)
#     ax.plot(time, sim.voltage[:,i])
#     if plot_all:
#         ax = plt.subplot(n_compartments,k,i*k + 2)
#         ax.plot(time, sim.m[:,i])
#         ax = plt.subplot(n_compartments,k,i*k + 3)
#         ax.plot(time, sim.n[:,i])
#         ax = plt.subplot(n_compartments,k,i*k + 4)
#         ax.plot(time, sim.h[:,i])
# if sim.synapses and plot_all:
#     for c,i in enumerate(sim.synapse_pre_idx):
#         ax = plt.subplot(n_compartments,k,i*k + 5)
#         print time.shape
#         print sim.gaba_conductance.shape
#         ax.plot(time, sim.gaba_conductance[:,c])
# plt.show()
#

# # Create remote process with a plot window
# import pyqtgraph as pg
# import numpy as np
# import pyqtgraph.multiprocess as mp
# # pg.mkQApp()
# proc = mp.QtProcess()
#
# rpg = proc._import('pyqtgraph')
#
# plotwin = rpg.plot()
#
# curve = plotwin.plot(pen='y')
# plotwin.setXRange(0, int(t_max / dt + 1))
#
# data = proc.transfer([])
#
# for i in xrange(t_max):
#     _, v = sim.step(glob_dt, [[0, 1]])
#
#     volt = list(v[:, 0])
#     data.extend(volt, _callSync='off')
#     curve.setData(y=data, _callSync='off')
# proc.close()
#
# # plt.show()

