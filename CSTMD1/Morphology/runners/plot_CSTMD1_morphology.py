from CSTMD1.Morphology.MultiCompartmentalNeuron import MultiCompartmentalNeuron
from CSTMD1.Morphology.NeuronCollection import NeuronCollection
from CSTMD1.Morphology.CSTMD1Visualiser import CSTMD1Visualiser

number_of_neurons = 5
morphology_path = "../DATA/"
morphology_prefix = "cstmd1_"
synapses_f_name = "cps15_2016-04-11_21:58:31_100_20.0_synapses.dat"

neuron_collection = NeuronCollection()
neuron_collection.debug = True

for i in xrange(number_of_neurons):
    mcn = MultiCompartmentalNeuron()
    # Need to do error checking to see if file could be opened correctly
    mcn.construct_from_SWC(morphology_path + morphology_prefix + str(i) + '.swc', [-10, 0, 0], 9)
    mcn.homogenise_lengths(offset=0.1)
    neuron_collection.add_neuron(mcn)


# neuron_collection.plot_plan()


neuron_collection.import_synapses_from_file(morphology_path + synapses_f_name)
# neuron_collection.cstmd1_sim_get_electrodes(True, True) #random=True, number=50)
#
# neuron_collection.load_bulk_compartment_data(spikes = spikes)
# neuron_collection.import_electrodes_from_file(morphology_path + "all_electrodes.dat")
# neuron_collection.load_spikes_from_file(morphology_path + "all_spikes_100_synapses.dat")

v = CSTMD1Visualiser(neuron_collection, synapses=True, enable_run=True, expand = 100)
v.run(False)



# vis = NeuronVisualiser()
# vis.save = 5
# vis.plot_from_neuron_collection(neuron_collection, synapses = True, expand = 50, plot_static_electrodes= False,
#                                  spikes = False)

