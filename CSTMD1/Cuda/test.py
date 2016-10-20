import cstmd1Cython
import numpy as np
import numpy.testing as npt

simple_morphology = 'data/identity.dat'
simple_synapses = 'data/synExample.dat'
simple_estmd1 = 'data/estmd1SimpleStimulus.dat'
simple_electrodes = 'data/simple_electrodes.dat'

complex_morpology = '/vol/project/2015/530/g1553011/DATA/CSTMD1/LargeNeuron/neuron0_1_electrical.dat'
complex_synapses =  '/vol/project/2015/530/g1553011/DATA/CSTMD1/LargeNeuron/neuron0_1_synapses.dat'


# synapses = np.array([[0,3]], dtype=np.int32)


def load_config_file(filename,type=np.int32,flatten=True):
    """

    Load a config file and ensure correct type
    Default is 32 bit int

    """
    data = np.loadtxt(filename)
    data = np.array(data, dtype=type)
    return data

#morph = load_config_file(simple_morphology)
#synapses = np.array([[0,3]]).astype(np.int32)
estmd = load_config_file(simple_estmd1,np.float32)
electrodes = load_config_file(simple_electrodes,np.int32,True)

morph = load_config_file(simple_morphology)
synapses = load_config_file(simple_synapses, np.int32)


print 'Python'
print '#############'
print 'morph :'
print morph
print 'synapses : '
print synapses
print 'estmd : '
print '  idx: ', estmd[:,0].astype(np.int32)
print '  mag: ', estmd[:,1].astype(np.float32)
print 'electrodes : '
print '  idx:', electrodes

# print 'Loading currents'
# sim.load_estmd_currents(estmd)
print
print 'CUDA:'
print '#############'
sim = cstmd1Cython.Cstmd1Sim(morph, 0.025, 1,100)
#LAST ARGUMENT IN CONSTRUCTOR IS THE DUBUG LEVEL 3 = verbose 2 = medium 1 = simple otherwise none
sim.__load_synapses(synapses)
sim.load_electrodes(electrodes)
for i in xrange(10):
    sim.load_estmd_currents(estmd[:,0].astype(np.int32), estmd[:,1].astype(np.float32))
    print sim.run(100+i*10)
