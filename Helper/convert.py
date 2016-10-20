import sys
import numpy as np

path = sys.argv[1]

for i in xrange(4):
    head = path + 'spikes_' + str(i+1)
    in_name = head + '.dat'
    out_name = head + '.npz'
    np.savez(out_name,spike_trains=np.loadtxt(in_name))
