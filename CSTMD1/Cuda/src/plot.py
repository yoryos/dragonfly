import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("../data/large_volt.dat", delimiter=' ')
spike = np.loadtxt("../data/large_spikes.dat", delimiter=' ')

N = data.shape[1]
dt = 0.025
T = len(data[:,0])
time  = np.arange(0,T,1) * dt
print len(time),'  ',len(data[:,0])
## plot membrane potential trace
ylabel_set = False

spikes = True
M = N
if(spikes):
    M = 2*N


for i in xrange(N):
  ax = plt.subplot(M,1,i+1)
  ax.plot(time,data[:,i])
  if i == 0:
    ax.set_title('Hodgkin-Huxley Active Compartment Example')
  if i == np.floor(N/2) and not ylabel_set:
    ax.set_ylabel('Membrane Potential (mV)')
    ylabel_set == True
  if i != N-1:
    ax.set_xticklabels([])
  ax.set_yticks([-20, 50, 120])

if(spikes):
    for i in xrange(N):
          ax = plt.subplot(M,1,N+i+1)
          ax.plot(time,spike[:,i])
          if i == np.floor(N/2) and not ylabel_set:
            ax.set_ylabel('Spike')
            ylabel_set == True
          if i != N-1:
            ax.set_xticklabels([])
          ax.set_yticks([0, 1.5])

plt.xlabel('Time (msec)')
plt.show()
