from CSTMD1.Cstmd1 import Cstmd1
from Helper.Configurer import  Configurer
import numpy as np
import pylab as plt

dt = 1.0
cstmd_speed_up = 20
run_time = 200

config_file_path="Integration/config/CSTMD1_test_1.ini"
config = Configurer(config_file_path)


cstmd = Cstmd1(global_dt = dt,
               speed_up=cstmd_speed_up,
               estmd_dim=[2, 2],
               buffer_size=1,
               debug_mode=0,
               sim_parameters=config.config_section_map("CSTMD1_Simulator"),
               morph_param=config.config_section_map("Morphology_Homogenised_Top"),
               run_id = "test")

#cstmd.simulator.enable_randomise_currents(0.0, 0.15)

response = raw_input("Press any key to continue")
# cstmd.verbose_debug = True

status = True
reached = run_time
for i in xrange(run_time):
    print "Outer step", i
    # response = raw_input("Press any key to continue")
    # cstmd.step(dt)
    # if i < run_time / 4:
    status, spike_train = cstmd.step(dt, [[0, 0, 1], [0, 1, 1]])
    print spike_train
    # else:
    #     status, spike_train = cstmd.step(dt, [[0, 1, 2]])

    if not status:
        print 'CSTMD1 step failed, exiting loop'
        break

n_compartments = cstmd.neuron_collection.total_compartments()

# cstmd.print_voltages()
# cstmd.print_spikes()

volt = cstmd.get_voltages()
check = np.isnan(volt)
np.set_printoptions(threshold=100000, edgeitems=100)
if check.any():
    print "Failed, nans found."

    # print np.transpose(np.nonzero(check))

else:
    print "No nans found"

cstmd.save_spikes()
cstmd.plot_firing_rate(window_size_ms=40)
cstmd.save_spike_rate_data()
cstmd.plot_spikes()
# print volt.shape
# time = np.linspace(0,run_time,volt.shape[0])
# print time.shape
# for i in xrange(n_compartments):
#     ax = plt.subplot(n_compartments,1,i + 1)
#     ax.plot(time, volt[:,i])
#     # ax.set_ylim(-30,130)
# plt.show()
