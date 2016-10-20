import numpy as np

from CSTMD1.Cstmd1 import Cstmd1
from Helper.Configurer import Configurer


def step(stddev):
    dt = 1.0
    cstmd_speed_up = 40
    run_time = 700

    config_file_path = "Integration/config/CSTMD1_test_1.ini"
    config = Configurer(config_file_path)

    cstmd = Cstmd1(global_dt=dt,
                   speed_up=cstmd_speed_up,
                   estmd_dim=[64, 48],
                   buffer_size=1,
                   debug_mode=0,
                   sim_parameters=config.config_section_map("CSTMD1_Simulator"),
                   morph_param=config.config_section_map("Morphology_Homogenised_Top"),
                   run_id="test")

    cstmd.simulator.enable_randomise_currents(0.0, stddev)

    # response = raw_input("Press any key to continue")
    cstmd.verbose_debug = False

    status = True
    reached = run_time
    spikes = 0
    for i in xrange(run_time):
        status, spike_train = cstmd.step(dt, [])
        for spike in spike_train:
            spikes += 1

        if not status:
            print 'CSTMD1 step failed, exiting loop'
            break

    volt = cstmd.get_voltages()
    check = np.isnan(volt)
    np.set_printoptions(threshold=100000, edgeitems=100)
    if check.any():
        print "Failed, nans found."
        return spikes * (-1)
    else:
        return spikes


x = np.linspace(0, 1, 100)
data = np.zeros((2, 100))
for i in xrange(len(x)):
    data[0, i] = x[i]
    data[1, i] = step(x[i])
    print
    print 'RESULT:::::'
    print data[0, i], data[1, i]
    print

np.savetxt('noise_dump.dat', data)
