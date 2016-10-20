import getpass
import os
from RLLoop import RL_Test_Loop

cstmd_buffer_size = 100
run_time = 2000
run_until = RL_Test_Loop.stage_RL


o = {"number_of_electrodes" : 1000, 'soma_electrodes' : False}

dragon = RL_Test_Loop(dt = 1.0,
                        cstmd_buffer_size=cstmd_buffer_size,
                        environment_config="Integration/config/Environment_test_5.ini",
                        estmd_config="Integration/config/ESTMD_norm.ini",
                        dragonfly_config="Integration/config/Dragonfly_test_1.ini",
                        cstmd_config="Integration/config/CSTMD1_norm.ini",
                        stdp_config ="Integration/config/STDP_test_1.ini",
                        run_until=run_until,
                        overwrite=o,
                        pyplot=True,
                        webcam=False)

if getpass.getuser() not in []:
    response = raw_input("Press any key to continue")

dragon.debug = True

for i in xrange(run_time):
    if not dragon.step():
        print "Breaking loop"
        break

dragon.data_dump(Environment=True, RL=True)
dragon.time_analysis()
