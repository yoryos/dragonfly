import getpass
import os
from DragonflyBrain import DragonflyBrain

cstmd_buffer_size = 1
run_time = 1000
run_until = DragonflyBrain.stage_ALL


o = {"number_of_electrodes" : 1000, 'soma_electrodes' : False, 'estmd_gain' : 10}

dragon = DragonflyBrain(dt = 1.0,
                        cstmd_buffer_size=cstmd_buffer_size,
                        environment_config="Integration/config/Environment_test_1.ini",
                        estmd_config="Integration/config/ESTMD_norm.ini",
                        dragonfly_config="Integration/config/Dragonfly_test_1.ini",
                        cstmd_config="Integration/config/CSTMD1_norm.ini",
                        stdp_config ="Integration/config/STDP_test_1.ini",
                        run_until=run_until,
                        overwrite=o,
                        pyplot=False,
                        webcam=False,
                        cuda_device=1)


raw_input("Press any key to continue")


dragon.debug = True
dragon.run(run_time)
#
# for i in xrange(run_time):
#     if not dragon.step():
#         print "Breaking loop"
#         break
#
# dragon.data_dump(Environment=True, ESTMD=True, CSTMD=True, STDP=True, RL=True)
loop_time, env_time, estmd_time, cstmd_time, stdp_time, rl_time = dragon.time_analysis()

print "Full loop took %f seconds" % (loop_time)
print "Time in environment: ", env_time
print "Time in ESTMD: ", estmd_time
print "Time in CSTMD: ", cstmd_time
print "Time in SDTP: ", stdp_time
print "Time in RL: ", rl_time
