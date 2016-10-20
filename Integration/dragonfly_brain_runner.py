import getpass
import os
from DragonflyBrain import DragonflyBrain

cstmd_buffer_size = 20
run_time = 600
run_until = DragonflyBrain.stage_CSTMD1

# o = {"synapse_max_conductance" : 1.0, "e_gaba":-20, "cstmd_speed_up":40}
o = {}

config_path = "Integration/config/clean_config"
envir = os.path.join(config_path, "Environment_up-down_left-right_.ini")
estmd = os.path.join(config_path, "ESTMD.ini")
cstmd = os.path.join(config_path, "CSTMD1_dump_strong.ini")
stdp  = os.path.join(config_path, "STDP.ini")
drag  = os.path.join(config_path, "Dragonfly.ini")

dragon = DragonflyBrain(dt = 1.0,
                        cstmd_buffer_size=cstmd_buffer_size,
                        environment_config=envir,
                        estmd_config=estmd,
                        dragonfly_config=drag,
                        cstmd_config=cstmd,
                        stdp_config =stdp,
                        run_until=run_until,
                        overwrite=o,
                        pyplot=False,
                        webcam=False)

if getpass.getuser() not in []:
    response = raw_input("Press any key to continue")

dragon.debug = True

for i in xrange(run_time):
    if not dragon.step():
        print "Breaking loop"
        break

dragon.data_dump(Environment=True, ESTMD=True, CSTMD=True, STDP=False)
dragon.time_analysis()
