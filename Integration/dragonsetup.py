import getpass
import os
from DragonflyBrain import DragonflyBrain

cstmd_buffer_size = 700
run_time = 700
run_until = DragonflyBrain.stage_CSTMD1

o = {"synapse_max_conductance" : 1.0, "e_gaba":-20, "cstmd_speed_up":20}

dragon = DragonflyBrain(dt = 1.0,
                        cstmd_buffer_size=cstmd_buffer_size,
                        environment_config="Integration/config/Environment_test_1.ini",
                        estmd_config="Integration/config/ESTMD_norm.ini",
                        dragonfly_config="Integration/config/Dragonfly_test_1.ini",
                        cstmd_config="Integration/config/CSTMD1_cps15.ini",
                        stdp_config ="Integration/config/STDP_test_1.ini",
                        run_until=run_until,
                        overwrite=o,
                        pyplot=False,
                        webcam=False)
