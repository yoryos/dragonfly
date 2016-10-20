import getpass
import os
from DragonflyBrain import DragonflyBrain

cstmd_buffer_size = 20
run_time = 1000
run_until = DragonflyBrain.stage_CSTMD1

o = {"save_neurons":"/home/lm1015/DATA/CSTMD1/CSTMD_Morph/neuron_collection_5n.p",
     'preloaded_morphology_path':None}

config_path = "Integration/config/clean_config"
envir = "Integration/config/Environment_test_1.ini"
estmd = os.path.join(config_path, "ESTMD.ini")
cstmd = os.path.join(config_path, "CSTMD1_preload.ini")
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

print "Done"

