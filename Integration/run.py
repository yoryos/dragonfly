from DragonflyBrain import DragonflyBrain

cstmd_buffer_size = 1
run_time = 100
run_until = DragonflyBrain.stage_ALL

####### Put paths to config files here:

ENVIR = "Integration/Sample_Config/Environment_1.ini"
ESTMD = "Integration/Sample_Config/ESTMD.ini"
CSTMD = "Integration/Sample_Config/CSTMD1.ini"
STDP  = "Integration/Sample_Config/STDP.ini"
DRAG  = "Integration/Sample_Config/Dragonfly.ini"

o = {}

dragon = DragonflyBrain(dt = 1.0,
                        cstmd_buffer_size=cstmd_buffer_size,
                        environment_config=ENVIR,
                        estmd_config=ESTMD,
                        dragonfly_config=DRAG,
                        cstmd_config=CSTMD,
                        stdp_config =STDP,
                        run_until=run_until,
                        overwrite=o,
                        pyplot=False,
                        webcam=False,
                        cuda_device=1)


raw_input("Press any key to continue")
dragon.debug = True
dragon.run(100)
dragon.data_dump(True,True,True,True)
loop_time, env_time, estmd_time, cstmd_time, stdp_time, rl_time = dragon.time_analysis()

print "Full loop took %f seconds" % (loop_time)
print "Time in environment: ", env_time
print "Time in ESTMD: ", estmd_time
print "Time in CSTMD: ", cstmd_time
print "Time in SDTP: ", stdp_time
print "Time in RL: ", rl_time
