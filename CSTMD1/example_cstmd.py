from CSTMD1.Cstmd1 import Cstmd1
from Helper.Configurer import  Configurer

dt = 1.0
cstmd_speed_up = 20
run_time = 200

config_file_path="Integration/Sample_Config/CSTMD1.ini"
config = Configurer(config_file_path)


cstmd = Cstmd1(global_dt = dt,
               speed_up=cstmd_speed_up,
               estmd_dim=[64, 48],
               buffer_size=1,
               debug_mode=0,
               sim_parameters=config.config_section_map("CSTMD1_Simulator"),
               morph_param=config.config_section_map("Morphology_Homogenised_Top"),
               run_id = "test")


response = raw_input("Press any key to continue")

status = True
estmd_stim = [[0, 0, 1], [0, 1, 1]]
for i in xrange(run_time):
    print "Outer step", i

    status, spike_train = cstmd.step(dt,estmd_stim)

    if not status:
        print 'CSTMD1 step failed, exiting loop'
        break