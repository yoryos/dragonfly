from TargetExperiment import TargetExperiment

env_config = ["TargetExperiment/config/004_wobble_experiment/env_"+str(i+1)+".ini" for i in xrange(3)]
o = {'number_of_electrodes': 5, 'soma_electrodes': True, 'random_electrodes': False}
test = TargetExperiment(env_config,show_plots=False,run_time=700,overwrite=o,cuda_device=3)
test.run()
