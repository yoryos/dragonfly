import csv
import numpy as np
import getpass
if getpass.getuser() in ["lm1015"]:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os
from itertools import product
import time
import datetime

from Integration.DragonflyBrain import DragonflyBrain
from Integration.BrainModule import BrainModule

class GenerateTrainingData:
    def __init__(self,
                 environment_configs,
                 estmd_config="Integration/config/ESTMD_norm.ini",
                 dragonfly_config="Integration/config/Dragonfly_test_1.ini",
                 cstmd_config="Integration/config/CSTMD1_norm.ini",
                 run_time=100,
                 train_time=50,
                 run_id=None,
                 path='./',
                 overwrite={},
                 cuda_device = 3,
                 debug = False):
        """
        Args:
            environment_configs (list): list of path strings pointing to environment config files
            estmd_config (str):
            dragonfly_config (str):
            cstmd_config (str):
            run_time (int) : number of ms to run the simulator for
            run_id (int) : id for saving files
            path (str) : path at which the storage folder should be create
            overwrite (dict) : dictionary of parameters to overwrite
            cuda_device (int) : path at which the storage folder should be create
            debug (bool) : send errors into the stddout not the log file
        """
        self.environment_configs=environment_configs
        self.estmd_config=estmd_config
        self.dragonfly_config=dragonfly_config
        self.cstmd_config=cstmd_config
        self.run_time=run_time
        self.train_time=train_time
        self.overwrite=overwrite
        self.cuda_device = cuda_device
        if run_id is None:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
            self.run_id = 'stdpTrainingData_' + getpass.getuser() + '_' + st
        else:
            self.run_id = 'stdpTrainingData_' + str(run_id)

        self.path = path + self.run_id + '/'
        assert not os.path.exists(self.path), 'Training data with this id already exists.'
        os.makedirs(self.path)

        self.overwrite["soma_electrodes"] = False
        self.overwrite["number_of_electrodes"] = 1000
        self.overwrite["random_electrodes"] = False

        print '  _______________________________________________________________'
        print '  STDP Training data generation with overwritten parameters : \n  ' , self.overwrite
        self.debug = debug

    def __step(self,id,environment_config,log_file):

        id = id + 1
        print '  ',id,'/',4,' : Running environment.'
        log = open(log_file, "w")

        if not self.debug:
            #Reroute the stdout and error stream to a file
            sys.stdout = log
            sys.stderr = log
        dragon = DragonflyBrain(1, cstmd_buffer_size=1,
                                environment_config=environment_config,
                                estmd_config=self.estmd_config,
                                dragonfly_config=self.dragonfly_config,
                                cstmd_config=self.cstmd_config,
                                stdp_config = None,
                                overwrite=self.overwrite,
                                run_until=BrainModule.CSTMD1,
                                cuda_device=self.cuda_device)
        for i in xrange(self.run_time):
            ok = dragon.step()
            if not ok:
                if not self.debug:
                    # Give back the stdout and stderr
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                print 'dragon failed at time ', i*dragon.global_dt
                return False

        if not self.debug:
            # Give back the stdout and stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        print '  ',id,'/',4,' : Finished.'
        dragon.data_dump(Environment=True,ESTMD=True, directory=self.path,run_id_prefix=True)

        # [:,1:] slices off the time index data
        spiking_rate_data = deepcopy(dragon.cstmd.calculate_spiking_rate_exp())[:,1:]
        min, max = (spiking_rate_data.min(), spiking_rate_data.max())

        # Warning time range is now half way to becoming depricated
        dragon.cstmd.save_spikes(directory=self.path,
                                 name='spikes_'+str(id)+'.dat',
                                 run_id_prefix=False,
                                 time_range=(self.run_time-self.train_time,self.run_time),
                                 transpose=True)

        np.savez(self.path+'spikes_'+str(id)+'.npz',
                 spike_trains=np.array(dragon.cstmd.spike_history).transpose())

        dragon.cstmd.plot_spikes(show=False,save_path=self.path+'spikes_'+str(id)+'.png')




        del dragon
        return True, min, max

    def run(self):
        for i,environment in enumerate(self.environment_configs):
            ok, min, max = self.__step(i,environment,self.path+'log_'+str(i)+'.txt')
            res = str(i) + ' ' + str(min) + ' ' + str(max) + '\n'
            results_file = open(self.path+'results.txt', 'a')
            results_file.write(res)
            results_file.close()
            if not ok:
                print 'Failed'
                return


        return True

env_config = ["TargetExperiment/config/001_stdp_env/env_"+str(i+1)+".ini" for i in xrange(4)]
o = {'synapse_max_conductance': 1.0, 'synapses_file_name': '1000.dat', 'estmd_gain': 10.0, 'tau_gaba': 400.0, 'noise_stddev': 0}
g = GenerateTrainingData(env_config,run_time=500,train_time=500,overwrite=o)
g.run()
