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

class TargetExperiment(object):
    def __init__(self,
                 environment_configs,
                 show_plots,
                 estmd_config="Integration/config/ESTMD_norm.ini",
                 dragonfly_config="Integration/config/Dragonfly_test_1.ini",
                 cstmd_config="Integration/config/CSTMD1_norm.ini",
                 run_time=700,
                 run_id=None,
                 path='./',
                 overwrite={},
                 cuda_device = 2):
        """
        Args:
            environment_configs (list): list of path strings pointing to environment config files
            show_plots (bool) : if true then plot graphs with matplotlib to screen
            estmd_config (str):
            dragonfly_config (str):
            cstmd_config (str):
            run_time (int) : number of ms to run the simulator for
            run_id (int) : id for saving files
            path (str) : path at which the storage folder should be create
            overwrite (dict) : dictionary of parameters to overwrite
        """
        assert len(environment_configs) == 3, "ParameterSearch requires 3 environment configs"
        self.environment_configs=environment_configs
        self.estmd_config=estmd_config
        self.dragonfly_config=dragonfly_config
        self.cstmd_config=cstmd_config
        self.run_time=run_time
        self.spiking_rate_data = []
        self.spike_count = []
        self.min_diffs = []
        self.show_plots = show_plots
        self.overwrite=overwrite
        self.cuda_device = cuda_device
        if run_id is None:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
            self.run_id = 'targetExperiment_' + getpass.getuser() + '_' + st
        else:
            self.run_id = 'targetExperiment_' + str(run_id)

        self.path = path + self.run_id + '/'
        assert not os.path.exists(self.path), 'Experiment with this id already exists.'
        os.makedirs(self.path)


        self.legend_names = { 1 : 'both', 2 : 'left', 3 : 'right'}

        print '  _______________________________________________________________'
        print '  Target Experiment with overwritten parameters : \n  ' , self.overwrite
        self.debug = False

    def __step(self,id,environment_config,log_file):
        id = id + 1
        print '  ',id,'/',3,' : Running simulator.'
        log = open(log_file, "w")
        if not self.debug:
            sys.stdout = log
            sys.stderr = log
        dragon = DragonflyBrain(1, cstmd_buffer_size=1,
                                environment_config=environment_config,
                                estmd_config=self.estmd_config,
                                dragonfly_config=self.dragonfly_config,
                                cstmd_config=self.cstmd_config,
                                stdp_config = None,
                                overwrite=self.overwrite,
                                run_until=DragonflyBrain.stage_CSTMD1,
                                cuda_device=self.cuda_device)
        for i in xrange(self.run_time):
            ok = dragon.step()
            if not ok:
                if not self.debug:
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                print 'dragon failed at time ', i*dragon.global_dt
                return False
        if not self.debug:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        print '  ',id,'/',3,' : Finished. Saving data.'


        srd = deepcopy(dragon.cstmd.calculate_spiking_rate_exp(20.0))
        self.spiking_rate_data.append(srd)
        self.spike_count.append(np.array(dragon.cstmd.spike_history).sum())
        dragon.cstmd.save_spike_rate_data(directory=self.path,
                                          name='rates_'+str(id)+'.dat',
                                          run_id_prefix=False)

        dragon.cstmd.plot_firing_rate(show=False,save_path=self.path+'rates_'+str(id)+'.png')
        dragon.cstmd.plot_spikes(show=False,save_path=self.path+'spikes_'+str(id)+'.png')

        dragon.cstmd.save_spikes(directory=self.path,
                                 name='spikes_'+str(id)+'.dat',
                                 run_id_prefix=False)
        del dragon
        return True

    def run(self):
        for i,environment in enumerate(self.environment_configs):
            ok = self.__step(i,environment,self.path+'log_'+str(i)+'.txt')
            if not ok:
                return False,0,0,0


        self.plot_superimposed_rates(self.spiking_rate_data,show=self.show_plots,save_path=self.path+'superimposed.png')
        self.plot_diff(self.spiking_rate_data,show=self.show_plots,save_path=self.path+'diff.png')

        results_file = open(self.path+'results.txt', 'w')
        min_sum = self.min_diff_sum(self.spiking_rate_data)
        max_sum = self.max_diff_sum(self.spiking_rate_data)
        results_file.write('min_diff_sum : ' +  str(min_sum))
        results_file.write('\nmax_diff_sum : ' +  str(max_sum))
        results_file.write('\nspikes = ' + str(self.spike_count))
        results_file.write('\ntotal spikes = ' + str(np.array(self.spike_count).sum()))
        results_file.write('\naverage spikes = ' + str(np.array(self.spike_count).mean()))
        results_file.close()

        return True, min_sum, max_sum, self.spike_count



    """
    Plot all the firing rates super imposed on one another
    """
    def plot_superimposed_rates(self,data,show,save_path):
        time = data[0][:,0]
        number_of_columns = len(data[0][0,:])
        electrodes = number_of_columns - 1
        for i in xrange(1,number_of_columns):
            ax = plt.subplot(electrodes, 1, i)
            for j in xrange(len(data)):
                ax.plot(time,data[j][:,i],label=self.legend_names[j+1])
                ax.yaxis.grid(True)
                plt.ylabel('Firing rate /s')

        plt.xlabel('Time (msec)')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),ncol=5)
        plt.subplot(electrodes,1,1)
        plt.title('Firing rates vs time for each environment input')
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.gcf().clear()

    """
    plot the min difference between the experiments
    """
    def plot_diff(self,data,show,save_path):
        cost = self.min_diff_sum(data)
        time = data[0][:,0]
        number_of_columns = len(data[0][0,:])
        electrodes = number_of_columns - 1
        plt.figure()
        for i in xrange(1,number_of_columns):
            ax = plt.subplot(electrodes, 1, i)
            res = self.min_array(data[0][:,i],data[1][:,i],data[2][:,i])
            ax.plot(time,res)
            ax.yaxis.grid(True)
            plt.ylabel('Firing rate /s')
        plt.xlabel('Time (msec)')
        plt.subplot(electrodes,1,1)
        plt.title('Minimum difference calculation. Total diff = ' + str(round(cost,2)))
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.gcf().clear()

    """
    for each element compute min(|x-y|,|x-z|)
    """
    def min_array(self,x,y,z):
        assert x.shape==y.shape==z.shape, "min_array : must get arrays of same shape"
        m, = x.shape
        res = np.zeros(m)
        for i in xrange(m):
                xy = np.abs(x[i]-y[i])
                xz = np.abs(x[i]-z[i])
                res[i] = min(xy,xz)
        return res

    """
    sum up all the min_arrays for 3 experiments
    """
    def min_diff_sum(self,data):
        number_of_columns = len(data[0][0,:])
        sum = 0
        for i in xrange(1,number_of_columns):
            sum += self.min_array(data[0][:,i],data[1][:,i],data[2][:,i]).sum()
        return sum

    def max_diff_sum(self,data):
        number_of_columns = len(data[0][0,:])
        sum = 0
        for i in xrange(1,number_of_columns):
            d1 = np.abs(data[0][:,i]-data[1][:,i]).sum()
            d2 = np.abs(data[0][:,i]-data[2][:,i]).sum()
            sum += max(d1,d2)
            # sum += self.min_array(,data[2][:,i]).sum()
        return sum
"""
#env_config = ["Integration/config/Environment_test_"+str(i+1)+".ini" for i in xrange(3)]
env_config = ["TargetExperiment/config/004_wobble_experiment/env_"+str(i+1)+".ini" for i in xrange(3)]
o = {'number_of_electrodes': 1000, 'soma_electrodes': True, 'random_electrodes': False}
test = TargetExperiment(env_config,show_plots=False,run_time=700,overwrite=o)
test.run()
"""
