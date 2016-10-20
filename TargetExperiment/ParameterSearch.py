from TargetExperiment import TargetExperiment
import numpy as np
import time
import os
from itertools import product
import datetime
import getpass
from timeit import default_timer as timer

class ParameterSearch(object):
    def __init__(self,run_id,parameter_bounds,path='./',cuda_device=0):

        self.cuda_device = cuda_device
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        self.run_id = 'pSearch_' + str(run_id) + '_' + getpass.getuser() + '_' + st

        self.path = path + self.run_id + '/'
        print self.path

        assert not os.path.exists(self.path), 'Experiment with this id already exists.'

        os.makedirs(self.path)

        self.overwrite_dict_list = self.get_all_experiments(parameter_bounds)
        print self.overwrite_dict_list


        print

    def run(self):
        env_config = ["Integration/config/Environment_test_"+str(i+1)+".ini" for i in xrange(3)]
        N = len(self.overwrite_dict_list)
        for i,d in enumerate(self.overwrite_dict_list):
            print '# Experiment:', i+1, '/' , N,
            start = timer()
            # ...
            test = TargetExperiment(env_config,
                                    path=self.path,
                                    show_plots=False,
                                    run_time=700,
                                    run_id=i,
                                    overwrite=d,
                                    cuda_device=self.cuda_device)
            ok, score1,score2, spikes = test.run()
            if ok:
                results_file = open(self.path+'results.txt', 'a')
                parameter_index_file = open(self.path+'parameter_index.txt', 'a')
                result_string = str(i) + ' ' + str(score1) + ' ' + str(score2)
                parameter_string = str(i) + ' '
                for s in spikes:
                    result_string += ' '
                    result_string += str(s)
                parameter_string += str(d)
                results_file.write(result_string + '\n')
                parameter_index_file.write(parameter_string + '\n')
                results_file.close()
                parameter_index_file.close()
                end = timer()
                print ' that took ', (end - start), ' seconds and so estimated time remaining is ', (end - start)*(N-i)/3600.0, ' hours.'
            else:
                print 'Warning! : Nans detected!'

            del test,score1,score2,spikes



    def get_all_experiments(self,bounds):
        """
        This function returns a list of overwrite dictionaries for use
        in the dragonfly brain
        Args:
            bounds (dict) : this defines the parameters you want to search over
            example:
            >>> bounds = {'type' : ['a','b'], 'value' : np.linspace(0,1,3)}
            >>> get_all_experiments(bounds)
            [{'type': 'a', 'value': 0.0},
             {'type': 'a', 'value': 0.5},
             {'type': 'a', 'value': 1.0},
             {'type': 'b', 'value': 0.0},
             {'type': 'b', 'value': 0.5},
             {'type': 'b', 'value': 1.0}]

        Return (dict) : list of overwrite dictionaries, as in example

        """
        ranges=[]
        for v in bounds.values():
            ranges.append(v)

        # List to store all the combinations
        overwrite_dict_list = []
        for element in product(*ranges):
            d = {}
            for i,key in enumerate(bounds.keys()):
                d[key] = element[i]
            overwrite_dict_list.append(d)
        return overwrite_dict_list


if __name__ == "__main__":
    import sys
    t = int(sys.argv[1])
    bounds = {}
    if t == 3:
        bounds["tau_gaba"]                = np.linspace(100,700,5)
        bounds["synapse_max_conductance"] = np.linspace(0.2,1.0,4)
        bounds["estmd_gain"]              = [10.0,20.0]
        bounds["synapses_file_name"] = ['200.dat', '300.dat', '400.dat', '1000.dat']
    elif t == 1:
        r = 2.0
        n = 5
        bounds["e_na"]   = np.linspace(115-r,115+r,n)
        bounds["e_k"]    = np.linspace(-12-r,-12+r,n)
        bounds["e_l"]    = np.linspace(10.613-r,10.613+r,n)
    elif t == 2:
        #bounds["synapse_max_conductance"] = [0.0,0.4,5]
        #bounds["e_gaba"]                  = [-30,10,5]
        bounds["tau_gaba"] = [100,1000,101]
    elif t == 0:
        bounds["noise_stddev"] = [0,1.0,101]
    elif t == 4:
        bounds["tau_gaba"]                = np.linspace(100,1000,10)
        bounds["synapse_max_conductance"] = np.linspace(0.0,1.0,11)
    elif t == 5:
        bounds["e_gaba"] = np.linspace(-30,10,4)
        bounds["synapses_file_name"] = ['200.dat',
                                        '300.dat',
                                        '400.dat',
                                        '500.dat',
                                        '600.dat',
                                        '700.dat',
                                        '800.dat',
                                        '900.dat',
                                        '1000.dat']
    p = ParameterSearch(t,bounds,cuda_device=t)
    p.run()
