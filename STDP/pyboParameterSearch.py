from Helper.Configurer import Configurer
import sys


import numpy as np
from pybo import solve_bayesopt
import time
import sys
import os
from itertools import product
import datetime
from stdp import Stdp
from copy import deepcopy
import sample as sg
import pickle

def stdp_cost_function(bounds):
    global filename, path_from, debug

    a_plus,a_ratio,t_plus,t_minus,alpha = bounds

    stdp_net = Stdp(config_stdp['run_id'],
                         config_stdp['num_afferents'], config_stdp['num_neurons'], config_stdp['weights'],
                         config_stdp['load_weights'], config_stdp['weights_list'], config_stdp['weights_path'],
                         config_stdp['training'], config_stdp['output_type'],
                         config_stdp['verbose'],
                         config_stdp['historic_weights_path'], config_stdp['saving_historic_weights'],
                         config_stdp['alpha'], config_stdp['theta'], config_stdp['a_plus'], config_stdp['a_ratio'],
                         config_stdp['tau_m'], config_stdp['tau_s'], config_stdp['t_plus'], config_stdp['t_minus'], config_stdp['k'], config_stdp['k1'], config_stdp['k2']
                         )


    if debug:
        return a_plus*a_ratio*t_plus*t_minus*alpha

    stdp_net.test_with_pattern(filename, path_from)
    sim_len = stdp_net.duration
    num_afferents = stdp_net.num_afferents
    #iterate through pattern calling  step function
    for j in xrange(sim_len):
        test_pattern = deepcopy(stdp_net.spike_trains[:,j])
        test_indices = []

        for i in xrange(len(test_pattern)):
            if test_pattern[i] == 1:
                test_indices.append(i)

        stdp_net.step(test_indices)
        # Progress bar.
        progress = (j / float(sim_len - 1)) * 100
        sys.stdout.write("Processing spikes: %d%% \r" % progress)
        sys.stdout.flush()

    score = stdp_net.calculate_metric(time_end = stdp_net.count, time_start = 0,pybo_mode=True)
    print 'metric: ', score

    return score


def start_pybo(n,filename,root_data_folder,results_folder,runID=None):
    bounds = [[0.02,0.035],[0.90,0.99],[16.00, 19.00],[25.00, 28.00],[0.20, 0.78]]
    xbest,model,info = solve_bayesopt(stdp_cost_function,bounds,niter=n,verbose=True)

    if runID == None:
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        run_id_str = 'pyboSearch_' + st + '_' + filename
        print '~~~~~~~~~~~~~~~~~~~~~~~~'
        print run_id_str
    else:
        run_id_str = 'pyboSearch_' + runID + '_'

    results_folder = os.path.join(root_data_folder,results_folder)

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    results_file = os.path.join(results_folder,run_id_str)

    result_tuple = (xbest, model, info)
    pickle.dump(result_tuple, open(results_file+'.p', "wb" ))
    vs = ['a_plus','a_ratio','t_plus','t_minus','alpha']

    # Open a file
    fo = open(results_file+'.txt', "wb")

    # Close opend file
    for i,v in enumerate(vs):
        s = v +' '+ str(xbest[i]) + '\n'
        print s,
        fo.write(s)
    fo.close()

    print 'Created the following files:'
    print results_file + '.txt'
    print results_file + '.p'

    print info

"""
Global variables:
"""
debug = None
filename = None
path_from = None
config_stdp = None
"""
Needed for pybo.
"""

def main(argv):
    global debug, filename,path_from,stpd_config_path,config_stdp
    argc = len(argv)
    if argc > 3 or argc == 1:
        print 'pyboParameterSearch usage:'
        print '\t config: .ini file decribing the parameter search'
        print '\t runID(optional):'
        return


    pybo_configurer = Configurer('STDP/config/pyboConfig.ini').config_section_map(argv[1])
    stpd_configurer = Configurer(pybo_configurer['stdp_config_file'])
    argc = len(argv)

    if argc == 3:
        runID = argv[2]
    else:
        runID = None

    print 'Starting pyboSearch'
    print '####################'
    print 'debug\t\t\t', pybo_configurer['debug']
    print 'number_of_iterations\t', pybo_configurer['number_of_iterations']
    print 'root_data_folder\t', pybo_configurer['root_data_folder']
    print 'root_sample_folder\t', pybo_configurer['root_sample_folder']
    print 'results_folder\t', pybo_configurer['results_folder']
    print 'sample\t\t\t', pybo_configurer['sample']

    """
    GLOBAL VARIABLES FOR PYBO START
    """
    debug = pybo_configurer['debug']
    filename = pybo_configurer['sample']
    path_from = os.path.join(pybo_configurer['root_data_folder'] + pybo_configurer['root_sample_folder'])
    config_stdp = stpd_configurer.config_section_map(pybo_configurer['stdp_config'])
    """
    GLOBAL VARIABLES FOR PYBO END
    """

    start_pybo(int(pybo_configurer['number_of_iterations']),
             filename,
             pybo_configurer['root_data_folder'],
             pybo_configurer['results_folder'],
             runID)

if __name__ == "__main__":
    main(sys.argv)
