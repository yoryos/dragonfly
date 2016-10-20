from Helper.Configurer import Configurer
import stdp
import sys
import numpy as np
from copy import deepcopy




def test_stdp_step(sample_name,path_to_sample, weights_path,
                    save_weights, load_weights,
                    weights_list,
                    parameters_path, save_parameters,
                    stdp_config, stdp_config_section,
                    verbose):
    """
    function which runs stdp_step
    for a sample_length timesteps
    """

    #initialize stdp

    configurer = Configurer(stdp_config)
    config_stdp = configurer.config_section_map(stdp_config_section)


    stdp_net = stdp.Stdp(config_stdp['run_id'],
                         config_stdp['num_afferents'], config_stdp['num_neurons'], config_stdp['weights'],
                         load_weights, weights_list, weights_path,
                         config_stdp['training'], config_stdp['output_type'],
                         config_stdp['verbose'],
                         config_stdp['pattern_filename'], config_stdp['pattern_path'],
                         config_stdp['historic_weights_path'], config_stdp['saving_historic_weights'],
                         config_stdp['alpha'], config_stdp['theta'], config_stdp['a_plus'], config_stdp['a_ratio'],
                         config_stdp['tau_m'], config_stdp['tau_s'], config_stdp['t_plus'], config_stdp['t_minus'], config_stdp['k'], config_stdp['k1'], config_stdp['k2']
                         )

        #load full length test pattern
    stdp_net.test_with_pattern(sample_name, path_to_sample)

    #iterate through pattern calling  step function
    sim_len = stdp_net.spike_trains.shape[1]
    if verbose:
        print 'running sim length: ', sim_len
    for j in xrange(sim_len):
        test_pattern = deepcopy(stdp_net.spike_trains[:,j])
        test_indices = []

        #transformation of pattern_input into indices (simulating cstmd1 output)
        for i in xrange(len(test_pattern)):
            if test_pattern[i] == 1:
                test_indices.append(i)

        new_output_spikes = stdp_net.step(test_indices,output = config_stdp['output_type'])


        # Track progress
        progress = (j / float(sim_len-1)) * 100
        sys.stdout.write("Generating spike trains: %d%% \r" % progress)
        sys.stdout.flush()

    #calculate_metric
    if stdp_net.start_positions == None:
        if verbose:
            print 'cannot compute metric for this sample'
    else:
        metric = stdp_net.calculate_metric(time_end = stdp_net.count, time_start = 25000)
        if verbose:
            print 'metric: ', metric

    if save_parameters == True:
        stdp_net.save_parameters(parameters_path)


    #saving weights
    if save_weights:
        stdp_net.save_current_weights(weights_path)


    #plotting
    if verbose:
        stdp_net.plot_membrane_potential(verbose = True, _duration = sim_len)


def main(argv):

    argc = len(argv)
    if argc > 3 or argc == 1:
        print 'runSTDP usage:'
        print '\t config: .ini file decribing the runSTDP'
        print '\t runID(optional):'
        return

    if argc == 3:
        runID = argv[2]
    else:
        runID = None


    configurer = Configurer('STDP/config/stdp_stepConfig.ini')
    config = configurer.config_section_map(argv[1])

    if config['verbose'] == True:
        print 'Starting runSTDP'
        print '####################'
        print 'sample used\t', config['path_to_sample'] + config['sample_name']


    test_stdp_step(config['sample_name'],config['path_to_sample'], config['weights_path'],
                   config['save_weights'], config['load_weights'],
                   config['weights_list'],
                   config['parameters_path'], config['save_parameters'],
                   config['stdp_config'], config['stdp_config_section'],
                   config['verbose'],
                    )




if __name__ == "__main__":

    main(sys.argv)
