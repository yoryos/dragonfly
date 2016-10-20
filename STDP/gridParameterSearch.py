import numpy as np
import time
import sys
import os
from itertools import product
import datetime
from stdp import Stdp
from copy import deepcopy
import sample as sg

class ParameterSearch:
    """
    class which performs a basic search through stdp_parameters,
    calculates the metric for each stdp_run with a set of parameters
    saves the metrics and parameters in a file and
    chooses the parameters that result in the best metric
    """
    def __init__(self,sample_name,parameter_bounds,path='./',run_id=None):

        self.sample_name = sample_name
        if run_id is None:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
            self.run_id = 'parameterSearch_' + self.sample_name + st
        else:
            self.run_id = 'parameterSearch_' + str(run_id)

        self.path = path + self.run_id + '/'
        print self.path
        self.path_figures = self.path+'figures/'
        self.path_patterns = self.path + 'patterns/'
        # assert not os.path.exists(self.path), 'Experiment with this id already exists.'

        os.makedirs(self.path)
        os.makedirs(self.path_figures)
        os.makedirs(self.path_patterns)

        self.overwrite_dict_list = self.get_all_experiments(parameter_bounds)
        print self.overwrite_dict_list


    def create_pattern(self, index, sample_duration, rate_path, spike_path, sample_directory):
        """
        function which creates a pattern using
        output information of the cstmd1
        parameters: path to spiking_rate_info
                    path to spike_trains
                    direction of the pattern
        """
        # pattern_path = self.path + pattern_nature + '.npz'
        #read info of samples
        rates = np.loadtxt(rate_path)
        maxima = rates[:,2]
        minima = rates[:,1]

        maximum_rate = maxima[index]
        minimum_rate = minima[index]
        spikes = (np.loadtxt(spike_path)).transpose()
        # print spikes.shape
        #generate sample
        num_neurons = spikes.shape[0]
        pattern_duration = spikes.shape[1]
        sample_filename = "cstmd1_training_sample_" + str(i)


        test_sample = sg.Sample(sample_duration, minimum_rate, maximum_rate, pattern_duration = pattern_duration , num_neurons = num_neurons, inv_ratio = 1, patterns = spikes)
        # no need to generate pattern as we are giving the pattern from cstmd1
        print "patterns are:"
        print test_sample.patterns
        #generate a sample
        test_sample.generate_sample()

        test_sample.insert_patterns()
        #set the filename
        test_sample.filename = sample_filename
        print 'start positions of the patterns are', test_sample.start_positions
        #save sample
        print "the spike trains after adding the pattern are: "
        print test_sample.spike_trains

        test_sample.save(sample_directory)




    def run(self):
        for k,d in enumerate(self.overwrite_dict_list):
            # results_file = open(self.path+'results.txt', 'a')
            # parameter_index_file = open(self.path+'parameter_index.txt', 'a')


            stdp_net = Stdp(run_id=k,
                            num_afferents=1000,num_neurons = 1, alpha=d['alpha'],a_plus=d['a_plus'],a_ratio=d['a_ratio'],
                            t_plus=d['t_plus'],t_minus=d['t_minus']
                            )

            #load full length test pattern
            stdp_net.test_with_pattern(self.sample_name, "/homes/zl4215/imperial_server/DATA/STDP/cstmd1_samples/stdpTrainingData_linear_02/")
            print 'loaded pattern'
            sim_len = 45000
            num_afferents = 1000
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



            score = stdp_net.calculate_metric(time_end = stdp_net.count, time_start = 5000)
            output = stdp_net.historic_output

            print 'metric: ', score
            #save parameter config
            d = {"RunID": self.run_id,
                 "ParameterRun_number":k,
                 "a_plus":a_plus,
                 "a_ratio":a_ratio,
                 "t_plus":t_plus,
                 "t_minus":t_minus,
                 "alpha":alpha,
                 "metric":score
                 }
            path_to = self.path
            path_figures = self.path_figures
            path_parameters = self.path_patterns


            parameter_filename = path_parameters + 'ParameterRun_number_' + str(k) + '.pyb'
            np.save(parameter_filename,d)
            #save_figure
            # figure_name = 'PyboSearch_' + runID + 'a_plus_' + str(a_plus) + 'a_ratio_' + str(a_ratio) + 't_plus_' + str(t_plus) + 't_minus_' + str(t_minus) + 'alpha_' + str(alpha)
            stdp_net.plot_membrane_potential(verbose = False, path = path_figures, parameter_config_num = k)



            # #save score, output, parameters and figure here
            # # output_filename = 'output_of_' + str(i) + 'th_parameter_combination.dat'
            # # np.savetxt(output_filename, output)
            # # output_file = np.open(output_filename, 'w')
            # # output_file.write(output)
            # # np.write(output_file, output)
            # # score_filename = 'score_of_' + str(i) + 'th_parameter_combination.dat'
            # # np.savetxt(score_filename, score)
            # # score_file = np.open(score_filename, 'w')
            # # np.write(score_file, score)
            #
            # # parameter_filename = 'parameters_of' + str(i)
            # parameter_string = str(k) + ' : '
            # parameter_string += str(d)
            # # np.savetxt(parameter_filename,parameter_string)
            # stdp_net.plot_membrane_potential(verbose = False, parameter_config_num = k, path=self.path_figures, _duration = sim_len)
            #
            # result_string = str(j) + ' : ' + str(score)
            # result_string += ' stdp_output: '
            # for s in output:
            #     result_string += ' '
            #     result_string += str(s)
            #
            # results_file.write(result_string + '\n')
            # parameter_index_file.write(parameter_string + '\n')
            # results_file.close()
            # parameter_index_file.close()
            # # del test,score,spikes
            # return score

    def get_all_experiments(self,bounds):
        ranges=[]
        for key,val in bounds.items():
            min,max,n = val
            values = np.linspace(min,max,n)
            ranges.append(values)

        # List of store all the combinations
        overwrite_dict_list = []
        for element in product(*ranges):
            d = {}
            for i,key in enumerate(bounds.keys()):
                d[key] = element[i]
            overwrite_dict_list.append(d)
        return overwrite_dict_list

#Neuron Parameters
a_plus = 0.03125
a_ratio = 0.95
tau_m = 10.0                    # Membrane time constant in ms.
tau_s = 2.5                     # Synapse time constant in ms.
t_plus = 17.8                   # LTP modification constant in ms.
t_minus = 26.7                  # LTD modification constant in ms.
theta = 300                     # Threshold in arbitrary units.
alpha = 0.75                    # Multiplicative constant for IPSP.
k = 2.1222619                   # Multiplicative constant for EPSP.
k1 = 2.0                          # Constant for positive pulse.
k2 = 4.0



range_percent = 5
range_step_percent = 1

def get_min(value):
    min = value - (value * range_percent / 100)
    return float(min)

def get_max(value):
    max = value + (value * range_percent / 100)
    return float(max)

def get_step(value):
    return float(value * range_step_percent / 100)


# bounds = {}
# bounds["tau_gaba"] = [3.0,7.0,5]
# bounds["synapse_max_conductance"] = [0.05,0.2,5]
# bounds["estmd_gain"] = [10.0,50.0,5]


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print 'useage: ParameterSearch path_name sample_name'

    path = sys.argv[1]
    sample_name = sys.argv[2]

    bounds = {}
    bounds['a_plus']=[0.0296875,0.0328125,5]
    bounds['a_ratio']=[0.85,0.9975,5]
    # bounds["tau_m"]=[9.5,10.5, 2]
    # bounds["tau_s"]=[2.375, 2.625,2]
    bounds["t_plus"] = [15.00, 19.00,10]
    bounds["t_minus"] = [24.00, 30.00,10]
    # bounds["k1"] = [1.9, 2.1,2]
    # bounds["k2"] = [3.8, 4.2,2]
    # bounds["k"] = [2.016148805, 2.228374995,2]
    bounds["alpha"] = [0.20, 0.8,10]
    # bounds["theta"] = [285.0, 315.0,2]


    sample_duration = 45000
    rate_path = "/homes/zl4215/imperial_server/stdpTrainingData_lm1015_2016-04-19_12:08:06/results.txt"
    spike_path = "/homes/zl4215/imperial_server/stdpTrainingData_lm1015_2016-04-19_12:08:06/spikes_1.dat"
    sample_directory = "/homes/zl4215/imperial_server/stdpTrainingSamples/"


    p = ParameterSearch(sample_name,bounds, path = path)
    # def create_pattern(self, index, sample_duration, rate_path, spike_path, sample_directory):
    # for i in xrange(1):
        # p.create_pattern( i, sample_duration, rate_path, spike_path, sample_directory)

    p.run()
