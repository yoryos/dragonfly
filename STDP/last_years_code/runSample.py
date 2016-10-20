import sample as sg
import os

#creates a test_pattern using the functions implemented in sample.py
def IntegratedPSearch(duration,
                      start_time,
                      end_time,
                      num_neurons,
                      sample_directory,
                      filename,
                      pattern_path,
                      pattern_filename,
                      reading_pattern,
                      num_patterns=1,
                      max_rate = 1.1):

    test_sample = sg.Sample(duration,
                            start_time,end_time,
                            num_neurons=num_neurons,
                            sample_directory = sample_directory,
                            filename = filename,
                            pattern_path = pattern_path,
                            pattern_filenames = [pattern_filename],
                            reading_patterns = True,
                            num_patterns=num_patterns,
                            rate_max = max_rate
                            )

    # generate a sample
    test_sample.generate_sample()

    print "pattern before generating is:"
    print test_sample.patterns

    print "the spike trains before adding the pattern are: "
    print test_sample.spike_trains

    #generate_patterns
    test_sample.generate_patterns()

    #read in patterns



    print "patterns are:"
    print test_sample.patterns
    test_sample.insert_patterns()

    print 'start positions of the patterns are', test_sample.start_positions

    #print generated sample:
    print "the spike trains after adding the pattern are: "
    print test_sample.spike_trains

    #save sample
    test_sample.save()

s = '/homes/lm1015/vol/DATA/STDP/cstmd1_samples/stdpTrainingData_falling_01'
import sys
i = sys.argv[1]
IntegratedPSearch(45000,50,100,1000,s, 'sample_'+i,s, '/spikes_'+i+ '.dat',True)
