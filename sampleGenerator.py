import sys
import sample as sg
from Helper.Configurer import Configurer


#creates a test_pattern using the functions implemented in sample.py

# Generate Sample
sample_duration = 500
num_neurons = 1000
num_patterns = 1 #this should be 2 but there is a bug that needs fixing!
start_time = 350
end_time = 400
pattern_filenames = ["spikes_4.dat"]

pattern_path = "/homes/zl4215/imperial_server/DATA/STDP/cstmd1_samples/stdpTrainingData_wobble_01/"
sample_filename = 'short_testing_sample'
# for i in xrange(len(pattern_filenames)):
    # sample_filename += pattern_filenames[i]
print 'creating sample: ', sample_filename
sample_directory = "STDP/tests/samples/"


def main(argv):

    if len(argv) > 2 or len(argv) == 1:
        print 'SampleGenerator.py useage: config_file option'
        exit()

    option = argv[1]


    configurer = Configurer('STDP/config/sample_GenConfig.ini')
    config     = configurer.config_section_map(option)
    
    test_sample = sg.Sample(config['sample_duration'], config['start_time'], config['end_time'],
                                    config['sample_directory'],rate_max = config['rate_max'], num_neurons = config['num_neurons'],
                                    num_patterns = config['num_patterns'], reading_pattern = config['reading_pattern'], pattern_path = config['pattern_path'],
                                    pattern_filename = config['pattern_filename'])

    #generate a sample
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

    #set the filename
    test_sample.filename = config['sample_filename']

    print 'start positions of the patterns are', test_sample.start_positions

    #print generated sample:
    print "the spike trains after adding the pattern are: "
    print test_sample.spike_trains

    #save sample
    test_sample.save()


if __name__ == "__main__":

    main(sys.argv)
