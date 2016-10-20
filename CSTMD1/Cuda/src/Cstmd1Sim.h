/******************************************************************************
 * Dragonfly Project 2016
 *
 *
 * @author: Dragonfly Project 2016 - Imperial College London
 *        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)
 *****************************************************************************/

#include "Cstmd1Sim_parameters.h"
// C++
#include <stdint.h>
#include <iostream>
#include <string>
#include <list>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
// Cublas
#include <cublas_v2.h>
// Thrust
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <map>
#include <string>

#define CURAND_CALL(err) { if((err) != CURAND_STATUS_SUCCESS) \
    { std::cout << __FILE__ << " line " << __LINE__ << \
    " : CuRand execution error." << std::endl; \
    return false; } };

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

//Type definitions for tupes
typedef thrust::tuple<float, float, float, float> Tuple4;
typedef thrust::tuple<float, bool, bool> Float1Bool2;
typedef thrust::tuple<float,float> Float2;

//Type definitions for permutation iterators
typedef thrust::device_vector<float>::iterator D_float_it;
typedef thrust::device_vector<bool>::iterator D_bool_it;
typedef thrust::device_vector<int>::iterator D_int_it;
typedef thrust::permutation_iterator<D_bool_it, D_int_it> perm_iterator_bool_int;
typedef thrust::permutation_iterator<D_float_it, D_int_it> perm_iterator_float_int;

//Forward declarations of structures used by thrust
struct Hh_propagate;
struct Synapse_current;
struct Euler;
struct Reset_conductance;
struct Nan_or_inf_check;
struct Spike_detection;

class Cstmd1Sim {

private:
    //Debugging levels
    bool verboseDebug, mediumDebug, simpleDebug, setupDebug;
    int dspace, debug_e;
    // The output stream where debug will be written
    std::ostream& debugStream;

    // Total number of compartments the neuron model contains
    int nCompartments;
    // Number of synapses in existance between compartments
    int nSynapses;
    // Number of electrodes connected to compartments
    int nElectrodes;
    // Number of unique post synaptic compartments
    int nUniquePostSynCompartments;

    int T;              //Step time
    float dt;           //Time step

    float Cm;
    float spike_threshold;
    float electricalCoeff;   // coefficient for intercomparmental connections -dt / (Cm * Ra)
    float stimulusMagnitude; // coefficient for current vector, normally: gain*dt/Cm
    float dt_Cm;             // dt/Cm
    float gaba_decay_coeff;  // -dt/tau_gaba

    /*******************Cublas Constants**************************************/
    const float zero;      // zero float, useful for cublas functions arguements
    const float one;       // one float, useful for cublas function aruguments
    cublasHandle_t handle; // the cublas handle

    /*******************Curand ***********************************************/
    bool randomise_currents;
    curandGenerator_t gen;
    float d_I_mean,d_I_stddev;

    /*******************Thrust Structures*************************************/
    // This contains the main dynamics of a hodgkin huxley neuron
    Hh_propagate* hh_propagate;
    // General euler method y[i+1] += c*y[i]*dt
    Euler* euler;
    // Calculate post synaptic currents
    Synapse_current* synapse_current;
    // Reset the conductance if a spike has occured
    Reset_conductance* reset_conductance;
    // Logical or two vectors into a third
    thrust::logical_or<bool> logical_or;
    // Check for non-finite values
    Nan_or_inf_check* nan_or_inf_check;
    // spike detection
    Spike_detection * spike_detection;

    /*******************Device Vectors****************************************/
    // Morpology matrix size : numberOfCompartments*numberOfCompartments
    thrust::device_vector<float> d_electrical_connections;
    // The voltage of each compartment = size numberOfCompartments * Run time
    thrust::device_vector<float> d_v;
    // Hodgkin Huxley gating variables  = size numberOfCompartments * Run time
    thrust::device_vector<float> d_m;
    thrust::device_vector<float> d_n;
    thrust::device_vector<float> d_h;
    // Storage for any electral stimulus into the system, size =  : size numberOfCompartments
    thrust::device_vector<float> d_I;
    // Gaba conductance of synapses
    thrust::device_vector<float> d_synapse_gaba_g;
    // Post synaptic current with duplicates for multiple connected compartments
    thrust::device_vector<float> d_synapse_current;
    // Post synaptic current without duplicates i.e. consolidating d_synapse_gaba_g
    thrust::device_vector<float> d_synapse_current_unique;
    // Matrix mapping synapse current to unqiue synapse current
    thrust::device_vector<float> d_synapse_gaba_mapping;
    // Currents from the estmd, gets added to the d_I current vector
    thrust::device_vector<float> d_estmd_input_current;

    // True if a compartment has spiked
    thrust::device_vector<bool> d_electrode_spike;
    // True if a compartment is in a refractory period
    thrust::device_vector<bool> d_refactory;
    // Spike record, size = numberOfCompartments * Run time
    thrust::device_vector<bool> d_spike_record;

    // Indexes of compartments where synapses start
    thrust::device_vector<int> d_synapse_gaba_pre_idx;
    // Indexes of compartments where synapses end
    thrust::device_vector<int> d_synapse_gaba_post_idx;
    // Same as previous but with no duplicates
    thrust::device_vector<int> d_synapse_gaba_post_idx_unique;
    // Compartments at which there are electrodes
    // electrodes are compartments at which we record spikes and voltage
    thrust::device_vector<int> d_electrode_idx;

    // Load morphology matrix onto device memory
    //
    // morph: inter-compartmental connection matrix, stored as a vector in
    //        row major form
    // numberOfCompartments: number of compartments in the morphology, length
    //                       of morph should be numberOfCompartments^2
    //
    // return: false if unable to allocate space for morphology otherwise true
    bool load_morphology(int * morph, int numberOfCompartments);

    // Reset the electrodes to false
    //
    // return : false if reset fails, true otherwise
    bool reset_electrodes();

    // Propagate intercompartmental current according to S matrix (cable theory)
    //
    // inputStart : input offset in the d_v matrix for time step
    // outputStart : output offset in the d_v matrix for time step
    //
    // return : false if cuBLAS fails otherwise true
    bool step_electrical_connections(int inputStart, int outputStart);

    // Add external stimulus currents to d_v matrix, includes currents from estmd input
    //
    // outputStart : output offset in the d_v matrix for time step
    //
    // return : false if cuBLAS fails otherwise true
    bool step_base_current(int outputStart);

    // Apply Hodgkin Huxley equations to voltage and gating variables
    // in order to propagate the model in time
    //
    // inputStart : input offset in the d_v matrix for time step
    // outputStart : output offset in the d_v matrix for time step
    void step_hodgkin_huxley(int inputStart, int inputEnd, int outputStart);

    // Check if any spikes have occurred using a threshold value and refractory boolean
    // for each compartment
    //
    // offset : offset in d_spike_record to index step in main run for loop
    void check_for_spikes(int offset);

    // Decay gaba conductances according to euler approximation for exponential function
    //
    // return : false if cublas fails otherwise true
    bool decay_gaba_conductance();

    // Reset synaptic conductances if the pre-synaptic compartment has spiked
    //
    // pre_synapse_compartments_spike: permutation iterator pointing to previous step
    //      spike boolean for pre-synaptic compartment
    //
    // returns: false if cuBLAS fails otherwise true
    void reset_spiked_conductance(perm_iterator_bool_int& pre_synapse_compartments_spike);

    // Add the influence of the updated  unique post-synaptic compartment input current to
    // the compartment voltage
    //
    // 1) Update post synaptic currents:
    //      POST SYNAPTIC CURRENT = GABA_CONDUCTANCE * (VOLTAGE - REVERSE POTENTIAL)
    // 2) Consolidate post synaptic currents:
    //      UNIQUE SYNAPTIC CURRENT = MAPPING *  POST SYNAPTIC CURRENT
    //      e.g. consider synapses 1->4 2->4
    //      d_synapse_current = [g_1*(V_4-E),g_2*(V_4-E)]^T
    //      d_synapse_current_unique = [1 1] * d_synapse_current
    //      d_V[4] += (dt * d_synapse_current_unique) / Cm
    //
    // post_synapse_compartments_v: post-synaptic voltages used to calculate (V(t)-E)
    // post_synapse_unique_compartments_v: unique post-synaptic voltages to add to
    bool step_synaptic_current(perm_iterator_float_int& post_synapse_compartments_v,
                               perm_iterator_float_int& post_synapse_unique_compartments_v);


    template<typename TYPE>
    void getArray(TYPE ** & destination, thrust::device_vector<TYPE>& source,
                  int begin, int end, int compartments[] , int numberToGet);

    // Print out a device vector to output stream s
    //
    // s: ostream to print to
    // vec: device vectors to print
    // offset: starting point of vector to print for
    // len: length of device vector
    // prec: precision of output
    template<typename TYPE>
    void print_d_vector(std::ostream &s, thrust::device_vector<TYPE>& vec,
        int offset, int len, int prec = 1, int center = 1);

    // Print out a device vector to file
    //
    // filename : ostream to print to
    // vec : device vectors to print
    template<typename TYPE>
    void save_d_vector(const char * filename, thrust::device_vector<TYPE>& vec,
        int m, int n);

    // Print a beutified device vector representing a matrix to output stream
    //
    // s: ostream to print to
    // vec: device vectors to print
    // m: height of matrix
    // n: width of matrix
    // len: length of device vector
    // prec: precision of output
    template<typename TYPE>
    void print_d_matrix(std::ostream &s, thrust::device_vector<TYPE>& vec,
        int m, int n, int prec = 1);

    // print an ascii spiketrian
    //
    // s : output stream
    // start : time to start the spike train at
    // end   : time to end the spike train at
    void print_ascii_spiketrain(std::ostream &s, int start, int end);

    // Print a plain device vector representing a matrix to output stream
    //
    // s: ostream to print to
    // vec: device vectors to print
    // m: height of matrix
    // n: width of matrix
    // len: length of device vector
    // prec: precision of output
    bool print(std::ostream &s, thrust::device_vector<float> &v, int T, int N,
            int prec = 5);

public:

    ~Cstmd1Sim();

    /*******************Functions exposed in by Cython wrapper****************/

    // Constructor
    //
    // morphology_data: inter-compartmental connection matrix, stored as a
    //                  vector in row major form
    // numberOfCompartments: number of compartments in the morphology, length
    //                       of morphology_data should be numberOfCompartments^2
    // _dt: time step length (ms)
    // debugMode: sets level of debugging statements:
    //              1 - Simple
    //              2 - Medium
    //              3 - Verbose
    //              4 - Simple + Print Electrode Spike Train
    //              5 - Print Electrode Spike Train
    // cstmd_buffer_size: max time that the simulator should run to (ms), used to reserve
    //        memory
    Cstmd1Sim(int *morphology_data, int numberOfCompartments, float _dt,
            int debugMode, int cstmd_buffer_size, std::map<std::string,float> parameters, int device = 0);

    // Load synapses onto device memory
    //
    // synapses: array of compartments idx [pre1.idx, post1.idx, pre2.idx,
    //           post2.idx..]
    // numberOfSynapses: number of synapses to add
    //
    // return: false if unable to allocate space for synapses otherwise true
    bool load_synapses(int * synapses, int numberOfSynapses, float g_max);



    // Load electrodes to record spikes
    //
    // electrodes: array of compartment idx to connect measurement electrodes
    // nummberOfElectrodes: number of electrodes to add (length of electrodes)
    bool load_electrodes(int * electrodes,int numberOfElectrodes);

    // Load currents from estmd module into d_estmd_input_current
    //
    // idx_data : compartment index for non-zero currents, valid if all are
    //            smaller than nCompartments
    // magnitude_data : magnitude of non-zero currents, valid if one exists for
    //                  each idx_data element
    // length : length of idx_data and magnitude_data, valid if non-zero and
    //          small than numberOfCompartments
    //
    // return : true if valid data successfully loaded and indices correct
    bool load_estmd_currents(int32_t * idx_data,float * magnitude_data,
        int length);

    // Save all voltage data to a file
    //
    // filename: file to write to
    //
    // returns: true if successully written else false
    bool save_data_to_file(const char *filename);

    // Run the simulator
    //
    // time_sec: the time to run to (ms)
    //
    // Examples:
    //      Run for 10ms then 10ms again
    //          .run(10)
    //          .run(20)
    //
    // returns: true if successully run otherwise false
    bool run(int time_sec);

    // Get the spikes from the connected electrodes
    //
    // spikes: empty vector to which is copied a bool of whether the electrode
    //         is connected to a compartment that spikes over the last run
    //
    // return: false if no electrodes are connected, true otherwise
    bool get_electrode_spikes(std::vector<bool>& spikes);

    // TODO: Write comment
    int get_voltage_size();
    void get_all_voltages(std::vector<float> &voltages);
    void get_recovery_variables(std::vector<float> &d_m,
				std::vector<float> &d_n,
				std::vector<float> &d_h);


    // Get the voltages for a given set of compartments over a time range
    //
    // voltages: pointer will be referenced to 2D array for resulting data.
    //           Memory will be assigned within the function
    // compartments: compartments to get voltages for
    // nCompartments: length of compartments (<= the number of compartments)
    // t_min: beginning of time range (>= 0)
    // t_max: end of time range (<= max time that the simulator has been run to)
    //
    // returns: false if invalid arguments or could not allocate memory on the
    //          heap for voltages
    bool get_voltages(float** &voltages, int compartments[], int nCompartments,
        int t_min, int t_max);


    // Get the spike train for a given set of compartments over a time range
    //
    // spikes: pointer will be referenced to 2D array for resulting data.
    //         Memory will be assigned within the function
    // compartments: compartments to get voltages for
    // nCompartments: length of compartments (<= the number of compartments)
    // t_min: beginning of time range (>= 0)
    // t_max: end of time range (<= max time that the simulator has been run to)
    //
    // returns: false if invalid arguments or could not allocate memory on the
    //          heap for spikes
    bool get_spikes(bool** &spikes, int compartments[], int nCompartments,
        int t_min, int t_max);

    // Fill the vector with random numbers distributed normally around
    //  a mean and with a standard deviation
    //
    //  v : previously initialised device_vector
    //  mean : mean of the random numbers
    //  stddev : standard deviation of the random numbers
    //
    //  returns: false if curand fails
    bool normalDistributionVector(thrust::device_vector<float> &v,float mean, float stddev);

    // use normalDistributionVector function to randomise the d_v vector
    //
    // mean : as in normalDistributionVector
    // stddev : as in normalDistributionVector
    //
    // returns : as in normalDistributionVector
    bool randomise_initial_voltages(float mean, float stddev);

    // enable the use of normalDistributionVector function to
    // randomise the d_I vector during each time step
    //
    // mean : as in normalDistributionVector
    // stddev : as in normalDistributionVector
    //
    void enable_randomise_currents(float mean, float stddev);

    /*******************Unused************************************************/

    // bool save_data_to_array(thrust::device_vector<float> &v, int** result, int T, int N);

    /*******************Functions is running from C++*************************/
    // bool load_morphology_from_file(const char* path);
    // bool load_synapses_from_file(const char* path);
    // bool load_electrodes_from_file(const char* path);
    // bool load_estmd_currents_from_file(const char* path);
};
