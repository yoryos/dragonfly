/******************************************************************************
 * Dragonfly Project 2016
 *
 *
 * @author: Dragonfly Project 2016 - Imperial College London
 *        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)
 *****************************************************************************/
#include "Cstmd1Sim.h"
#include "Cstmd1Sim_helper.h"

//Checks if element is in the fist max of a container with size
bool is_in(int element, std::vector<int> container) {

    for(std::vector<int>::iterator it = container.begin(); it != container.end(); ++it) {
        if (*it == element) {
            return true;
        }
    }
    return false;
}

// Ionic current propagation
struct Hh_propagate : public thrust::unary_function<Tuple4, Tuple4> {

    const float dt, epsilon, gbar_Na, gbar_K, gbar_l, Cm, E_Na, E_K, E_l;

    Hh_propagate(float _dt, float _epsilon,
                float _gbar_Na, float _gbar_K,float _gbar_l,
                float _Cm, float _E_Na,float _E_K,float _E_l) :
        dt(_dt), epsilon(_epsilon),
        gbar_Na(_gbar_Na), gbar_K(_gbar_K), gbar_l(_gbar_l),
        Cm(_Cm), E_Na(_E_Na), E_K(_E_K), E_l(_E_l)
        {}

    __host__ __device__
    Tuple4 operator()(const Tuple4& state) const {

      float v, m, n, h, alpha_n, alpha_m;

        thrust::tie(v, m, n, h) = state;

        if((v > 10.0 + epsilon) || (v < 10.0 - epsilon)) {
            alpha_n =  ALPHA_N(v);
        } else {
            alpha_n =  0.01;
        }

         if((v > 25.0 + epsilon) || (v < 25.0 - epsilon)) {
            alpha_m = ALPHA_M(v);
         } else {
            alpha_m = 0.1;
         }

        float m_next = m + dt * ((alpha_m * (1.0 - m)) - BETA_M(v) * m);
        float n_next = n + dt * ((alpha_n * (1.0 - n)) - BETA_N(v) * n);
        float h_next = h + dt * ((ALPHA_H(v) * (1.0 - h)) - BETA_H(v) * h);

        float g_Na = gbar_Na * powf(m_next, 3) * h_next;
        float g_K = gbar_K * powf(n_next, 4);
        float g_l = gbar_l;

        float dV = -(g_Na * (v - E_Na) + g_K * (v - E_K) + g_l * (v - E_l));

        float new_V = v + dt * dV / Cm;

        return Tuple4(new_V, m_next, n_next, h_next);
    }
};

// Synaptic current propogation
struct Synapse_current: public thrust::unary_function<Float2, float> {

    float reversePotential;
    Synapse_current(float _reversePotential) : reversePotential(_reversePotential) {}

    __host__ __device__
    float operator()(const Float2& conductance_voltage) const {

        float v, g, pd;
        thrust::tie(g, v) = conductance_voltage;
        // (E - V)
        pd = reversePotential - v;
        return g * pd;
    }
};

// Reset conductance to max
struct Reset_conductance : public thrust::binary_function<float,bool, float> {

    float resetVal;
    Reset_conductance(float _resetValue) : resetVal(_resetValue) {}

    __host__ __device__
    float operator()(float g, bool spike) {
        return (spike) ? resetVal : g;
    }
};

// Spike Detection
struct Spike_detection {

    float threshold;

    Spike_detection(float _threshold) : threshold(_threshold) {}

    template<typename Float1Bool2>
    __host__ __device__
    void operator()(Float1Bool2 triple) {
        // Triple contains <float voltage, bool refactory, bool spiked>

        float voltage = thrust::get<0>(triple);
        bool refactory = thrust::get<1>(triple);

        if (!refactory && voltage > threshold) {
            thrust::get<1>(triple) = true;  // set refractory to true
            thrust::get<2>(triple) = true;
        } else if (refactory && voltage <= threshold) {
            thrust::get<1>(triple) = false; // set refractory to false
        }
    }
};

//Euler step v += c * dV
struct Euler : public thrust::binary_function<float,float,float> {

    float coefficient;
    Euler(float _coefficient) : coefficient(_coefficient) {}

    __host__ __device__
    float operator()( float dV, float pre) {
        return pre + coefficient * dV;
    }
};

struct Nan_or_inf_check: public thrust::unary_function<float,bool> {
    __device__
    bool operator()(const float &x) const {
        if(isinf(x) || isnan(x)) {
            return true;
        } else {
            return false;
        }
    };
};

bool Cstmd1Sim::normalDistributionVector(thrust::device_vector<float> &v,float mean, float stddev) {
    // Completely fill the vetr with random numbers
    CURAND_CALL(curandGenerateNormal(gen,thrust::raw_pointer_cast(&v[0]),v.size(),mean,stddev));
    return true;
}

bool Cstmd1Sim::randomise_initial_voltages(float mean, float stddev) {
    CURAND_CALL(normalDistributionVector(d_v,mean,stddev));
    return true;
}

void Cstmd1Sim::enable_randomise_currents(float mean, float stddev) {
    // this will randomise the d_I vector at each time step
    randomise_currents = true;
    // set the mean of the random numbers
    d_I_mean = mean;
    // set the stddev of the random numbers
    d_I_stddev = stddev;
}

//Constructor
Cstmd1Sim::Cstmd1Sim(int *morphology_data, int numberOfCompartments, float _dt, int debugMode, int cstmd_buffer_size, std::map<std::string,float> parameters,int device) :
    dt(_dt),
    zero(0.0),
    one(1.0),
    T(0),
    nElectrodes(0),
    nSynapses(0),
    nCompartments(numberOfCompartments),
    debugStream(std::cerr),
    nan_or_inf_check(new Nan_or_inf_check()),
    randomise_currents(false),
    d_I_mean(0.0),
    d_I_stddev(0.0)
{

    Cm = parameters["cm"];
    dt_Cm = _dt / Cm;
    gaba_decay_coeff = -_dt / parameters["tau_gaba"];
    float sRa = parameters["sra"];
    float l = parameters["l"];
    float r = parameters["r"];
    float Ra          = (sRa*l)/(3.14*r*r);
    electricalCoeff = -dt / (Cm * Ra);
    stimulusMagnitude = parameters["estmd_gain"] * _dt / Cm;
    hh_propagate = new Hh_propagate(_dt,
                                    0.00001,
                                    parameters["gbar_na"],
                                    parameters["gbar_k"],
                                    parameters["gbar_l"],
                                    parameters["cm"],
                                    parameters["e_na"],
                                    parameters["e_k"],
                                    parameters["e_l"]
                                    );

    // enable randomisation of currents
    if(parameters["noise_stddev"] > 0.0) {
        enable_randomise_currents(0.0,parameters["noise_stddev"]);
    }

    // Try to avoid using the first graphic card as its typically the one that
    // gets used the most by others, for some reason using random cards > 0
    // created memory leaks, it would be nice to know why.
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if(nDevices > 1) {
        cudaSetDevice(device);
    } else {
        cudaSetDevice(0);
    }

    synapse_current = new Synapse_current(parameters["e_gaba"]);
    spike_detection = new Spike_detection(parameters["spike_threshold"]);
    euler = new Euler(dt_Cm);

    // typedef std::map<std::string, float>::iterator it_type;
    // for(it_type iterator = parameters.begin(); iterator != parameters.end(); iterator++) {
    //     debugStream << "[" << iterator->first << " " << iterator->second << "]" << std::endl;
    // }
    if(debugMode > 0) {
        debugStream << "--->>> Debug level: " << debugMode << std::endl;
    }
    setupDebug = debugMode == 4 || debugMode == 5;
    verboseDebug = debugMode == 3 || debugMode == 5;
    mediumDebug = debugMode == 2 || verboseDebug;
    simpleDebug = debugMode == 1 || verboseDebug || mediumDebug;

    int storage_size = ((int) ((float) cstmd_buffer_size / (float) dt) + 1) * nCompartments;

    if(setupDebug) {
        debugStream << " ==================Constructing Cuda Simulator==================" << std::endl;
        debugStream << "Attemping to reserve: " << storage_size << std::endl;
    }

    try {
        d_v.reserve(storage_size);
        d_m.reserve(storage_size);
        d_n.reserve(storage_size);
        d_h.reserve(storage_size);
        d_spike_record.reserve(storage_size);

    } catch (std::length_error &e) {
        std::cerr << "Could not reserve the memory for the large records" << std::endl;
    }


    if(!load_morphology(morphology_data,numberOfCompartments)) {
        std::cerr << "Could not load morphology" << std::endl;
        return;
    }

    try {

        // Resize, device vectors for initial values
        d_v.resize(storage_size, 0.0);
        thrust::fill_n(d_v.begin(), nCompartments, parameters["v_rest"]);

        d_m.resize(storage_size, 0.0);
        thrust::fill_n(d_m.begin(), nCompartments, M_INF(parameters["v_rest"]));

        d_n.resize(storage_size, 0.0);
        thrust::fill_n(d_n.begin(), nCompartments, N_INF(parameters["v_rest"]));

        d_h.resize(storage_size, 0.0);
        thrust::fill_n(d_h.begin(), nCompartments, H_INF(parameters["v_rest"]));

        d_I.resize(nCompartments, 0.0);
        d_refactory.resize(nCompartments, false);
        d_spike_record.resize(storage_size, false);

    } catch(thrust::system_error &e) {
        std::cerr << "Could not resize vectors in constructor: " << e.what() << std::endl;
        return;
    }

    //Stimulus
    //d_I[0] = 1.0;

    // Create cublas handle
    if(CUBLAS_STATUS_SUCCESS != cublasCreate(&handle)) {
        std::cerr << "Could not create cublas handle" << std::endl;
    }


    if(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        std::cerr << "Could not create curand generator" << std::endl;
    }

    srand(time(NULL));
    // NOTE CURRENTLY ALWAYS USE 0 SEED FOR DEBUGGING
    if(curandSetPseudoRandomGeneratorSeed(gen,0) != CURAND_STATUS_SUCCESS) {
        std::cerr << "Could not seed curand generator" << std::endl;
    }

    if(setupDebug) {
        debugStream << "(d_v): size: " << d_v.size() << " start: " << &(*d_v.begin()) << " end : " << &(*d_v.end()) <<  std::endl;
        debugStream << "(d_m): size: " << d_m.size() << " start: " << &(*d_m.begin()) << " end : " << &(*d_m.end())<< std::endl;
        debugStream << "(d_n): size: " << d_n.size() << " start: " << &(*d_n.begin()) << " end : " << &(*d_n.end())<< std::endl;
        debugStream << "(d_h): size: " << d_h.size() << " start: " << &(*d_h.begin()) << " end : " << &(*d_h.end())<< std::endl;
        debugStream << "(d_I): size: " << d_h.size() << " start: " << &(*d_I.begin()) << " end : " << &(*d_I.end())<< std::endl;
        debugStream << "(d_refactory): size: " << d_refactory.size() << " start: " << &(*d_refactory.begin()) << " end : " << &(*d_refactory.end())<< std::endl;
        debugStream << "(d_spike_record): size: " << d_spike_record.size() << " start: " << &(*d_spike_record.begin()) << " end : " << &(*d_spike_record.end())<< std::endl;
    }

    if(verboseDebug) {
        debugStream << "gaba_decay_coeff = " << gaba_decay_coeff << " dt = " << dt << std::endl;
        debugStream << "d_v: ";
        print_d_vector(debugStream, d_v, 0, nCompartments);
        debugStream << "d_m: ";
        print_d_vector(debugStream, d_m, 0, nCompartments);
        debugStream << "d_n: ";
        print_d_vector(debugStream, d_n, 0, nCompartments);
        debugStream << "d_h: ";
        print_d_vector(debugStream, d_h, 0, nCompartments);
        debugStream << "d_I: ";
        print_d_vector(debugStream, d_I, 0, nCompartments);
    }

     if(setupDebug) {
        debugStream << " ==================Finished Cuda Simulator Construction==================" <<std::endl;
    }
}

Cstmd1Sim::~Cstmd1Sim() {

    // Destroy the Cublas handle
    cublasDestroy(handle);
    // Destroy the curand generator
    curandDestroyGenerator(gen);

    // Clean up the structures
    delete hh_propagate;
    delete synapse_current;
    delete euler;
    if (nSynapses > 0) {
        delete reset_conductance;
    }
}

//Load morphology onto device
bool Cstmd1Sim::load_morphology(int* morph, int numberOfCompartments) {

    if(setupDebug || simpleDebug) debugStream << "Loading morphology with " << numberOfCompartments << " compartments" << std::endl;

    nCompartments = numberOfCompartments;
    thrust::host_vector<float> h_InterCompartmentalConnections(nCompartments * nCompartments, 0);

    for (int j = 0; j < nCompartments; ++j) {
        for (int i = 0; i < nCompartments; ++i) {
            h_InterCompartmentalConnections[IDX2C(i, j, nCompartments)] = morph[IDX2C(j, i, nCompartments)];
        }
    }

    d_electrical_connections = h_InterCompartmentalConnections;

     if(setupDebug) {
        debugStream << "(d_electrical_connections): size: " << d_electrical_connections.size() << " start: "
                    << &(*d_electrical_connections.begin()) << " end : " << &(*d_electrical_connections.end()) <<  std::endl;
    }

    return true;
}

//Load synapses onto device
bool Cstmd1Sim::load_synapses(int * synapses, int numberOfSynapses, float g_max) {

    reset_conductance = new Reset_conductance(g_max);
    if(setupDebug || simpleDebug) debugStream << "Loading " << numberOfSynapses << " synapses" << std::endl;

    nSynapses = numberOfSynapses;

    thrust::host_vector<int> h_synapse_gaba_pre_idx(nSynapses);
    thrust::host_vector<int> h_synapse_post(nSynapses);

    std::vector<int> uniquePost;

    for(int i = 0; i < nSynapses; ++i) {
        h_synapse_gaba_pre_idx[i] = synapses[IDX2C(0,i,2)];
        h_synapse_post[i] = synapses[IDX2C(1,i,2)];
        if(!is_in(h_synapse_post[i], uniquePost)) {
            uniquePost.push_back(h_synapse_post[i]);
        }
    }

    nUniquePostSynCompartments = uniquePost.size();
    thrust::host_vector<int> h_synapse_post_unique(nUniquePostSynCompartments);
    thrust::host_vector<float> h_synapse_unique_mapping(nUniquePostSynCompartments * nSynapses, 0.0);

    for(int i = 0; i < nUniquePostSynCompartments; ++i) {
        h_synapse_post_unique[i] = uniquePost[i];
    }

    for(int i = 0; i < nUniquePostSynCompartments; ++i) {
        for(int j = 0; j < nSynapses; ++j) {
            if(h_synapse_post_unique[i] == h_synapse_post[j])
                h_synapse_unique_mapping[IDX2C(i,j,nUniquePostSynCompartments)] = 1.0;
        }
    }

    try {

        //d_synapse_gaba_post_idx_unique.resize(nUniquePostSynCompartments);
        //d_synapse_gaba_mapping.resize(nUniquePostSynCompartments * nSynapses);
        //d_synapse_gaba_pre_idx.resize(nSynapses);
        //d_synapse_gaba_post_idx.resize(nSynapses);

        d_synapse_gaba_g.resize(nSynapses, 0.0);
        d_synapse_current.resize(nSynapses, 0.0);
        d_synapse_current_unique.resize(nUniquePostSynCompartments, 0.0);

    } catch (thrust::system_error &e) {
        std::cerr << "Could not allocate synapse space on device: " << e.what() << std::endl;
        return false;
    }

    d_synapse_gaba_post_idx_unique = h_synapse_post_unique;
    d_synapse_gaba_mapping = h_synapse_unique_mapping;
    d_synapse_gaba_pre_idx = h_synapse_gaba_pre_idx;
    d_synapse_gaba_post_idx = h_synapse_post;

    // std::ofstream file;
    // file.open("synapse_mapping.txt");
    // print_d_matrix(file, d_synapse_gaba_mapping,  nUniquePostSynCompartments, nSynapses);
    // file.close();

    if(setupDebug) {
        debugStream << "(d_synapse_gaba_post_idx_unique): size: " << d_synapse_gaba_post_idx_unique.size()
                    << " start: " << &(*d_synapse_gaba_post_idx_unique.begin()) << " end : " << &(*d_synapse_gaba_post_idx_unique.end()) <<  std::endl;
        debugStream << "(d_synapse_gaba_mapping): size: " << d_synapse_gaba_mapping.size()
                    << " start: " << &(*d_synapse_gaba_mapping.begin()) << " end : " << &(*d_synapse_gaba_mapping.end())<< std::endl;
        debugStream << "(d_synapse_gaba_pre_idx): size: " << d_synapse_gaba_pre_idx.size()
                    << " start: " << &(*d_synapse_gaba_pre_idx.begin()) << " end : " << &(*d_synapse_gaba_pre_idx.end())<< std::endl;
        debugStream << "(d_synapse_gaba_post_idx): size: " << d_synapse_gaba_post_idx.size()
                    << " start: " << &(*d_synapse_gaba_post_idx.begin()) << " end : " << &(*d_synapse_gaba_post_idx.end())<< std::endl;
        debugStream << "(d_synapse_gaba_g): size: " << d_synapse_gaba_g.size()
                    << " start: " << &(*d_synapse_gaba_g.begin()) << " end : " << &(*d_synapse_gaba_g.end())<< std::endl;
        debugStream << "(d_synapse_current): size: " << d_synapse_current.size()
                    << " start: " << &(*d_synapse_current.begin()) << " end : " << &(*d_synapse_current.end())<< std::endl;
        debugStream << "(d_synapse_current_unique): size: " << d_synapse_current_unique.size()
                    << " start: " << &(*d_synapse_current_unique.begin()) << " end : " << &(*d_synapse_current_unique.end())<< std::endl;
    }

    // for(int i= 0; i < d_synapse_gaba_post_idx_unique.size(); ++i) {
    //   int e = d_synapse_gaba_post_idx_unique[i];
    //   if(e==7165) {
	// std::cout << e << " !!!!!!!!!!!!!!!!!!!!!!!!! " << i << std::endl;
    //   }
    // }
    // for(int i= 0; i < d_synapse_gaba_post_idx.size(); ++i) {
    //   int e = d_synapse_gaba_post_idx[i];
    //   if(e==7165) {
	// std::cout << e << " !!!!!!!!!!!!!!!!!!!!!!!!! non unique index " << i << std::endl;
    //   }
    // }

    if(verboseDebug) {
        debugStream << "Presynaptic compartments idx: ";
        print_d_vector(debugStream, d_synapse_gaba_pre_idx, 0,  nSynapses);
        debugStream << "Postsynaptic compartments idx: ";
        print_d_vector(debugStream, d_synapse_gaba_post_idx, 0,  nSynapses);
        debugStream << "Postsynaptic compartments unique idx: ";
        print_d_vector(debugStream, d_synapse_gaba_post_idx_unique, 0,  nUniquePostSynCompartments);
        debugStream << "Gaba Mapping ";
        print_d_matrix(debugStream, d_synapse_gaba_mapping, nUniquePostSynCompartments,nSynapses);
        debugStream << "Synapse conductances: ";
        print_d_vector(debugStream, d_synapse_gaba_g, 0,  nSynapses);
        debugStream << "Syaptic currents: ";
        print_d_vector(debugStream, d_synapse_current, 0,  nSynapses);
        debugStream << "Unique synaptic currents: ";
        print_d_matrix(debugStream, d_synapse_current_unique, 0, nUniquePostSynCompartments);
    }

    return true;
}

//Load electrodes
bool Cstmd1Sim::load_electrodes(int* electrodes,int numberOfElectrodes) {

    if(setupDebug || simpleDebug) debugStream << "Loading " << numberOfElectrodes << " electrodes" << std::endl;

    nElectrodes = numberOfElectrodes;
    thrust::host_vector<float> h_Electrodes(nElectrodes);

    for (int i = 0; i < nElectrodes; ++i) {
        h_Electrodes[i] = electrodes[i];
    }

    d_electrode_idx = h_Electrodes;

    try {

        d_electrode_spike.resize(nElectrodes);

    } catch (thrust::system::detail::bad_alloc &ba) {
        std::cerr << "Could not allocate electrode spike vector" << std::endl;
        return false;
    }

    if(verboseDebug) {
        debugStream << "Electrode idx: ";
        print_d_vector(debugStream, d_electrode_idx, 0, numberOfElectrodes);
    }

    if(setupDebug) {
        debugStream << "(d_electrode_idx): size: " << d_electrode_idx.size() << " start: "
                    << &(*d_electrode_idx.begin()) << " end : " << &(*d_electrode_idx.end()) <<  std::endl;
    }

    return true;
}

bool Cstmd1Sim::load_estmd_currents(int32_t * idx_data, float * current_data, int length) {


    // Check there is data to work with
    if(length == 0) {
        std::cerr << "Warning: empty estmd file." << std::endl;
        return false;
    }

    // Host vector
    thrust::host_vector<float> h_estmd_input_current(nCompartments, 0.0);
    int index;
    for(int i = 0; i < length; ++i) {
        index = idx_data[i];
        if(index < nCompartments) {
            h_estmd_input_current[index] = current_data[i];
        } else {
            std::cerr << "Warning: got estmd input index larger than number of compartments." << std::cerr;
        }
    }
    // Copy
    d_estmd_input_current = h_estmd_input_current;


    if (CUBLAS_STATUS_SUCCESS != cublasSaxpy(handle,
            nCompartments,
            &one,
            thrust::raw_pointer_cast(&d_estmd_input_current[0]),
            1,
            thrust::raw_pointer_cast(&d_I[0]),
            1)) {
        std::cerr << "Base Current: Kernel execution error." << std::endl;
        return false;
    }

    if(setupDebug) {
        debugStream << "(d_estmd_input_current): size: " << d_estmd_input_current.size() << " start: "
                    << &(*d_estmd_input_current.begin()) << " end : " << &(*d_estmd_input_current.end()) <<  std::endl;
    }

    return true;

}

bool Cstmd1Sim::reset_electrodes() {

    if(setupDebug) debugStream << "Resetting spike vector" << std::endl;
    if(nElectrodes > 0) {
        thrust::fill_n(d_electrode_spike.begin(), nElectrodes, false);
    }

    return true;
}

//Propogate Hodgkin Huxley
void Cstmd1Sim::step_hodgkin_huxley(int inputStart, int inputEnd, int outputStart) {

    if(inputEnd < inputStart) {
        inputEnd = inputStart + nCompartments;
    }
    // std::cerr << inputStart << " " << inputEnd << " " << outputStart << std::endl;
    try{

        thrust::transform(
            thrust::make_zip_iterator(make_tuple(d_v.begin() + inputStart, d_m.begin() + inputStart, d_n.begin() + inputStart, d_h.begin() + inputStart)),
            thrust::make_zip_iterator(make_tuple(d_v.begin() + inputEnd, d_m.begin() + inputEnd, d_n.begin() + inputEnd, d_h.begin() + inputEnd)),
            thrust::make_zip_iterator(make_tuple(d_v.begin() + outputStart, d_m.begin() + outputStart, d_n.begin() + outputStart, d_h.begin() + outputStart)),
            *hh_propagate);

    } catch (thrust::system_error &e) {
        std::cerr << "Some error happened during Hodgkin Huxey: " << e.what() << std::endl;
        exit(-1);
    }

}

//Propogate Electrical Connections
bool Cstmd1Sim::step_electrical_connections(int inputStart, int outputStart) {

    if(CUBLAS_STATUS_SUCCESS != cublasSsymv(handle,
                                            CUBLAS_FILL_MODE_LOWER,
                                            nCompartments,
                                            &electricalCoeff,
                                            thrust::raw_pointer_cast(&d_electrical_connections[0]),
                                            nCompartments,
                                            thrust::raw_pointer_cast(&d_v[inputStart]),
                                            1,
                                            &one,
                                            thrust::raw_pointer_cast(&d_v[outputStart]),
                                            1)) {
        std::cerr << "Electrical Synapse: Kernel execution error." << std::endl;
        return false;
    }

    return true;
}

//Propogate Base Current
bool Cstmd1Sim::step_base_current(int outputStart) {

    if (CUBLAS_STATUS_SUCCESS != cublasSaxpy(handle,
            nCompartments,
            &stimulusMagnitude,
            thrust::raw_pointer_cast(&d_I[0]),
            1,
            thrust::raw_pointer_cast(&d_v[outputStart]),
            1)) {
        std::cerr << "Base Current: Kernel execution error." << std::endl;
        return false;
    }
    return true;
}

//Check for spikes
void Cstmd1Sim::check_for_spikes(int offset) {

    try {

        thrust::for_each_n(thrust::make_zip_iterator(make_tuple(d_v.begin() + offset,
                           d_refactory.begin(),
                           d_spike_record.begin() + offset)),
                           nCompartments,
                           *spike_detection);

    } catch(std::bad_alloc e) {
        std::cerr << "Ran out of memory while checking for spikes" << std::endl;
        std::cerr << e.what() << std::endl;
        exit(-1);
    } catch(thrust::system_error e) {
        std::cerr << "Some other error happened while checking for spikes"  << std::endl;
        std::cerr << e.what() << std::endl;
        exit(-1);
    }


    thrust::permutation_iterator<D_bool_it, D_int_it> electrode_spike_it(d_spike_record.begin() + offset,
            d_electrode_idx.begin());

    if(nElectrodes > 0) {
        thrust::transform(d_electrode_spike.begin(),
                          d_electrode_spike.end(),
                          electrode_spike_it,
                          d_electrode_spike.begin(),
                          logical_or);
    }

}

//Reset conductances
void Cstmd1Sim::reset_spiked_conductance(thrust::permutation_iterator<D_bool_it, D_int_it>& pre_synapse_compartments_spike) {

    thrust::transform(d_synapse_gaba_g.begin(),
                      d_synapse_gaba_g.end(),
                      pre_synapse_compartments_spike,
                      d_synapse_gaba_g.begin(),
                      *reset_conductance);

    //Reset conductance
    //thrust::replace_if(d_synapse_gaba_g.begin(), d_synapse_gaba_g.end(), pre_synapse_compartments_spike, _has_spiked, g_max);
    // thrust::transform(thrust::make_zip_iterator(make_tuple(d_synapse_gaba_g.begin(), pre_synapse_compartments_spike)),
    //                   thrust::make_zip_iterator(make_tuple(d_synapse_gaba_g.end() ,pre_synapse_compartments_spike + nSynapses)),
    //                   d_synapse_gaba_g.begin(),
    //                   *reset_conductance);

}

//Decay conductances
bool Cstmd1Sim::decay_gaba_conductance() {

    if(CUBLAS_STATUS_SUCCESS != cublasSaxpy(handle,
                                            nSynapses,
                                            &gaba_decay_coeff,
                                            thrust::raw_pointer_cast(&d_synapse_gaba_g[0]),
                                            1,
                                            thrust::raw_pointer_cast(&d_synapse_gaba_g[0]),
                                            1)) {
        std::cerr << "Electrical Synapse: Kernel execution error." << std::endl;
        return false;
    }

    return true;

}

bool Cstmd1Sim::step_synaptic_current(thrust::permutation_iterator<D_float_it, D_int_it>& post_synapse_compartments_v,
                                      thrust::permutation_iterator<D_float_it, D_int_it>& post_synapse_unique_compartments_v) {


    //Get current from conductances and voltages
    thrust::transform(thrust::make_zip_iterator(make_tuple(d_synapse_gaba_g.begin(), post_synapse_compartments_v)),
                      thrust::make_zip_iterator(make_tuple(d_synapse_gaba_g.end(), post_synapse_compartments_v + nSynapses)),
                      d_synapse_current.begin(),
                      *synapse_current);

    //std::cerr << "current1 "<<   d_synapse_current[258] << " current2 " << d_synapse_current[298] << std::endl;
    //Amalgamate currents from all incoming synapses
    if(CUBLAS_STATUS_SUCCESS != cublasSgemm(handle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            nUniquePostSynCompartments,
                                            1,
                                            nSynapses,
                                            &one, //dt_Cm
                                            thrust::raw_pointer_cast(&d_synapse_gaba_mapping[0]),
                                            nUniquePostSynCompartments,
                                            thrust::raw_pointer_cast(&d_synapse_current[0]),
                                            nSynapses,
                                            &zero, //one
                                            thrust::raw_pointer_cast(&d_synapse_current_unique[0]),
                                            nUniquePostSynCompartments)) {
        std::cerr << "Synaptic Current: Kernel execution error." << std::endl;
        return false;
    }

    //std::cerr << "voltage "<<  post_synapse_unique_compartments_v[249] << " current:" << d_synapse_current_unique[249] << "dt_Cm" << dt_Cm;

    //Add current
    thrust::transform(d_synapse_current_unique.begin(),
                      d_synapse_current_unique.end(),
                      post_synapse_unique_compartments_v,
                      post_synapse_unique_compartments_v,
                      *euler);
    //std::cerr << "after voltage "<<  post_synapse_unique_compartments_v[249] << std::endl;
    return true;
}

bool Cstmd1Sim::run(int time_sec) {

    dspace = 50;

    int b, e, old_T = T;
    T = (int) ((float) time_sec / (float) dt);

    if(!reset_electrodes()) return false;

    if(mediumDebug) debugStream << "Running from T: " << T << " = " << std::setprecision(3) << T * dt << "ms" << std::endl;

    double start = (double) std::clock();

    // False if issues encountered during loop
    bool status = true;

    if(setupDebug) debugStream << "Beginning Cuda Loop" << std::endl;

    int n = d_v.size()/nCompartments - 1;
    for (int i = old_T; i < T; ++i) {

        if(verboseDebug) debugStream << std::endl << "=====================Step " << i << "=====================" << std::endl;


        b = (i % n) * nCompartments;
        e = ((i + 1) % n) * nCompartments;
        debug_e = e;

        // Check for nans or infs
        // Do one more step in order to recieve as much debug info as possible
        if(setupDebug || verboseDebug || mediumDebug || simpleDebug) {
            if(thrust::any_of(d_v.begin()+b,d_v.begin()+e,*nan_or_inf_check)) {
                debugStream << "Detected non-finite voltage! Increasing debug level and quitting.";
                debugStream << std::endl;
                setupDebug = true;
                verboseDebug = true;
                mediumDebug = true;
                simpleDebug = true;
                status = false;
                // save_d_vector("d_v.dat",d_v,i+1,nCompartments);
                // save_d_vector("d_m.dat",d_m,i+1,nCompartments);
                // save_d_vector("d_n.dat",d_n,i+1,nCompartments);
                // save_d_vector("d_h.dat",d_h,i+1,nCompartments);
            }
        } else {
          if(thrust::any_of(d_v.begin()+b,d_v.begin()+e,*nan_or_inf_check)) {
            return false;
          }
        }

        if(verboseDebug) {
            debugStream << std::setw(dspace) << std::left << "VOLTAGE: carried from prev step: ";
            print_d_vector(debugStream, d_v, b, nCompartments, 5);
            debugStream << std::setw(dspace) << std::left << "M: carried from prev step: ";
            print_d_vector(debugStream, d_m, b, nCompartments, 5);
            debugStream << std::setw(dspace) << std::left << "N: carried from prev step: ";
            print_d_vector(debugStream, d_n, b, nCompartments, 5);
            debugStream << std::setw(dspace) << std::left << "H: carried from prev step: ";
            print_d_vector(debugStream, d_h, b, nCompartments, 5);
        }

        step_hodgkin_huxley(b, e, e);

        if(verboseDebug) {
            debugStream << std::setw(dspace) << std::left << "VOLTAGE: after Hodgkin Huxley: ";
            print_d_vector(debugStream, d_v, e, nCompartments, 5);
            debugStream << std::setw(dspace) << std::left << "M: after Hodgkin Huxley: ";
            print_d_vector(debugStream, d_m, e, nCompartments, 5);
            debugStream << std::setw(dspace) << std::left << "N: after Hodgkin Huxley: ";
            print_d_vector(debugStream, d_n, e, nCompartments, 5);
            debugStream << std::setw(dspace) << std::left << "H: after Hodgkin Huxley: ";
            print_d_vector(debugStream, d_h, e, nCompartments, 5);
        }

        step_electrical_connections(b, e);

        if(verboseDebug) {
            debugStream << std::setw(dspace) << std::left << "VOLTAGE: after propagating current: ";
            print_d_vector(debugStream, d_v, e, nCompartments, 5);
        }

        if(!step_base_current(e)) {
            return false;
        }
        if(verboseDebug) {
            debugStream << std::setw(dspace) << std::left << "VOLTAGE: after adding stimulating current: ";
            print_d_vector(debugStream, d_v, e, nCompartments, 5);
        }


        thrust::permutation_iterator<D_bool_it, D_int_it> pre_synapse_compartments_spike(d_spike_record.begin() + b, d_synapse_gaba_pre_idx.begin());
        thrust::permutation_iterator<D_float_it, D_int_it> post_synapse_compartments_v(d_v.begin() + b, d_synapse_gaba_post_idx.begin());
        thrust::permutation_iterator<D_float_it, D_int_it> post_synapse_unique_compartments_v(d_v.begin() + e, d_synapse_gaba_post_idx_unique.begin());


        if(nSynapses > 0) {
            step_synaptic_current(post_synapse_compartments_v, post_synapse_unique_compartments_v);
            if(verboseDebug) {
                debugStream << std::setw(dspace) << std::left << "VOLTAGE: after adding synaptic current: ";
                print_d_vector(debugStream, d_v, e, nCompartments, 5);
            }
        }

        check_for_spikes(e);
        if(verboseDebug) {
            debugStream << std::setw(dspace) << std::left << "SPIKE: ";
            print_d_vector(debugStream, d_spike_record, e, nCompartments,1);
        }

        if(nSynapses > 0) {

            decay_gaba_conductance();
            if(verboseDebug) {
                debugStream  << std::setw(dspace) << std::left << "CONDUCTANCE: after decaying conductance: ";
                print_d_vector(debugStream, d_synapse_gaba_g, 0, nSynapses, 5);
            }

            //Reset spiked conductance

            reset_spiked_conductance(pre_synapse_compartments_spike);

            if(verboseDebug) {
                debugStream << std::setw(dspace) << std::left << "CONDUCTANCE: after resetting conductance: ";
                print_d_vector(debugStream, d_synapse_gaba_g, 0, nSynapses, 5);
            }



        }

        if(status == false) {
            break;
        }
    }

    if(setupDebug) debugStream << "Finished Cuda Loop" << std::endl;

    debugStream << std::setprecision(6);
    double end =(double) std::clock();

    if(simpleDebug) debugStream << "Simulated (computation time) " << dt* (T-old_T) << "ms in " <<  1000 * ((end - start) / CLOCKS_PER_SEC) << "ms" << std::endl;

    if(randomise_currents) {
        status = normalDistributionVector(d_I,d_I_mean,d_I_stddev);
    } else {
        thrust::fill_n(d_I.begin(), nCompartments, 0.0);
    }

    thrust::fill_n(d_spike_record.begin(), d_spike_record.size(), false);

    return status;
}

bool Cstmd1Sim::get_electrode_spikes(std::vector<bool>& spikes) {

    if(nElectrodes == 0) {
        return false;
    }


    try {
        if(setupDebug || verboseDebug) {
            debugStream << "Retrieving electrode spikes start: " << &(*(d_electrode_spike.begin()))
                        << " end " << &(*(d_electrode_spike.end())) << std::endl;
            print_d_vector(debugStream, d_electrode_spike, 0, nElectrodes);
        }
        thrust::copy(d_electrode_spike.begin(), d_electrode_spike.end(), spikes.begin());
    } catch(thrust::system_error &e) {
        std::cerr << "d_electrode_spike size " << d_electrode_spike.size() << std::endl;
        std::cerr << "Container size " << spikes.size() << std::endl;
        std::cerr << "Number of electrodes " << nElectrodes << std::endl;
        std::cerr << "Could not retrieve electrode spikes: " << e.what() << std::endl;

        return false;
    }

    if(setupDebug || verboseDebug) debugStream << "Retrieved electrode spikes" << std::endl;
    return true;

};

int Cstmd1Sim::get_voltage_size()
{
    return d_v.size();
}

void Cstmd1Sim::get_all_voltages(std::vector<float> &voltages) {

    thrust::copy(d_v.begin(), d_v.end(), voltages.begin());
}

void Cstmd1Sim::get_recovery_variables(std::vector<float> &m,
				                       std::vector<float> &n,
				                       std::vector<float> &h) {

    thrust::copy(d_m.begin(), d_m.end(), m.begin());
    thrust::copy(d_n.begin(), d_n.end(), n.begin());
    thrust::copy(d_h.begin(), d_h.end(), h.begin());
}

//Print out a device vector
template<typename TYPE>
void Cstmd1Sim::print_d_vector(std::ostream &s, thrust::device_vector<TYPE>& vec, int offset, int len, int prec, int center) {

    bool reduced_print = false;
    if(len > 15) {
        len = 15;
        reduced_print = true;
    }
    thrust::host_vector<TYPE> host = vec;
    s << "[";
    int start,end;
    if((int)(center - (float)len/2.0) <= 0) {
        start = 0;
        end = len;
    } else {
        s << " ... ";
        start = center - len/2;
        end = center + len/2;
    }

    for(int i = start; i < end; ++i) {
        s <<  std::setprecision(prec) << std::setw(prec + 3)  << std::setiosflags(std::ios::right) << host[i + offset] << " ";
    }
    if(reduced_print) {
        s << " ... ";
    }
    s << "]" << std::endl;
}

template<typename TYPE>
void Cstmd1Sim::save_d_vector(const char * filename, thrust::device_vector<TYPE>& vec, int m, int n) {
    thrust::host_vector<TYPE> h_vec = vec;
    std::ofstream file;
    file.open(filename);
    if (not file.is_open()) {
        debugStream << "Failed to write to " << filename << std::endl;
    }
    for(int r = 0; r < m; ++r) {
        for(int c = 0; c < n; ++c) {
            file << h_vec[IDX2C(r,c,m)] << " ";
        }
        file << std::endl;
    }
    file.close();
}

//Print out a device matrix
template<typename TYPE>
void Cstmd1Sim::print_d_matrix(std::ostream &s, thrust::device_vector<TYPE>& vec, int m, int n, int prec) {

    bool reduced_print = false;
    // if(m > 15 or n > 15) {
    //     m = 15;
    //     n = 15;
    //     reduced_print = true;
    // }

    s << "--------------------" << std::endl;
    for(int r = 0; r < m; ++r) {
        for(int c = 0; c < n; ++c) {
            s << std::setprecision(prec) << std::setw(prec + 3) << std::setiosflags(std::ios::right) << vec[IDX2C(r,c,m)] << " ";
        }
        if(reduced_print) {
            s << " ... ";
        }
        s << std::endl;
    }
    if(reduced_print) {
        for(int c = 0; c < n; ++c) {
            s << std::setw(prec + 3) << std::setfill('.') << " ";
        }
    }
    s << std::setw(prec + 3) << std::setfill(' ') << std::endl;
    s << "--------------------" << std::endl;
}


//Print out the spike train
void Cstmd1Sim::print_ascii_spiketrain(std::ostream &s, int start, int end) {

    int precision = std::abs(std::ceil(std::log10(dt)));
    int timeStringLength = std::floor(std::log10(T) + 1);
    int stepLength =  std::floor(std::log10(T * dt) + 1);
    bool fired;

    thrust::host_vector<bool> h_spike_record = d_spike_record;

    std::cerr << "Spike record:" << std::endl;
    for(int i = start; i < end; ++i) {
        fired = false;
        for(int j = 0; j < nCompartments; ++j) {
            fired = fired || h_spike_record[j + i * nCompartments];
        }
        if(fired) {
            s << "Time: " << std::setw(precision + stepLength + 2) << std::setiosflags(std::ios::left) << i * dt
              << " Step:" << std::setw(timeStringLength) << std::setiosflags(std::ios::left)  << i << " : ";
            for(int j = 0; j < nCompartments; ++j) {
                s << h_spike_record[j + i*nCompartments] << " ";
            }
            s << std::endl;
        }
    }
    s << std::endl;
}

/*
*****************************************UNUSED********************************************************
*/


bool Cstmd1Sim::get_spikes(bool ** &spikes, int compartments[], int numberToGet, int t_min_sec, int t_max_sec) {

    int t_min_step = (int) ((float) t_min_sec / (float) dt);
    int t_max_step = (int) ((float) t_max_sec / (float) dt);

    if(t_min_step < 0) {
        std::cerr << "t_min too small" <<std::endl;
        return false;
    }
    if(t_max_step > T) {
        std::cerr << "t_max too big" <<std::endl;
        return false;
    }
    if(numberToGet > nCompartments) {
        std::cerr << "numberToGet too big" <<std::endl;
        return false;
    }

    try {
        spikes = new bool*[numberToGet];
        for(int i = 0; i < numberToGet; ++i) {
            spikes[i] = new bool[t_max_step - t_min_step];
        }
    } catch (std::bad_alloc &ba) {
        std::cerr << "Could not allocate space on the heap for spike retrieval" << std::endl;
        return false;
    }

    getArray(spikes, d_spike_record, t_min_step, t_max_step, compartments, numberToGet);
    return true;
}

template<typename TYPE>
void Cstmd1Sim::getArray(TYPE ** & destination, thrust::device_vector<TYPE>& source, int begin, int end, int compartments[] , int numberToGet) {

    std::cout << "Starting retrieval" << std::endl;
    for(int t = begin; t < end; ++t) {
        for(int c = 0; c < numberToGet; ++c) {
            destination[c][t] = source[t * nCompartments + compartments[c]];
        }
    }
    std::cout << "Finish retreival" << std::endl;
}

bool Cstmd1Sim::save_data_to_file(const char *filename) {

    std::ofstream file(filename);
    if(!file || !print(file, d_v, T + 1, nCompartments)) {
        return false;
    }

    file.close();
    return true;
}

bool Cstmd1Sim::print(std::ostream &s, thrust::device_vector<float> &v, int T, int N, int prec) {

    for (int j = 0; j < T; ++j) {
        for (int i = 0; i < N; ++i) {
            if(s) {
                s << std::fixed << std::setprecision(prec) << v[i + j * N];
                if (i != N - 1) s << " ";
            } else {
                std::cerr << "Stream printing failed" << std::endl;
                return false;
            }
        }
        s << std::endl;
    }
    return true;
}
