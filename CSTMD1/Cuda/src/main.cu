/******************************************************************************
 * Dragonfly Project 2016
 *
 *
 * @author: Dragonfly Project 2016 - Imperial College London
 *        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)
 */

#include "Cstmd1Sim_helper.h"
#include "Cstmd1Sim.h"



int main()
{
    // std::string simple_morphology = "../data/simple.dat";
    // std::string simple_synapses = "../data/synExample.dat";
    // std::string simple_estmd1 = "../data/estmd1SimpleStimulus.dat";
    // std::string simple_electrodes = "../data/simple_electrodes.dat";
    //
    // std::cerr << "Loading morphology" << std::endl;
    //
    // Cstmd1Sim test(simple_morphology.c_str(), 0.01);
    //
    // std::cerr << "Loaded morphology" << std::endl;
    //
    //
    // if(!test.load_estmd_currents_from_file(simple_estmd1.c_str())) {
    //     std::cerr << "Failed to load estmd data" <<std::endl;
    //   return false;
    // }
    // std::cerr << "Loaded currents" << std::endl;
    //
    //
    // if(!test.load_synapses_from_file(simple_synapses.c_str())) {
    //   std::cerr << "Failed to read" <<std::endl;
    //   return false;
    // }
    // std::cerr << "Loaded synapses" << std::endl;
    //
    // if(!test.load_electrodes_from_file(simple_electrodes.c_str())) {
    //     std::cerr << "Could not load electrodes" << std::endl;
    //     return false;
    // }
    // std::cout << "Loaded electrodes" << std::endl;
    //
    // int T = 100;
    // if(test.run(T)){
    //   std::cerr << "Ran successfully" << std::endl;
    // } else {
    //   std::cerr << "Failed" << std::endl;
    // }

    // std::list<bool> spikes(test.nElectrodes);
    // test.get_electrode_spikes(spikes);
    //
    // std::cout << "Electrode Spikes" << std::endl;
    // for(std::list<bool>::iterator it = spikes.begin(); it != spikes.end(); ++it) {
    //     std::cout << *it << " ";
    // }
    // std::cout << std::endl;

    //
    //  float** v;
    //  const int n = 2;
    //  int c[2] = {0,2};
    //
    // if(!test.get_voltages(v, c, n, 0, T))
    //   std::cerr << "Could not get voltages" << std::endl;
    //
    // bool ** s;
    //
    // if(!test.get_spikes(s, c, n, 0, T))
    //   std::cerr << "Could not get voltages" << std::endl;
    //
    // save_matrix("../data/large_spikes.dat", s, T/0.05, n);
    // save_matrix("../data/large_volt.dat", v, T/0.05, n);


    return 0;
}
