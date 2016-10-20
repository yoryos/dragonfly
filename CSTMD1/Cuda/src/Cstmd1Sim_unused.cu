//Save data to array
// bool Cstmd1Sim::save_data_to_array(thrust::device_vector<float> &v, int** result, int T, int N){
//
//     result = NULL;
//
//     try {
//
//       result = new int*[N];
//       for(int i = 0; i < T; i++) {
//         result[i] = new int[T];
//       }
//
//     } catch (std::bad_alloc& ba) {
//       return false;
//     }
//
//     for (int j = 0; j < T; ++j) {
//       for (int i = 0; i < N; ++i) {
//           result[i][j] = v[i + j * N];
//     	}
//     }
//
//     return true;
// }

//Load Electrodes from files
// bool Cstmd1Sim::load_electrodes_from_file(const char* path) {
//
//     int m_dim, n_dim;
//     int ** matrix_2D;
//
//     if(!load_matrix_from_file(path, matrix_2D, n_dim, m_dim)) {
//         std::cerr << "Could not load electrode file" << std::endl;
//     }
//     if(n_dim != 1) {
//         std::cerr << "Electrodes should be an nx1 array: (" << m_dim << "x" <<n_dim << " )" << std::endl;
//         return false;
//     }
//
//     // Set N to the dimension
//     nElectrodes = m_dim;
//     bool success = load_electrodes(matrix_2D, nElectrodes);
//
//     for(int i = 0; i < m_dim; ++i) {
//       delete [] matrix_2D[i];
//     }
//
//     return success;
// }

//Load synapses from file
// bool Cstmd1Sim::load_synapses_from_file(const char * path) {
//
//   int m_dim, n_dim;
//   int ** matrix_2D;
//
//   if(!load_matrix_from_file(path, matrix_2D, n_dim, m_dim)) {
//       std::cerr << "Could not load synpase file" << std::endl;
//       return false;
//   } else if(n_dim != 2) {
//       std::cerr << "Error cannot synapse not given in pairs found dimension: (" << m_dim << "x" <<n_dim << " )" << std::endl;
//       return false;
//   }
//
//   nSynapses = m_dim;
//   bool status = load_synapses(matrix_2D, nSynapses);
//
//   for(int i = 0; i <  m_dim; ++i) {
//     delete [] matrix_2D[i];
//   }
//
//   return status;
// }

//Load estmd current from file
// bool Cstmd1Sim::load_estmd_currents_from_file(const char * path) {
//   float ** data;
//   int length, width;
//   bool status = load_matrix_from_file(path,data,width,length);
//   return load_estmd_currents(data,length);
// }

//Load morphology from file
// bool Cstmd1Sim::load_morphology_from_file(const char* path) {
//
//   int m_dim, n_dim;
//   int ** matrix_2D;
//
//   if(!load_matrix_from_file(path, matrix_2D, n_dim, m_dim)) {
//       std::cerr << "Could not load morphology file" << std::endl;
//   }
//   if(m_dim != n_dim) {
//       std::cerr << "Error cannot use non-square morphology, got dimensions: (" << m_dim << "x" <<n_dim << " )" << std::endl;
//       return false;
//   }
//
//   // Set N to the dimension
//   nCompartments = m_dim;
//
//   bool success = load_morphology(matrix_2D,nCompartments);
//
//   for(int i = 0; i < m_dim; ++i) {
//     delete [] matrix_2D[i];
//   }
//
//   return success;
// }

//Consume estmd current
// bool Cstmd1Sim::add_estmd_current_to_d_I(int currentTimeStep) {
//   // Search for a estmd entry corresponding to the current time index
//   for(int i = 0; i < h_estmd_input_time.size(); ++i) {
//     if(h_estmd_input_time[i] == currentTimeStep) {
//         if (CUBLAS_STATUS_SUCCESS != cublasSaxpy(handle,
//                                                nCompartments,
//                                                &one,
//                                                thrust::raw_pointer_cast(&d_estmd_input_current[i*nCompartments]),
//                                                1,
//                                                thrust::raw_pointer_cast(&d_I[0]),
//                                                1)) {
//           std::cerr << "Base Current: Kernel execution error." << std::endl;
//           return false;
//         }
//
//         return true;
//     }
//     // No estmd current found for this time
//     if(h_estmd_input_time[i] > currentTimeStep) {
//       return true;
//     }
//   }
//
//   return true;
// }
//


// bool Cstmd1Sim::get_voltages(float** &voltages, int compartments[], int numberToGet, int t_min_sec, int t_max_sec) {
//
//     int t_min_step = (int) ((float) t_min_sec / (float) dt);
//     int t_max_step = (int) ((float) t_max_sec / (float) dt);
//
//     if(t_min_step < 0) {
//         std::cerr << "t_min too small" <<std::endl;
//         return false;
//     }
//     if(t_max_step > T) {
//         std::cerr << "t_max too big" <<std::endl;
//         return false;
//     }
//     if(numberToGet > nCompartments) {
//         std::cerr << "numberToGet too big" <<std::endl;
//         return false;
//     }
//
//     try {
//         voltages = new float*[numberToGet];
//         for(int i = 0; i < numberToGet; ++i) {
//             voltages[i] = new float[t_max_step - t_min_step];
//         }
//     } catch (std::bad_alloc &ba) {
//         std::cerr << "Could not allocate space on the heap for voltage retrieval" << std::endl;
//         return false;
//     }
//     std::cout << "Starting retrieval" << std::endl;
//     for(int t = t_min_step; t < t_max_step; ++t) {
//         for(int c = 0; c < numberToGet; ++c) {
//             voltages[c][t] = d_v[t * nCompartments + compartments[c]];
//         }
//     }
//     std::cout << "Finish retreival" << std::endl;
//
//     getArray(voltages, d_v, t_min_step, t_max_step, compartments, numberToGet);
//     return true;
//
// }