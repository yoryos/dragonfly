'''
Cstmd1 Simulator Cython Wrapper

Description

__author__: Dragonfly Project 2016 - Imperial College London ({anc15, cps15, dk2015, gk513, lm1015,
zl4215}@imperial.ac.uk)

'''

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
# for bool type support
from libcpp cimport bool

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/Cstmd1Sim_helper.h":
  pass

cdef extern from "src/Cstmd1Sim_parameters.h":
  pass

cdef extern from "src/Cstmd1Sim.h":
    cdef cppclass C_Cstmd1Sim "Cstmd1Sim":
        C_Cstmd1Sim(np.int32_t *,int,float, int, np.int32_t,map[string,float],int)
        # void save_data(char *)
        bool run(int)
        void load_synapses(np.int32_t *,int,  float)
        void load_electrodes(np.int32_t *,int)
        void load_estmd_currents(np.int32_t *, np.float32_t *,int)
        bool get_electrode_spikes(vector[bool] &)
        void get_all_voltages(vector[float] &)
        void get_recovery_variables(vector[float] &,vector[float] &,vector[float] &)
        int get_voltage_size()
        bool randomise_initial_voltages(float, float)
        void enable_randomise_currents(float,float)

cdef class Cstmd1Sim:
    cdef C_Cstmd1Sim * cstmd1
    cdef int numberOfElectrodes

    def __cinit__(self,np.ndarray[ndim=2, dtype=np.int32_t] py_array, float dt, int debugMode, np.int32_t cstmd_buffer_size, dict parameters, int device):
        cdef map[string,float] cparameters = parameters
        cdef np.ndarray[ndim=2,dtype=np.int32_t,mode='c'] contig = np.ascontiguousarray(py_array)
        self.cstmd1 = new C_Cstmd1Sim(&contig[0,0],py_array.shape[0],dt, debugMode, cstmd_buffer_size,cparameters, device)

    # def pass_dict(self,dict d):
    #     cdef map[string,float] cdict = d
    #     print d["hello"]
    #     print cdict["hello"]
    #     print d["bye"]
    #     print cdict["bye"]
    #     return

    # Returns false if error otherwise returns spikes
    def run(self,int t):
        status = self.cstmd1.run(t)
        if(status == False):
            return False,[]
        else:
            return True,self.get_electrode_spikes()

    def load_synapses(self, np.ndarray[ndim=2, dtype=np.int32_t] py_array, float g_max = 0.05):
        cdef np.ndarray[ndim=2,dtype=np.int32_t,mode='c'] contig = np.ascontiguousarray(py_array)
        self.cstmd1.load_synapses(&contig[0,0],py_array.shape[0], g_max)

    def load_electrodes(self, np.ndarray[ndim=1, dtype=np.int32_t,mode="c"] py_array):
        cdef np.ndarray[ndim=1,dtype=np.int32_t,mode='c'] contig = np.ascontiguousarray(py_array)
        self.numberOfElectrodes = py_array.shape[0]
        self.cstmd1.load_electrodes(&contig[0],py_array.shape[0])

    def load_estmd_currents(self, np.ndarray[ndim=1, dtype=np.int32_t,mode="c"] idx, np.ndarray[ndim=1, dtype=np.float32_t,mode="c"] magnitude):
        cdef np.ndarray[ndim=1,dtype=np.int32_t,mode='c'] c_idx = np.ascontiguousarray(idx)
        cdef np.ndarray[ndim=1,dtype=np.float32_t,mode='c'] c_magnitude = np.ascontiguousarray(magnitude)
        self.cstmd1.load_estmd_currents(&c_idx[0],&c_magnitude[0],idx.shape[0])

    def get_electrode_spikes(self):
        cdef vector[bool] spikes = vector[bool](self.numberOfElectrodes)
        status = self.cstmd1.get_electrode_spikes(spikes)
        if(status != False):
            return spikes
        else:
            return []

    def get_voltages(self):
        cdef int N = self.cstmd1.get_voltage_size()
        cdef vector[float] voltages = vector[float](N)
        status = self.cstmd1.get_all_voltages(voltages)
        return voltages

    def get_recovery_variables(self):
        cdef int N = self.cstmd1.get_voltage_size()
        cdef vector[float] m = vector[float](N)
        cdef vector[float] n = vector[float](N)
        cdef vector[float] h = vector[float](N)
        status = self.cstmd1.get_recovery_variables(m,n,h)
        return m,n,h

    def randomise_initial_voltages(self,mean,stddev):
        return self.cstmd1.randomise_initial_voltages(mean,stddev)

    # def enable_randomise_currents(self,mean,stddev):
    #     self.cstmd1.enable_randomise_currents(mean,stddev)

    def __dealloc__(self):
         del self.cstmd1
