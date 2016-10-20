/******************************************************************************
 * Dragonfly Project 2016
 *
 *
 * @author: Dragonfly Project 2016 - Imperial College London
 *        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)
 */

#ifndef HODGKIN_H
#define HODGKIN_H

// /*============================== HH Parameters ==============================*/
// const float V_rest      = 0.0;                  // mV
// const float Cm          = 1.0;                  // uF/cm2
// const float gbar_Na     = 120.0;                // mS/cm2
// const float gbar_K      = 36.0;                 // mS/cm2
// const float gbar_l      = 0.3;                  // mS/cm2
// const float E_Na        = 115.0;                // mV
// const float E_K         = -12.0;                // mV
// const float E_l         = 10.613;               // mV
//
// /*============================== Morphology Parameters ======================*/
// const float S           = 7.0;                  // number of compartments // is this used
// const float RA          = 0.1;                  // specific intracellular resistivity (kOhm*cm2)
// const float r           = 2e-4;                 // compartment radius (cm)
// const float l           = 0.00001;              // compartment length (cm)
// const float Ra          = (RA*l)/(3.14*r*r);    // intracellular resistance (kOhm*cm)
//
// /*============================== Synapse Parameters =========================*/
// const float tau_gaba    = 10.0;
// //const float g_max       = 0.05;
// const float E_gaba      = 80.0;   //Which sign is inhibitory
// const float gain        = 10.0;
// /*============================== Simulation Parameters ======================*/
//
// const float THRESHOLD   = 30.0;

#define HH_NA(v,g_Na) (g_Na * (v - E_Na))
#define HH_K(v,g_K) (g_K  * (v - E_K ))
#define HH_L(v,g_l) (g_l  * (v - E_l ))
#define HH(v,g_Na,g_K,g_l)  HH_NA(v,g_Na) + HH_K(v,g_K) + HH_L(v,g_l)

#define ALPHA_N(v) ( (0.01*(-v + 10))/(expf((-v + 10.0)/10.0) - 1.0) ) //Check v = 10
#define BETA_N(v) ( 0.125*expf(-v/80.0) )
#define N_INF(v) (ALPHA_N(v)/(ALPHA_N(v) + BETA_N(v)))

#define ALPHA_M(v) (0.1*(-v + 25.0))/( expf((-v + 25.0)/10.0) - 1.0) //Check v = 25
#define BETA_M(v) (4.0*expf(-v/18.0))
#define M_INF(v) (ALPHA_M(v)/(ALPHA_M(v) + BETA_M(v)))

#define ALPHA_H(v) ( 0.07*expf(-v/20.0) )
#define BETA_H(v) (1.0/( expf((-v + 30.0)/10.0) + 1.0))
#define H_INF(v) (ALPHA_H(v)/(ALPHA_H(v) + BETA_H(v)))

//Unused
//const float E_ampa      = 0.0;
//const float E_nmda      = 0.0;
//const float Mg          = 1.0;
//const float alpha       = 0.062;
//const float beta        = 3.57;
//const float tau_ampa    = 2.0;
//const float tau_nmda    = 80.0;

//#define G_GABA(d) (G_max * expf(-d / tau_gaba))
//#define G_AMPA(d) (G_max * expf(-d / tau_ampa))
//#define G_NMBA(d) (G_max * expf(-d / tau_nmda))
//#define V_NMDA(v) (1.0 / (1.0 + expf(-alpha * v) * Mg / beta))

#endif
