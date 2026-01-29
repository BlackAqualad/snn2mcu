/* 
 Copyright (C) 2025 Simone Delvecchio
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License.

 This work is part of the MSc Thesis: 
 "Optimization of Spiking Neural Networks execution on low-power microcontrollers."
 Politecnico di Torino.

 Thesis: https://webthesis.biblio.polito.it/38593/
 GitHub: https://github.com/BlackAqualad/snn2mcu
*/

#ifndef LIF_NEURON_GEN_H
#define LIF_NEURON_GEN_H

#include <stdint.h>
#include "arm_math.h"

typedef struct {
    q15_t threshold;     // Firing threshold in Q15
    q15_t reset_value;   // Reset potential in Q15
    q15_t membrane_potential; // Current membrane potential in Q15
    q15_t decay_factor;  // Precomputed beta (decay factor) in Q15
} LIFNeuron;

// Utility functions
void usart1_print(const char* str);
void print_float(const char* prefix, float_t value);

// LIF Neuron functions
void LIFNeuron_Init(LIFNeuron* neuron, q15_t threshold, q15_t reset_value);

// Layer update functions
void LIFNeuron_Layer_Update_Vectorized(LIFNeuron* neurons, const q7_t* input_spikes, 
                                     const q15_t* weights, uint16_t num_inputs, 
                                     uint16_t num_neurons, q7_t* output_spikes,
                                     const q7_t* recurrent_spikes, const q15_t* recurrent_weights,
                                     uint8_t is_one_to_one);

void LIFNeuron_Layer_Update_Vectorized_NoRecurrent(LIFNeuron* neurons, const q7_t* input_spikes, 
                                                  const q15_t* weights, uint16_t num_inputs, 
                                                  uint16_t num_neurons, q7_t* output_spikes,
                                                  uint8_t is_one_to_one);

// Weight loading function
void Load_NIR_Weights(void);

// SNN main functions
void SNN_Init(void);
void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes);
void SNN_Reset_State(void);

#endif // LIF_NEURON_GEN_H



