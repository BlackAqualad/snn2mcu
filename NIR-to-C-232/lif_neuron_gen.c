// Copyright (c) 2025 Simone Delvecchio
// Licensed under the MIT License (see LICENSE file for details)
// Part of the "snn2mcu" project - MSc Thesis, Politecnico di Torino

#include "stm32h7xx_hal.h"
#include "../Inc/lif_neuron_gen.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "../Inc/usart.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Network architecture from NIR
// Input size: 2
// Layers: 3
// Layer 0: 2 -> 2 (1-to-1, no recurrent, uniform params)
// Layer 1: 2 -> 3 (fully connected, no recurrent, uniform params)
// Layer 2: 3 -> 2 (fully connected, no recurrent, uniform params)

// Global variables for the SNN
#define NUM_INPUTS 2
#define NUM_NEURONS_LAYER1 2
#define NUM_NEURONS_LAYER2 3
#define NUM_NEURONS_LAYER3 2

static LIFNeuron layer1[NUM_NEURONS_LAYER1], layer2[NUM_NEURONS_LAYER2], layer3[NUM_NEURONS_LAYER3];
static q7_t l1_spikes[NUM_NEURONS_LAYER1], l2_spikes[NUM_NEURONS_LAYER2], l3_spikes[NUM_NEURONS_LAYER3];

static q15_t weights1[NUM_INPUTS]; // 1-to-1 connection (vector)
static q15_t weights2[NUM_NEURONS_LAYER1*NUM_NEURONS_LAYER2]; // Fully connected
static q15_t weights3[NUM_NEURONS_LAYER2*NUM_NEURONS_LAYER3]; // Fully connected

// Utility functions for USART printing
void usart1_print(const char* str) {
    HAL_UART_Transmit(&huart1, (uint8_t*)str, strlen(str), 1000);
}

void print_float(const char* prefix, float_t value) {
    char buf[100];
    int int_part = (int)value;
    int frac_part = (int)((fabs(value) - fabs((float)int_part)) * 10000); // 4 decimal places
    
    // Handle negative numbers between -1 and 0
    if (value < 0.0f && int_part == 0) {
        snprintf(buf, sizeof(buf), "%s-%d.%04d\r\n", prefix, int_part, frac_part);
    } else {
        snprintf(buf, sizeof(buf), "%s%d.%04d\r\n", prefix, int_part, frac_part);
    }
    usart1_print(buf);
}


void LIFNeuron_Init(LIFNeuron* neuron, q15_t threshold, q15_t reset_value) {
    neuron->threshold = threshold;
    neuron->reset_value = reset_value;
    neuron->membrane_potential = reset_value;
    // decay_factor (beta) will be set in SNN_Init
}

void LIFNeuron_Layer_Update_Vectorized(LIFNeuron* neurons, const q7_t* input_spikes, 
                                     const q15_t* weights, uint16_t num_inputs, 
                                     uint16_t num_neurons, q7_t* output_spikes,
                                     const q7_t* recurrent_spikes, const q15_t* recurrent_weights,
                                     uint8_t is_one_to_one) {
    q15_t membrane_potentials[num_neurons];
    q15_t reset_values[num_neurons];
    q15_t decay_factors[num_neurons];
    q15_t thresholds[num_neurons];
    q15_t weighted_inputs[num_neurons];

    // Extract neuron parameters
    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    // Initialize weighted_inputs to zero
    arm_fill_q15(0, weighted_inputs, num_neurons);

    // Calculate weighted input currents (feedforward)
    if (is_one_to_one) {
        // 1-to-1 connection: weights are stored as a vector, each input connects to corresponding neuron
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                // For 1-to-1, weight vector: weights[i] corresponds to connection i->i
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        // Fully connected: each input connects to all neurons
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }
    
    // Add recurrent connections (self-connections from previous timestep, always 1-to-1)
    if (recurrent_spikes != NULL && recurrent_weights != NULL) {
        for (uint16_t i = 0; i < num_neurons; i++) {
            if (recurrent_spikes[i]) {
                // Recurrent weights are stored as vector: recurrent_weights[i] for neuron i's self-loop
                arm_add_q15(&weighted_inputs[i], &recurrent_weights[i], &weighted_inputs[i], 1);
            }
        }
    }

    // Vectorized membrane potential update: V = reset + (V - reset) * beta + weighted_input
    q15_t temp1[num_neurons], temp2[num_neurons], temp3[num_neurons];
    
    arm_sub_q15(membrane_potentials, reset_values, temp1, num_neurons);
    arm_mult_q15(temp1, decay_factors, temp2, num_neurons);
    arm_add_q15(reset_values, temp2, temp3, num_neurons);
    arm_add_q15(temp3, weighted_inputs, membrane_potentials, num_neurons);

    // Check for spikes and reset
    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            membrane_potentials[i] = reset_values[i];
        } else {
            output_spikes[i] = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];
    }
}

void LIFNeuron_Layer_Update_Vectorized_NoRecurrent(LIFNeuron* neurons, const q7_t* input_spikes, 
                                                  const q15_t* weights, uint16_t num_inputs, 
                                                  uint16_t num_neurons, q7_t* output_spikes,
                                                  uint8_t is_one_to_one) {
    q15_t membrane_potentials[num_neurons];
    q15_t reset_values[num_neurons];
    q15_t decay_factors[num_neurons];
    q15_t thresholds[num_neurons];
    q15_t weighted_inputs[num_neurons];

    // Extract neuron parameters
    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    // Initialize weighted_inputs to zero
    arm_fill_q15(0, weighted_inputs, num_neurons);

    // Calculate weighted input currents (no recurrent)
    if (is_one_to_one) {
        // 1-to-1 connection: weights are stored as a vector, each input connects to corresponding neuron
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                // For 1-to-1, weight vector: weights[i] corresponds to connection i->i
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        // Fully connected: each input connects to all neurons
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }

    // Vectorized membrane potential update
    q15_t temp1[num_neurons], temp2[num_neurons], temp3[num_neurons];
    
    arm_sub_q15(membrane_potentials, reset_values, temp1, num_neurons);
    arm_mult_q15(temp1, decay_factors, temp2, num_neurons);
    arm_add_q15(reset_values, temp2, temp3, num_neurons);
    arm_add_q15(temp3, weighted_inputs, membrane_potentials, num_neurons);

    // Check for spikes and reset
    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            membrane_potentials[i] = reset_values[i];
        } else {
            output_spikes[i] = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];
    }
}

void Load_NIR_Weights(void) {
    const float scale = 100.0f;

    // Layer 1 weights - 1-to-1 connection (vector of 2 values)
    float fc1_weights_vector[2] = {
        1.2601e+01f, 1.4125e+01f
    };

    // Layer 2 feedforward weights - fully connected (2x3)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    float fc2_weights_vector[6] = {
        1.6587e+01f, 1.2009e+01f, 1.5863e+01f, 2.3923e+00f, 1.7966e+01f, 1.0343e+01f
    };

    // Layer 3 feedforward weights - fully connected (3x2)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    float fc3_weights_vector[6] = {
        1.8012e+01f, 1.8814e+01f, 1.8337e+01f, 5.5851e+00f, 2.0452e+00f, 1.4922e+01f
    };

    // Convert and store feedforward weights
    for (int i = 0; i < 2; i++) {
        float scaled = fc1_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights1[i], 1);
    }

    for (int i = 0; i < 6; i++) {
        float scaled = fc2_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights2[i], 1);
    }

    for (int i = 0; i < 6; i++) {
        float scaled = fc3_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights3[i], 1);
    }

}

void SNN_Init(void) {
    const float scale = 100.0f;

    // Layer 1 initialization
    // Uniform parameters for all neurons
    q15_t threshold_1, reset_value_1, decay_factor_1;
    float threshold_f_1 = 1.5000e+01 / scale;
    float reset_value_f_1 = 0.0000e+00 / scale;
    float beta_1 = 9.0484e-01f;

    arm_float_to_q15(&threshold_f_1, &threshold_1, 1);
    arm_float_to_q15(&reset_value_f_1, &reset_value_1, 1);
    arm_float_to_q15(&beta_1, &decay_factor_1, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        LIFNeuron_Init(&layer1[i], threshold_1, reset_value_1);
        layer1[i].decay_factor = decay_factor_1;
    }

    // Layer 2 initialization
    // Uniform parameters for all neurons
    q15_t threshold_2, reset_value_2, decay_factor_2;
    float threshold_f_2 = 1.5000e+01 / scale;
    float reset_value_f_2 = 0.0000e+00 / scale;
    float beta_2 = 9.0484e-01f;

    arm_float_to_q15(&threshold_f_2, &threshold_2, 1);
    arm_float_to_q15(&reset_value_f_2, &reset_value_2, 1);
    arm_float_to_q15(&beta_2, &decay_factor_2, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        LIFNeuron_Init(&layer2[i], threshold_2, reset_value_2);
        layer2[i].decay_factor = decay_factor_2;
    }

    // Layer 3 initialization
    // Uniform parameters for all neurons
    q15_t threshold_3, reset_value_3, decay_factor_3;
    float threshold_f_3 = 1.5000e+01 / scale;
    float reset_value_f_3 = 0.0000e+00 / scale;
    float beta_3 = 9.0484e-01f;

    arm_float_to_q15(&threshold_f_3, &threshold_3, 1);
    arm_float_to_q15(&reset_value_f_3, &reset_value_3, 1);
    arm_float_to_q15(&beta_3, &decay_factor_3, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER3; i++) {
        LIFNeuron_Init(&layer3[i], threshold_3, reset_value_3);
        layer3[i].decay_factor = decay_factor_3;
    }

    // Load weights from NIR
    Load_NIR_Weights();

}

void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes) {
    // Layer 1 (no recurrent, 1-to-1)
    LIFNeuron_Layer_Update_Vectorized_NoRecurrent(layer1, input_spikes, weights1, NUM_INPUTS, NUM_NEURONS_LAYER1, l1_spikes, 1);

    // Layer 2 (no recurrent, fully connected)
    LIFNeuron_Layer_Update_Vectorized_NoRecurrent(layer2, l1_spikes, weights2, NUM_NEURONS_LAYER1, NUM_NEURONS_LAYER2, l2_spikes, 0);

    // Layer 3 (no recurrent, fully connected)
    LIFNeuron_Layer_Update_Vectorized_NoRecurrent(layer3, l2_spikes, weights3, NUM_NEURONS_LAYER2, NUM_NEURONS_LAYER3, l3_spikes, 0);

    // Copy output spikes
    for (int i = 0; i < NUM_NEURONS_LAYER3; i++) {
        output_spikes[i] = l3_spikes[i];
    }
}

void SNN_Reset_State(void) {
    // Reset layer 1
    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        layer1[i].membrane_potential = layer1[i].reset_value;
        l1_spikes[i] = 0;
    }

    // Reset layer 2
    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        layer2[i].membrane_potential = layer2[i].reset_value;
        l2_spikes[i] = 0;
    }

    // Reset layer 3
    for (int i = 0; i < NUM_NEURONS_LAYER3; i++) {
        layer3[i].membrane_potential = layer3[i].reset_value;
        l3_spikes[i] = 0;
    }

}

