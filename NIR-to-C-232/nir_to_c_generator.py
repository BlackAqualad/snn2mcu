# Copyright (C) 2025 Simone Delvecchi
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This work is part of the MSc Thesis: 
# "Optimization of Spiking Neural Networks execution on low-power microcontrollers."
# Politecnico di Torino.
#
# Thesis: https://webthesis.biblio.polito.it/38593/
# GitHub: https://github.com/BlackAqualad/snn2mcu

"""
NIR to C Code Generator for STM32H7 with ARM CMSIS-DSP
This program converts NIR network descriptions to embedded C code (.c and .h files)
following the LIF neuron implementation patterns.

Supports:
- Leaky LIF neurons
- RLeaky (recurrent) LIF neurons with 1-to-1 self-connections
- Fully connected feed-forward layers
- 1-to-1 connections (diagonal matrices stored as vectors)
- Q15 fixed-point arithmetic with ARM CMSIS-DSP

Weight Storage Pattern:
- NIR matrices are [outputs, inputs] = [neurons, inputs]
- C code uses INPUT-MAJOR order: [in0→n0, in0→n1, ..., in1→n0, in1→n1, ...]
- Conversion: weight.T.flatten() (transpose then flatten)
- For 1-to-1 connections: only diagonal values are stored as a vector
- For recurrent: only 1-to-1 is supported, stored as vector

Weight Formatting:
- Scientific notation with 4 decimal digits (e.g., 1.2345e-02f)
- Ensures portability across different architectures
- Matches base file format pattern

Limitations:
- Bias is NOT supported (must be zero in NIR file)
"""

import nir
import numpy as np
import os
from typing import Dict, List, Tuple

class NIRToCGenerator:
    def __init__(self, nir_file_path: str, output_prefix: str = "snn"):
        """
        Initialize the NIR to C code generator.
        
        Args:
            nir_file_path: Path to the .nir file
            output_prefix: Prefix for output files (default: "snn")
        """
        self.nir_graph = nir.read(nir_file_path)
        self.output_prefix = output_prefix
        self.scale_factor = 60.0  # Q15 scaling factor
        
        # Extract network architecture
        self.layers = []
        self.analyze_network()
    
    @staticmethod
    def _format_weight(value: float) -> str:
        """
        Format weight value in scientific notation with 4 decimal digits.
        This ensures portability across architectures and matches the base file format.
        
        Args:
            value: Weight value to format
            
        Returns:
            Formatted string like "1.2345e-02f" or "-3.4567e+00f"
        """
        # Use scientific notation with 4 decimal places
        return f"{value:.4e}f"
        
    def analyze_network(self):
        """Analyze the NIR graph to extract layer information."""
        print("Analyzing NIR graph structure...")
        
        # Build adjacency list
        adjacency = {}
        for src, dst in self.nir_graph.edges:
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(dst)
        
        # Find input size
        input_node = self.nir_graph.nodes.get('input')
        if input_node is None:
            raise ValueError("No input node found in NIR graph")
        
        # input_type is a dictionary with 'input' key containing numpy array
        if isinstance(input_node.input_type, dict):
            # Get the first value from the dictionary
            input_type_value = list(input_node.input_type.values())[0]
            if isinstance(input_type_value, np.ndarray):
                self.num_inputs = int(input_type_value[0])
            else:
                self.num_inputs = int(input_type_value)
        elif isinstance(input_node.input_type, np.ndarray):
            # If it's directly a numpy array
            self.num_inputs = int(input_node.input_type[0])
        else:
            # If it's a scalar
            self.num_inputs = int(input_node.input_type)
        
        # Traverse the graph to identify layers (avoiding recurrent loops)
        current = 'input'
        layer_idx = 0
        visited_lif_nodes = set()  # Track visited LIF nodes to avoid infinite loops
        
        while current in adjacency:
            next_nodes = adjacency[current]
            
            # Look for Affine (weight) node (skip recurrent connections)
            affine_node = None
            for node_name in next_nodes:
                node = self.nir_graph.nodes[node_name]
                if isinstance(node, nir.Affine) and 'rec' not in node_name:  # Skip recurrent nodes
                    affine_node = node_name
                    break
            
            if affine_node is None:
                break
            
            # Get the LIF node following the Affine
            if affine_node not in adjacency:
                break
                
            lif_node = None
            for node_name in adjacency[affine_node]:
                if isinstance(self.nir_graph.nodes[node_name], nir.LIF):
                    lif_node = node_name
                    break
            
            if lif_node is None or lif_node in visited_lif_nodes:
                break
            
            visited_lif_nodes.add(lif_node)  # Mark as visited
            
            # Check for recurrent connection
            has_recurrent = False
            recurrent_weights = None
            
            if lif_node in adjacency:
                for next_node in adjacency[lif_node]:
                    if 'rec' in next_node and isinstance(self.nir_graph.nodes[next_node], nir.Affine):
                        # Found recurrent connection
                        has_recurrent = True
                        recurrent_weights = self.nir_graph.nodes[next_node].weight
                        # Check if it's diagonal (1-to-1)
                        if not np.allclose(recurrent_weights, np.diag(np.diag(recurrent_weights))):
                            raise ValueError(f"Layer {layer_idx}: Only diagonal (1-to-1) recurrent connections are supported")
                        break
            
            # Extract layer information
            affine = self.nir_graph.nodes[affine_node]
            lif = self.nir_graph.nodes[lif_node]
            
            # Check connection type: fully connected or 1-to-1
            is_one_to_one = (affine.weight.shape[0] == affine.weight.shape[1] and 
                           np.allclose(affine.weight, np.diag(np.diag(affine.weight))))
            
            # Extract weights: if 1-to-1, just take diagonal; if fully connected, keep full matrix
            if is_one_to_one:
                weights_to_store = np.diag(affine.weight)  # Extract diagonal as 1D vector
            else:
                weights_to_store = affine.weight  # Keep full matrix
            
            # Check if bias is non-zero (not supported in embedded implementation)
            if not np.allclose(affine.bias, 0.0):
                print(f"WARNING: Layer {layer_idx} has non-zero bias values. Bias is NOT supported and will be ignored!")
            
            # Store per-neuron parameters (each neuron can have different values)
            layer_info = {
                'index': layer_idx,
                'affine_name': affine_node,
                'lif_name': lif_node,
                'num_inputs': affine.weight.shape[1],
                'num_neurons': affine.weight.shape[0],
                'weights': weights_to_store,  # Either 1D vector (1-to-1) or 2D matrix (fully connected)
                # NOTE: bias is NOT supported in the embedded C implementation
                'is_one_to_one': is_one_to_one,
                # Per-neuron parameters
                'tau': lif.tau,  # Array of tau values (one per neuron)
                'threshold': lif.v_threshold,  # Array
                'v_leak': lif.v_leak,  # Array
                'v_reset': lif.v_reset,  # Array
                'beta': np.exp(-1.0 / lif.tau),  # Convert tau to beta for each neuron
                'has_recurrent': has_recurrent,
                'recurrent_weights': np.diag(recurrent_weights) if has_recurrent else None  # 1D vector
            }
            
            # Check if all neurons in layer have same parameters (for optimization)
            layer_info['uniform_params'] = (
                np.all(lif.tau == lif.tau[0]) and
                np.all(lif.v_threshold == lif.v_threshold[0]) and
                np.all(lif.v_leak == lif.v_leak[0]) and
                np.all(lif.v_reset == lif.v_reset[0])
            )
            
            self.layers.append(layer_info)
            conn_type = "1-to-1" if is_one_to_one else "fully connected"
            param_type = "uniform" if layer_info['uniform_params'] else "per-neuron"
            print(f"Layer {layer_idx}: {layer_info['num_inputs']} -> {layer_info['num_neurons']} neurons, "
                  f"Connection: {conn_type}, Recurrent: {has_recurrent}, Params: {param_type}")
            
            layer_idx += 1
            current = lif_node
        
        print(f"Found {len(self.layers)} layers")
        
    def generate_header_file(self) -> str:
        """Generate the .h header file content."""
        h_content = f"""#ifndef LIF_NEURON_GEN_H
#define LIF_NEURON_GEN_H

#include <stdint.h>
#include "arm_math.h"

typedef struct {{
    q15_t threshold;     // Firing threshold in Q15
    q15_t reset_value;   // Reset potential in Q15
    q15_t membrane_potential; // Current membrane potential in Q15
    q15_t decay_factor;  // Precomputed beta (decay factor) in Q15
}} LIFNeuron;

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
"""
        return h_content
    
    def generate_c_file(self) -> str:
        """Generate the .c implementation file content."""
        
        # Generate layer definitions
        layer_defs = self._generate_layer_definitions()
        
        # Generate weight array definitions
        weight_defs = self._generate_weight_definitions()
        
        # Generate utility functions
        utility_funcs = self._generate_utility_functions()
        
        # Generate LIF neuron functions
        lif_funcs = self._generate_lif_functions()
        
        # Generate weight loading function
        weight_load_func = self._generate_weight_loading_function()
        
        # Generate SNN initialization
        snn_init = self._generate_snn_init()
        
        # Generate SNN timestep function
        snn_timestep = self._generate_snn_timestep()
        
        # Generate SNN reset function
        snn_reset = self._generate_snn_reset()
        
        c_content = f"""#include "stm32h7xx_hal.h"
#include "../Inc/lif_neuron_gen.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "../Inc/usart.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Network architecture from NIR
// Input size: {self.num_inputs}
// Layers: {len(self.layers)}
{self._generate_architecture_comment()}

{layer_defs}

{weight_defs}

{utility_funcs}

{lif_funcs}

{weight_load_func}

{snn_init}

{snn_timestep}

{snn_reset}
"""
        return c_content
    
    def _generate_architecture_comment(self) -> str:
        """Generate architecture description comment."""
        lines = []
        for layer in self.layers:
            rec_str = "with recurrent" if layer['has_recurrent'] else "no recurrent"
            conn_str = "1-to-1" if layer['is_one_to_one'] else "fully connected"
            param_str = "uniform params" if layer['uniform_params'] else "per-neuron params"
            lines.append(f"// Layer {layer['index']}: {layer['num_inputs']} -> {layer['num_neurons']} ({conn_str}, {rec_str}, {param_str})")
        return '\n'.join(lines)
    
    def _generate_layer_definitions(self) -> str:
        """Generate layer array definitions."""
        lines = [f"// Global variables for the SNN"]
        lines.append(f"#define NUM_INPUTS {self.num_inputs}")
        
        for i, layer in enumerate(self.layers):
            lines.append(f"#define NUM_NEURONS_LAYER{i+1} {layer['num_neurons']}")
        
        lines.append("")
        
        # Layer neuron arrays
        layer_arrays = ", ".join([f"layer{i+1}[NUM_NEURONS_LAYER{i+1}]" for i in range(len(self.layers))])
        lines.append(f"static LIFNeuron {layer_arrays};")
        
        # Spike arrays
        spike_arrays = ", ".join([f"l{i+1}_spikes[NUM_NEURONS_LAYER{i+1}]" for i in range(len(self.layers))])
        lines.append(f"static q7_t {spike_arrays};")
        
        # Previous spike arrays for recurrent layers
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"static q7_t l{i+1}_spikes_prev[NUM_NEURONS_LAYER{i+1}];")
        
        return '\n'.join(lines)
    
    def _generate_weight_definitions(self) -> str:
        """Generate weight array definitions."""
        lines = []
        
        # Weight arrays
        for i, layer in enumerate(self.layers):
            if layer['is_one_to_one']:
                # 1-to-1 connection: only diagonal values (vector)
                if i == 0:
                    lines.append(f"static q15_t weights{i+1}[NUM_INPUTS]; // 1-to-1 connection (vector)")
                else:
                    lines.append(f"static q15_t weights{i+1}[NUM_NEURONS_LAYER{i}]; // 1-to-1 connection (vector)")
            else:
                # Fully connected: full weight matrix
                if i == 0:
                    lines.append(f"static q15_t weights{i+1}[NUM_INPUTS*NUM_NEURONS_LAYER{i+1}]; // Fully connected")
                else:
                    lines.append(f"static q15_t weights{i+1}[NUM_NEURONS_LAYER{i}*NUM_NEURONS_LAYER{i+1}]; // Fully connected")
        
        # Recurrent weight arrays (always 1-to-1, stored as vectors)
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"static q15_t recurrent_weights{i+1}[NUM_NEURONS_LAYER{i+1}]; // Recurrent 1-to-1 (vector)")
        
        return '\n'.join(lines)
    
    def _generate_utility_functions(self) -> str:
        """Generate utility functions for USART printing."""
        return """// Utility functions for USART printing
void usart1_print(const char* str) {
    HAL_UART_Transmit(&huart1, (uint8_t*)str, strlen(str), 1000);
}

void print_float(const char* prefix, float_t value) {
    char buf[100];
    int int_part = (int)value;
    int frac_part = (int)((fabs(value) - fabs((float)int_part)) * 10000); // 4 decimal places
    
    // Handle negative numbers between -1 and 0
    if (value < 0.0f && int_part == 0) {
        snprintf(buf, sizeof(buf), "%s-%d.%04d\\r\\n", prefix, int_part, frac_part);
    } else {
        snprintf(buf, sizeof(buf), "%s%d.%04d\\r\\n", prefix, int_part, frac_part);
    }
    usart1_print(buf);
}"""
    
    def _generate_lif_functions(self) -> str:
        """Generate LIF neuron update functions."""
        return """
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
}"""
    
    def _generate_weight_loading_function(self) -> str:
        """Generate function to load weights from NIR data."""
        lines = [
            "void Load_NIR_Weights(void) {",
            f"    const float scale = {self.scale_factor}f;",
            ""
        ]
        
        # Generate weight arrays for each layer
        for i, layer in enumerate(self.layers):
            if layer['is_one_to_one']:
                # 1-to-1: weights is a 1D vector (diagonal values)
                lines.append(f"    // Layer {i+1} weights - 1-to-1 connection (vector of {layer['weights'].shape[0]} values)")
                lines.append(f"    float fc{i+1}_weights_vector[{layer['weights'].shape[0]}] = {{")
                
                weights_flat = layer['weights']  # Already 1D vector
                
                for idx, w in enumerate(weights_flat):
                    if idx % 8 == 0:
                        lines.append("        " if idx > 0 else "        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(weights_flat) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(weights_flat) - 1:
                        lines.append("")
                
                lines.append("    };")
            else:
                # Fully connected: weights is 2D matrix, flatten in input-major order
                # NIR format: [neurons, inputs] - need to TRANSPOSE to get [inputs, neurons]
                # Pattern: in0→n0, in0→n1, in0→n2, ..., in1→n0, in1→n1, ...
                lines.append(f"    // Layer {i+1} feedforward weights - fully connected ({layer['num_inputs']}x{layer['num_neurons']})")
                lines.append(f"    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]")
                lines.append(f"    float fc{i+1}_weights_vector[{layer['num_inputs'] * layer['num_neurons']}] = {{")
                
                # Transpose NIR matrix [neurons, inputs] to [inputs, neurons], then flatten
                weights_flat = layer['weights'].T.flatten()  # .T converts to input-major order
                
                for idx, w in enumerate(weights_flat):
                    if idx % 8 == 0:
                        lines.append("        " if idx > 0 else "        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(weights_flat) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(weights_flat) - 1:
                        lines.append("")
                
                lines.append("    };")
            lines.append("")
        
        # Generate recurrent weights (always 1-to-1, stored as vectors)
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    // Layer {i+1} recurrent weights - 1-to-1 (vector of {layer['num_neurons']} values)")
                lines.append(f"    float recurrent_weights_layer{i+1}[{layer['num_neurons']}] = {{")
                
                rec_weights = layer['recurrent_weights']  # Already 1D vector
                for idx, w in enumerate(rec_weights):
                    if idx % 8 == 0:
                        lines.append("        " if idx > 0 else "        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(rec_weights) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(rec_weights) - 1:
                        lines.append("")
                
                lines.append("    };")
                lines.append("")
        
        # Convert and store weights
        lines.append("    // Convert and store feedforward weights")
        for i, layer in enumerate(self.layers):
            if layer['is_one_to_one']:
                size = layer['weights'].shape[0]
            else:
                size = layer['num_inputs'] * layer['num_neurons']
            
            lines.append(f"    for (int i = 0; i < {size}; i++) {{")
            lines.append(f"        float scaled = fc{i+1}_weights_vector[i] / scale;")
            lines.append(f"        arm_float_to_q15(&scaled, &weights{i+1}[i], 1);")
            lines.append("    }")
            lines.append("")
        
        # Convert recurrent weights
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    // Convert recurrent weights (1-to-1)")
                lines.append(f"    for (int i = 0; i < {layer['num_neurons']}; i++) {{")
                lines.append(f"        float scaled = recurrent_weights_layer{i+1}[i] / scale;")
                lines.append(f"        arm_float_to_q15(&scaled, &recurrent_weights{i+1}[i], 1);")
                lines.append("    }")
                lines.append("")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _generate_snn_init(self) -> str:
        """Generate SNN initialization function."""
        lines = [
            "void SNN_Init(void) {",
            f"    const float scale = {self.scale_factor}f;",
            ""
        ]
        
        # Initialize neurons layer by layer with their specific parameters
        for i, layer in enumerate(self.layers):
            lines.append(f"    // Layer {i+1} initialization")
            
            if layer['uniform_params']:
                # All neurons have same parameters - optimize
                lines.append(f"    // Uniform parameters for all neurons")
                lines.append(f"    q15_t threshold_{i+1}, reset_value_{i+1}, decay_factor_{i+1};")
                lines.append(f"    float threshold_f_{i+1} = {self._format_weight(layer['threshold'][0])[:-1]} / scale;")  # Remove 'f' suffix
                lines.append(f"    float reset_value_f_{i+1} = {self._format_weight(layer['v_reset'][0])[:-1]} / scale;")
                lines.append(f"    float beta_{i+1} = {self._format_weight(layer['beta'][0])};")
                lines.append("")
                lines.append(f"    arm_float_to_q15(&threshold_f_{i+1}, &threshold_{i+1}, 1);")
                lines.append(f"    arm_float_to_q15(&reset_value_f_{i+1}, &reset_value_{i+1}, 1);")
                lines.append(f"    arm_float_to_q15(&beta_{i+1}, &decay_factor_{i+1}, 1);")
                lines.append("")
                lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
                lines.append(f"        LIFNeuron_Init(&layer{i+1}[i], threshold_{i+1}, reset_value_{i+1});")
                lines.append(f"        layer{i+1}[i].decay_factor = decay_factor_{i+1};")
                lines.append("    }")
            else:
                # Each neuron has different parameters
                lines.append(f"    // Per-neuron parameters")
                lines.append(f"    float thresholds_f[NUM_NEURONS_LAYER{i+1}] = {{")
                for idx, val in enumerate(layer['threshold']):
                    if idx % 8 == 0 and idx > 0:
                        lines.append("")
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(val)
                    if idx < len(layer['threshold']) - 1:
                        lines[-1] += ", "
                lines.append("    };")
                
                lines.append(f"    float reset_values_f[NUM_NEURONS_LAYER{i+1}] = {{")
                for idx, val in enumerate(layer['v_reset']):
                    if idx % 8 == 0 and idx > 0:
                        lines.append("")
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(val)
                    if idx < len(layer['v_reset']) - 1:
                        lines[-1] += ", "
                lines.append("    };")
                
                lines.append(f"    float betas_f[NUM_NEURONS_LAYER{i+1}] = {{")
                for idx, val in enumerate(layer['beta']):
                    if idx % 8 == 0 and idx > 0:
                        lines.append("")
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(val)
                    if idx < len(layer['beta']) - 1:
                        lines[-1] += ", "
                lines.append("    };")
                lines.append("")
                
                lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
                lines.append(f"        q15_t threshold_q15, reset_q15, beta_q15;")
                lines.append(f"        float thresh_scaled = thresholds_f[i] / scale;")
                lines.append(f"        float reset_scaled = reset_values_f[i] / scale;")
                lines.append(f"        arm_float_to_q15(&thresh_scaled, &threshold_q15, 1);")
                lines.append(f"        arm_float_to_q15(&reset_scaled, &reset_q15, 1);")
                lines.append(f"        arm_float_to_q15(&betas_f[i], &beta_q15, 1);")
                lines.append(f"        LIFNeuron_Init(&layer{i+1}[i], threshold_q15, reset_q15);")
                lines.append(f"        layer{i+1}[i].decay_factor = beta_q15;")
                lines.append("    }")
            
            lines.append("")
        
        # Load weights
        lines.append("    // Load weights from NIR")
        lines.append("    Load_NIR_Weights();")
        lines.append("")
        
        # Initialize previous spike arrays
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    arm_fill_q7(0, l{i+1}_spikes_prev, NUM_NEURONS_LAYER{i+1});")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _generate_snn_timestep(self) -> str:
        """Generate SNN timestep execution function."""
        lines = [
            "void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes) {"
        ]
        
        # Process each layer
        for i, layer in enumerate(self.layers):
            if i == 0:
                input_var = "input_spikes"
                input_size = "NUM_INPUTS"
            else:
                input_var = f"l{i}_spikes"
                input_size = f"NUM_NEURONS_LAYER{i}"
            
            is_one_to_one_flag = "1" if layer['is_one_to_one'] else "0"
            
            if layer['has_recurrent']:
                lines.append(f"    // Layer {i+1} with recurrent connections ({'1-to-1' if layer['is_one_to_one'] else 'fully connected'})")
                lines.append(f"    LIFNeuron_Layer_Update_Vectorized(layer{i+1}, {input_var}, weights{i+1}, {input_size}, NUM_NEURONS_LAYER{i+1}, l{i+1}_spikes, l{i+1}_spikes_prev, recurrent_weights{i+1}, {is_one_to_one_flag});")
            else:
                lines.append(f"    // Layer {i+1} (no recurrent, {'1-to-1' if layer['is_one_to_one'] else 'fully connected'})")
                lines.append(f"    LIFNeuron_Layer_Update_Vectorized_NoRecurrent(layer{i+1}, {input_var}, weights{i+1}, {input_size}, NUM_NEURONS_LAYER{i+1}, l{i+1}_spikes, {is_one_to_one_flag});")
            lines.append("")
        
        # Store current spikes as previous for recurrent layers
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    // Store spikes for layer {i+1} recurrent connections")
                lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
                lines.append(f"        l{i+1}_spikes_prev[i] = l{i+1}_spikes[i];")
                lines.append("    }")
                lines.append("")
        
        # Copy output
        last_layer_idx = len(self.layers)
        lines.append(f"    // Copy output spikes")
        lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{last_layer_idx}; i++) {{")
        lines.append(f"        output_spikes[i] = l{last_layer_idx}_spikes[i];")
        lines.append("    }")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _generate_snn_reset(self) -> str:
        """Generate SNN state reset function."""
        lines = [
            "void SNN_Reset_State(void) {"
        ]
        
        for i, layer in enumerate(self.layers):
            lines.append(f"    // Reset layer {i+1}")
            lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
            lines.append(f"        layer{i+1}[i].membrane_potential = layer{i+1}[i].reset_value;")
            lines.append(f"        l{i+1}_spikes[i] = 0;")
            if layer['has_recurrent']:
                lines.append(f"        l{i+1}_spikes_prev[i] = 0;")
            lines.append("    }")
            lines.append("")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def generate_files(self, output_dir: str = "."):
        """Generate the .h and .c files."""
        h_file = os.path.join(output_dir, f"lif_neuron_gen.h")
        c_file = os.path.join(output_dir, f"lif_neuron_gen.c")
        
        print(f"\nGenerating files...")
        
        # Generate header
        h_content = self.generate_header_file()
        with open(h_file, 'w') as f:
            f.write(h_content)
        print(f"✓ Generated: {h_file}")
        
        # Generate implementation
        c_content = self.generate_c_file()
        with open(c_file, 'w') as f:
            f.write(c_content)
        print(f"✓ Generated: {c_file}")
        
        # Generate usage example
        self._generate_usage_example(output_dir)
        
        print("\n" + "="*70)
        print("CODE GENERATION COMPLETE")
        print("="*70)
        self._print_summary()
    
    def _generate_usage_example(self, output_dir: str):
        """Generate example usage code."""
        example = f"""/*
 * Example usage of the generated SNN code
 * 
 * This shows how to use the NIR-generated SNN in your main.c
 * 
 * NOTE: Input must be binary spikes (q7_t: 0 or 1)
 */

#include "lif_neuron_gen.h"

void example_usage(void) {{
    // 1. Initialize the SNN (call once at startup)
    SNN_Init();
    
    // 2. Prepare input spikes (size: {self.num_inputs})
    // Input must be binary values: 0 (no spike) or 1 (spike)
    q7_t input_spikes[{self.num_inputs}];
    q7_t output_spikes[{self.layers[-1]['num_neurons']}];
    
    // Example: Set some input spikes
    for (int i = 0; i < {self.num_inputs}; i++) {{
        input_spikes[i] = (i % 2 == 0) ? 1 : 0; // Example pattern
    }}
    
    // 3. Run for multiple timesteps (e.g., 256 timesteps per sample)
    for (int t = 0; t < 256; t++) {{
        SNN_Run_Timestep(input_spikes, output_spikes);
        
        // Process output_spikes as needed
        // output_spikes contains {self.layers[-1]['num_neurons']} values (0 or 1)
    }}
    
    // 4. Reset state between samples
    SNN_Reset_State(); // (Only if testing from dataset, or if you want to reset the network state)
}}

/*
 * Network Architecture:
 * Input: {self.num_inputs} neurons (binary spikes: 0 or 1)
"""
        for i, layer in enumerate(self.layers):
            rec_str = "with 1-to-1 recurrent" if layer['has_recurrent'] else "feedforward only"
            conn_str = "1-to-1 connection" if layer['is_one_to_one'] else "fully connected"
            param_str = "uniform params" if layer['uniform_params'] else "per-neuron params"
            example += f" * Layer {i+1}: {layer['num_neurons']} neurons ({conn_str}, {rec_str}, {param_str})\n"
        
        example += f""" * Output: {self.layers[-1]['num_neurons']} neurons (binary spikes: 0 or 1)
 * 
 * Total parameters: {sum(l['num_inputs'] * l['num_neurons'] for l in self.layers)} feedforward weights
 * Recurrent parameters: {sum(l['num_neurons'] if l['has_recurrent'] else 0 for l in self.layers)} recurrent weights
 * 
 * Input Format:
 * - Binary spikes only: q7_t values must be 0 or 1
 * - If you have analog/float sensor data, convert to binary before calling SNN_Run_Timestep()
 * - Conversion method depends on your application (threshold, rate coding, etc.)
 */
"""
        
        example_file = os.path.join(output_dir, "example_usage.c")
        with open(example_file, 'w') as f:
            f.write(example)
        print(f"✓ Generated: {example_file}")
    
    def _print_summary(self):
        """Print generation summary."""
        print(f"\nNetwork Summary:")
        print(f"  Input size: {self.num_inputs}")
        print(f"  Number of layers: {len(self.layers)}")
        print(f"  Architecture: {self.num_inputs}", end="")
        for layer in self.layers:
            print(f" → {layer['num_neurons']}", end="")
        print()
        
        print(f"\nLayer Details:")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i+1}:")
            print(f"    Neurons: {layer['num_neurons']}")
            print(f"    Connection: {'1-to-1' if layer['is_one_to_one'] else 'Fully connected'}")
            print(f"    Parameters: {'Uniform' if layer['uniform_params'] else 'Per-neuron'}")
            if layer['uniform_params']:
                print(f"    Beta (decay): {layer['beta'][0]:.6f}")
                print(f"    Threshold: {layer['threshold'][0]:.6f}")
                print(f"    Reset: {layer['v_reset'][0]:.6f}")
            else:
                print(f"    Beta (decay): {layer['beta'][0]:.6f} to {layer['beta'][-1]:.6f}")
                print(f"    Threshold: {layer['threshold'][0]:.6f} to {layer['threshold'][-1]:.6f}")
                print(f"    Reset: {layer['v_reset'][0]:.6f} to {layer['v_reset'][-1]:.6f}")
            print(f"    Recurrent: {'Yes (1-to-1)' if layer['has_recurrent'] else 'No'}")
        
        total_params = sum(l['num_inputs'] * l['num_neurons'] for l in self.layers)
        total_rec_params = sum(l['num_neurons'] if l['has_recurrent'] else 0 for l in self.layers)
        print(f"\nTotal Parameters:")
        print(f"  Feedforward weights: {total_params}")
        print(f"  Recurrent weights: {total_rec_params}")
        print(f"  Total: {total_params + total_rec_params}")
        
        print(f"\nFeatures:")
        print(f"  ✓ Float input conversion supported")
        print(f"  ✓ Q15 fixed-point arithmetic (scale: {self.scale_factor})")
        print(f"  ✓ ARM CMSIS-DSP vectorized operations")
        print(f"  ✓ USART debug printing functions included")
        
        print(f"\nFiles generated successfully!")
        print(f"  - lif_neuron_gen.h")
        print(f"  - lif_neuron_gen.c")
        print(f"  - example_usage.c")


def main():
    """Main function to run the NIR to C generator."""
    import sys
    
    print("="*70)
    print("NIR TO C CODE GENERATOR")
    print("="*70)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        nir_file = sys.argv[1]
    else:
        nir_file = 'snntorch_braille7_model.nir'
    
    if not os.path.exists(nir_file):
        print(f"Error: NIR file '{nir_file}' not found!")
        print(f"\nUsage: python nir_to_c_generator.py [nir_file.nir]")
        return
    
    print(f"\nInput NIR file: {nir_file}")
    
    try:
        # Create generator
        generator = NIRToCGenerator(nir_file)
        
        # Generate files
        generator.generate_files()
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




