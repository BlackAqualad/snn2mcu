# NIR to C Code Generator for STM32H7

This tool converts Spiking Neural Networks (SNNs) from PyTorch/SNNTorch into optimized C code for STM32H7 microcontrollers using ARM CMSIS-DSP.

## Overview

The toolchain transforms SNNs through the Neural Intermediate Representation (NIR) format into embedded C code suitable for deployment on resource-constrained microcontrollers.

## Workflow

1. **Define SNN in PyTorch** ([SNNLIF.py](SNNLIF.py))
   - Create your SNN using SNNTorch library
   - Configure LIF neuron parameters (tau, threshold, etc.)
   - Define network architecture and weights

2. **Convert to NIR** ([SNNTorchToNIR_232.py](SNNTorchToNIR_232.py))
   - Exports the PyTorch model to NIR format
   - Preserves network structure and parameters
   - Outputs `.nir` file

3. **Inspect NIR (Optional)** ([inspect_nir.py](inspect_nir.py))
   - View the NIR graph structure
   - Verify nodes, edges, and parameters
   - Debug conversion issues

4. **Generate C Code** ([nir_to_c_generator.py](nir_to_c_generator.py))
   - Converts NIR to C source files (.c and .h)
   - Implements Q15 fixed-point arithmetic
   - Generates ARM CMSIS-DSP optimized code

## Features

- **Supported Neuron Types**: Leaky LIF, RLeaky (recurrent) LIF
- **Layer Types**: Fully connected, 1-to-1 connections
- **Fixed-Point Arithmetic**: Q15 format for efficient embedded execution
- **Hardware Optimization**: ARM CMSIS-DSP library integration

## Generated Files

- `lif_neuron_gen.h` - Header with network definitions and function prototypes
- `lif_neuron_gen.c` - Implementation with weight arrays and inference functions

## Usage Example

```python
# 1. Create and export your SNN
python SNNTorchToNIR_232.py

# 2. Inspect the NIR file (optional)
python inspect_nir.py

# 3. Generate C code
python nir_to_c_generator.py
```

## Requirements

- Python with nir, numpy, snntorch, torch
- ARM CMSIS-DSP library (for deployment)
- STM32H7 target (or compatible ARM Cortex-M7)

## Limitations

- Bias is not supported (must be zero)
- Recurrent connections only support 1-to-1 (diagonal) patterns
- Q15 fixed-point precision constraints

## Note on NIR Compatibility

The NIR descriptions used in this tool are slightly different from the official NIR format because recursion (recurrent connections) is not officially supported in the standard NIR specification. This implementation extends NIR to handle RLeaky neurons with 1-to-1 recurrent connections.
