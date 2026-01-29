# Copyright (C) 2025 Simone Delvecchio
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
Convert the small SNN (2-2-3-2 architecture) to NIR format.
This script mirrors the style of `SNNTorchToNIR.py` but embeds the parameters
from `SNNLIF.py` and writes a .nir file.
"""

import numpy as np
import nir

# Network parameters (from SNNLIF.py)
NUM_IN = 2
NUM_L1 = 2
NUM_L2 = 3
NUM_L3 = 2

# LIF parameters
tau = 10.0
threshold = 15.0
v_leak = 0.0
v_reset = 0.0
r = 1.0

# Weights as in SNNLIF.py
weights1 = [12.6005, 14.1253]
weights2 = [16.5873, 12.0087, 15.8634, 2.3923, 17.9656, 10.3427]
weights3 = [18.0123, 18.8139, 18.3371, 5.5851, 2.0452, 14.9224]

# Build weight matrices matching PyTorch shapes used in SNNLIF.py
# fc1: (NUM_L1, NUM_IN) - diagonal weights
w_fc1 = np.zeros((NUM_L1, NUM_IN), dtype=float)
for i, w in enumerate(weights1):
    w_fc1[i, i] = w

# fc2: (NUM_L2, NUM_L1)
w_fc2 = np.zeros((NUM_L2, NUM_L1), dtype=float)
# SNNLIF orders weights2 as: [in0->out0, in0->out1, in0->out2, in1->out0, in1->out1, in1->out2]
for i in range(NUM_L1):
    for j in range(NUM_L2):
        idx = i * NUM_L2 + j
        w_fc2[j, i] = weights2[idx]

# fc3: (NUM_L3, NUM_L2)
w_fc3 = np.zeros((NUM_L3, NUM_L2), dtype=float)
# SNNLIF orders weights3 as: [in0->out0, in0->out1, in1->out0, in1->out1, in2->out0, in2->out1]
for i in range(NUM_L2):
    for j in range(NUM_L3):
        idx = i * NUM_L3 + j
        w_fc3[j, i] = weights3[idx]

# Compute beta from tau assuming dt=1: beta = exp(-dt/tau)
beta = float(np.exp(-1.0 / tau))
# Convert to time constant tau array for NIR's LIF (tau in same units)
tau1 = np.ones(NUM_L1) * tau
tau2 = np.ones(NUM_L2) * tau
tau3 = np.ones(NUM_L3) * tau

# Build NIR nodes
nodes = {}
nodes['input'] = nir.Input(input_type=np.array([NUM_IN]))

nodes['fc1'] = nir.Affine(weight=w_fc1, bias=np.zeros(NUM_L1))
nodes['lif1'] = nir.LIF(
    tau=tau1,
    v_threshold=np.ones(NUM_L1) * threshold,
    v_leak=np.ones(NUM_L1) * v_leak,
    v_reset=np.ones(NUM_L1) * v_reset,
    r=np.ones(NUM_L1) * r
)

nodes['fc2'] = nir.Affine(weight=w_fc2, bias=np.zeros(NUM_L2))
nodes['lif2'] = nir.LIF(
    tau=tau2,
    v_threshold=np.ones(NUM_L2) * threshold,
    v_leak=np.ones(NUM_L2) * v_leak,
    v_reset=np.ones(NUM_L2) * v_reset,
    r=np.ones(NUM_L2) * r
)

nodes['fc3'] = nir.Affine(weight=w_fc3, bias=np.zeros(NUM_L3))
nodes['lif3'] = nir.LIF(
    tau=tau3,
    v_threshold=np.ones(NUM_L3) * threshold,
    v_leak=np.ones(NUM_L3) * v_leak,
    v_reset=np.ones(NUM_L3) * v_reset,
    r=np.ones(NUM_L3) * r
)

nodes['output'] = nir.Output(output_type=np.array([NUM_L3]))

# Edges following the forward path
edges = [
    ('input', 'fc1'),
    ('fc1', 'lif1'),
    ('lif1', 'fc2'),
    ('fc2', 'lif2'),
    ('lif2', 'fc3'),
    ('fc3', 'lif3'),
    ('lif3', 'output')
]

nir_graph = nir.NIRGraph(nodes=nodes, edges=edges)

output_filename = 'snntorch_snn_232.nir'
nir.write(output_filename, nir_graph)

print(f"Wrote NIR graph to '{output_filename}'")
print("Network:")
print(f"  Input: {NUM_IN}")
print(f"  Layer1: {NUM_L1} (tau={tau}, threshold={threshold}, beta={beta:.6f})")
print(f"  Layer2: {NUM_L2} (tau={tau}, threshold={threshold})")
print(f"  Layer3: {NUM_L3} (tau={tau}, threshold={threshold})")



