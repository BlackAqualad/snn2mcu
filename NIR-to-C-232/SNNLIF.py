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

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Parameters 
NUM_IN = 2
NUM_L1 = 2
NUM_L2 = 3
NUM_L3 = 2

tau = 10.0
threshold = 15.0  
reset_value = 0.0  
beta = torch.exp(torch.tensor(-1.0 / tau))  # decay factor


weights1 = [12.6005, 14.1253]  
weights2 = [16.5873, 12.0087, 15.8634, 2.3923, 17.9656, 10.3427]
weights3 = [18.0123, 18.8139, 18.3371, 5.5851, 2.0452, 14.9224]

# Network definition
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_IN, NUM_L1, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, 
                             reset_mechanism="zero", spike_grad=surrogate.fast_sigmoid())
        
        self.fc2 = nn.Linear(NUM_L1, NUM_L2, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, 
                             reset_mechanism="zero", spike_grad=surrogate.fast_sigmoid())
        
        self.fc3 = nn.Linear(NUM_L2, NUM_L3, bias=False)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, 
                             reset_mechanism="zero", spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, mem1, mem2, mem3):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        
        return spk3, mem1, mem2, mem3

# Initialize network
net = SimpleSNN()

# Load weights exactly as in C code
with torch.no_grad():
    # First layer (diagonal weights)
    w1_mat = torch.zeros((NUM_L1, NUM_IN))
    for i, w in enumerate(weights1):
        w1_mat[i, i] = w 
    net.fc1.weight.copy_(w1_mat)
    
    # Second layer - match C ordering: [in0->out0, in0->out1, in0->out2, in1->out0, in1->out1, in1->out2]
    # Reshape to (NUM_L2, NUM_L1) to match PyTorch's (out_features, in_features) format
    w2_mat = torch.zeros((NUM_L2, NUM_L1))
    for i in range(NUM_L1):  # Input neurons
        for j in range(NUM_L2):  # Output neurons
            idx = i * NUM_L2 + j
            w2_mat[j, i] = weights2[idx] 
    net.fc2.weight.copy_(w2_mat)
    
    # Third layer - match C ordering: [in0->out0, in0->out1, in1->out0, in1->out1, in2->out0, in2->out1]
    # Reshape to (NUM_L3, NUM_L2) to match PyTorch's (out_features, in_features) format
    w3_mat = torch.zeros((NUM_L3, NUM_L2))
    for i in range(NUM_L2):  # Input neurons
        for j in range(NUM_L3):  # Output neurons
            idx = i * NUM_L3 + j
            w3_mat[j, i] = weights3[idx] 
    net.fc3.weight.copy_(w3_mat)

# Simulation
print("SNN Simulation Starting...\n")

# Initialize membrane potentials
mem1 = torch.full((1, NUM_L1), reset_value)
mem2 = torch.full((1, NUM_L2), reset_value)
mem3 = torch.full((1, NUM_L3), reset_value)

# Input spikes 
inputs = torch.tensor([
    [1, 0],  # timestep 1: neuron 0 fires
    [0, 1],  # timestep 2: neuron 1 fires
    [1, 0],  # timestep 3: neuron 0 fires
    [0, 1],  # timestep 4: neuron 1 fires
], dtype=torch.float)

for t in range(4):
    print(f"Timestep {t+1}:")
    print("Input Spikes:", inputs[t].tolist())
    
    # Forward pass
    output, mem1, mem2, mem3 = net(inputs[t], mem1, mem2, mem3)
    
    # Print membrane potentials and spikes
    print("Layer 1 Membrane Potentials:", [f"{v:.4f}" for v in mem1.squeeze().tolist()])
    print("Layer 2 Membrane Potentials:", [f"{v:.4f}" for v in mem2.squeeze().tolist()])
    print("Layer 3 Membrane Potentials:", [f"{v:.4f}" for v in mem3.squeeze().tolist()])
    print("Output Spikes:", [int(s) for s in output.squeeze().tolist()])

    print()


