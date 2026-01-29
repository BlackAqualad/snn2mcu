"""
Convert SNNTorch model to NIR (Neuromorphic Intermediate Representation) format.
This script creates a .nir file with identical characteristics to the network in SNNBraille7_Test.py
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import nir
from snntorch.export_nir import export_to_nir
import numpy as np

# Load the trained model
tmp_model = torch.load('./retrained_snntorch_20251007_161214.pt', map_location=torch.device('cpu'))

# Parameters (match the network in SNNBraille7_Test.py)
NUM_IN = 12
NUM_L1 = 38
NUM_L2 = 7

threshold = tmp_model['lif1.threshold']

# Recurrent weights (self-loops)
recurrent_weights1 = tmp_model['lif1.V']

# Network definition with RLeaky neurons (same as SNNBraille7_Test.py)
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(NUM_IN, NUM_L1, bias=False)
        self.lif1 = snn.RLeaky(beta=tmp_model['lif1.beta'], threshold=threshold, 
                          reset_mechanism="zero", spike_grad=surrogate.fast_sigmoid(),
                          all_to_all=False, V=recurrent_weights1, learn_recurrent=False,
                          reset_delay=False)

        self.fc2 = nn.Linear(NUM_L1, NUM_L2, bias=False)
        self.lif2 = snn.Leaky(beta=tmp_model['lif2.beta'], threshold=threshold, 
                          reset_mechanism="zero", spike_grad=surrogate.fast_sigmoid(),
                          reset_delay=False)

    def forward(self, x):
        # Initialize hidden states at t=0
        spk1, mem1 = self.lif1.init_rleaky()
        mem2 = self.lif2.init_leaky()

        # Record output spikes and membrane potentials for all layers
        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []

        num_steps = x.size(0)  # x should be (time_steps, batch_size, features)

        # time-loop
        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, spk1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Record all layer outputs
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        # convert lists to tensors
        spk1_rec = torch.stack(spk1_rec)
        mem1_rec = torch.stack(mem1_rec)
        spk2_rec = torch.stack(spk2_rec)
        mem2_rec = torch.stack(mem2_rec)  

        return spk1_rec, mem1_rec, spk2_rec, mem2_rec

# Initialize network
net = SimpleSNN()

# Load weights exactly as in SNNBraille7_Test.py
with torch.no_grad():
    net.fc1.weight.copy_(tmp_model['fc1.weight'])
    net.fc2.weight.copy_(tmp_model['fc2.weight'])

print("Building NIR graph...")

# Extract parameters
beta1 = tmp_model['lif1.beta'].item()
beta2 = tmp_model['lif2.beta'].item()
threshold_val = threshold.item()

# Get weights as numpy arrays
w_fc1 = net.fc1.weight.detach().cpu().numpy()  # Shape: (NUM_L1, NUM_IN)
w_fc2 = net.fc2.weight.detach().cpu().numpy()  # Shape: (NUM_L2, NUM_L1)
w_recurrent1 = recurrent_weights1.detach().cpu().numpy()  # Shape: (NUM_L1,) - diagonal recurrent

# Create NIR nodes
nodes = {}

# Input node
nodes['input'] = nir.Input(input_type=np.array([NUM_IN]))

# First linear layer (fc1)
nodes['fc1'] = nir.Affine(weight=w_fc1, bias=np.zeros(NUM_L1))

# First LIF layer with recurrence (RLeaky)
# For LIF neurons: beta = exp(-dt/tau), so tau = -dt / ln(beta)
# Assuming dt = 1 (timestep), tau = -1 / ln(beta)
# v_leak is typically 0 for these models
# v_reset = 0.0 because reset_mechanism="zero"
tau1 = -1.0 / np.log(beta1)
nodes['lif1'] = nir.LIF(
    tau=np.ones(NUM_L1) * tau1,
    v_threshold=np.ones(NUM_L1) * threshold_val,
    v_leak=np.zeros(NUM_L1),
    v_reset=np.zeros(NUM_L1),  # Reset to 0 (reset_mechanism="zero")
    r=np.ones(NUM_L1)  # Resistance, typically 1.0 for each neuron
)

# Recurrent connection for layer 1
# NOTE (important): this node implements a per-neuron self-loop (diagonal recurrent
# weights) taken from the trained snntorch RLeaky layer (`lif1.V`).
# Conventions for downstream runtimes / converters:
#  - Node name `rec1` indicates a recurrent connection feeding back to `lif1`.
#  - The value carried on the rec edge MUST be from the previous timestep (t-1).
#  - The feedback signal assumed here is the LIF output spike train.
#    If your runtime expects membrane instead, change `source_signal` accordingly.
#  - We use an Affine with a diagonal weight matrix for compatibility and future
#    extensibility (dense recurrence / bias). If your backend supports an
#    elementwise Scale primitive, prefer that to save memory.
# Create a diagonal matrix from the recurrent weights
w_rec_full = np.diag(w_recurrent1)
nodes['rec1'] = nir.Affine(weight=w_rec_full, bias=np.zeros(NUM_L1))

# Second linear layer (fc2)
nodes['fc2'] = nir.Affine(weight=w_fc2, bias=np.zeros(NUM_L2))

# Second LIF layer (Leaky - no recurrence)
# beta = exp(-dt/tau), so tau = -dt / ln(beta)
tau2 = -1.0 / np.log(beta2)
nodes['lif2'] = nir.LIF(
    tau=np.ones(NUM_L2) * tau2,
    v_threshold=np.ones(NUM_L2) * threshold_val,
    v_leak=np.zeros(NUM_L2),
    v_reset=np.zeros(NUM_L2),  # Reset to 0 (reset_mechanism="zero")
    r=np.ones(NUM_L2)  # Resistance, typically 1.0 for each neuron
)

# Output node
nodes['output'] = nir.Output(output_type=np.array([NUM_L2]))

# Create edges (connections between nodes)
edges = [
    ('input', 'fc1'),           # Input to first linear layer
    ('fc1', 'lif1'),            # First linear to first LIF
    ('lif1', 'rec1'),           # LIF1 output to recurrent connection
    ('rec1', 'lif1'),           # Recurrent connection back to LIF1
    ('lif1', 'fc2'),            # First LIF to second linear layer
    ('fc2', 'lif2'),            # Second linear to second LIF
    ('lif2', 'output')          # Second LIF to output
]

# Create NIR graph
nir_graph = nir.NIRGraph(nodes=nodes, edges=edges)

# Save to file
output_filename = 'snntorch_braille7_model.nir'
nir.write(output_filename, nir_graph)

print(f"NIR graph saved to '{output_filename}'")
print("\nNetwork architecture:")
print(f"  Input: {NUM_IN} neurons")
print(f"  Layer 1 (RLeaky): {NUM_L1} neurons")
print(f"    - Beta: {beta1}")
print(f"    - Tau: {tau1:.4f}")
print(f"    - Threshold: {threshold_val}")
print(f"    - Recurrent: Yes (diagonal/self-loops)")
print(f"  Layer 2 (Leaky): {NUM_L2} neurons")
print(f"    - Beta: {beta2}")
print(f"    - Tau: {tau2:.4f}")
print(f"    - Threshold: {threshold_val}")
print(f"    - Recurrent: No")
print(f"  Output: {NUM_L2} neurons")
print("\nNIR conversion complete!")

# RLeaky is not yet supported in snntorch.export_nir
# sample_data = torch.randn(1, 12)  # Example input data
# nir_export = export_to_nir(net, sample_data)

# nir.write('snntorch_braille7_model_v2.nir', nir_export)
