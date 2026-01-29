# With Reset delay = False, it works fine, but I cannot see the values above threshold (as expected)
# reset_mechanism_val = 1 is reset to 0, 0 is reset with subtracting threshold
# for now no bias and no synaptic

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Load the dataset
tmp = torch.load('C:/Users/simon/Desktop/Polito/TESI/SNNmodels/SNNTorchmodel/ds_test.pt', map_location=torch.device('cpu'), weights_only= False)

# Load the trained model
tmp_model = torch.load('./retrained_snntorch_20251007_161214.pt', map_location=torch.device('cpu'))

# Parameters (match C code)
NUM_IN = 12
NUM_L1 = 38
NUM_L2 = 7

threshold = tmp_model['lif1.threshold']

# Recurrent weights (self-loops)
recurrent_weights1 = tmp_model['lif1.V']

# Network definition with RLeaky neurons
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

# Load weights exactly as in C code
with torch.no_grad():

    net.fc1.weight.copy_(tmp_model['fc1.weight'])

    net.fc2.weight.copy_(tmp_model['fc2.weight'])
# Simulation
print("Recurrent SNN Simulation Starting...\n")

# Calculate and print accuracy over all samples
num_samples = len(tmp.tensors[0])
num_matches = 0

for i in tmp.tensors[0]:  # Iterate over samples in the dataset

  inputs = i.detach().clone().float().unsqueeze(1)  # take one sample from dataset

  # Forward pass through all timesteps at once
  spk1_rec, mem1_rec, spk2_rec, mem2_rec = net.forward(inputs)

  act_total_out = torch.sum(spk2_rec, 0)  # sum over time
  _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax over output units to identify the prediction```

  # Print prediction result
  print("Test number:", tmp.tensors[0].tolist().index(i.tolist()))
  print("Total output spikes per neuron:", act_total_out.squeeze().tolist())
  print("Predicted class (argmax):", neuron_max_act_total_out.item())

  # Get true label from dataset using index of i
  idx = tmp.tensors[0].tolist().index(i.tolist())
  true_label = tmp.tensors[1][idx].item()
  print("True class in dataset:", true_label)

  # Check match or miss
  if neuron_max_act_total_out.item() == true_label:
      print("Match\n")
      num_matches += 1
  else:
      print("Miss\n")

accuracy = num_matches / num_samples * 100
print(f"Accuracy over {num_samples} samples: {accuracy:.2f}%")
