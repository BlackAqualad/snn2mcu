"""
Compare NIR file with original SNNTorch model to verify conversion accuracy.
"""

import nir
import torch
import numpy as np

# Load the original SNNTorch model
print("Loading original SNNTorch model...")
tmp_model = torch.load('./retrained_snntorch_20251007_161214.pt', map_location=torch.device('cpu'))

# Load the NIR file
print("Loading NIR file...")
nir_graph = nir.read('snntorch_braille7_model.nir')

print("\n" + "=" * 70)
print("CONVERSION VERIFICATION")
print("=" * 70)

# Extract SNNTorch parameters
beta1 = tmp_model['lif1.beta'].item()
beta2 = tmp_model['lif2.beta'].item()
threshold = tmp_model['lif1.threshold'].item()
w_fc1_torch = tmp_model['fc1.weight'].cpu().numpy()
w_fc2_torch = tmp_model['fc2.weight'].cpu().numpy()
w_rec1_torch = tmp_model['lif1.V'].cpu().numpy()

# Calculate expected tau values
tau1_expected = -1.0 / np.log(beta1)
tau2_expected = -1.0 / np.log(beta2)

print("\n--- LAYER 1 (RLeaky) ---")
print(f"SNNTorch Beta: {beta1:.6f}")
print(f"Expected Tau: {tau1_expected:.6f}")
print(f"NIR Tau: {nir_graph.nodes['lif1'].tau[0]:.6f}")
print(f"Match: {np.isclose(nir_graph.nodes['lif1'].tau[0], tau1_expected)}")

print(f"\nSNNTorch Threshold: {threshold:.6f}")
print(f"NIR Threshold: {nir_graph.nodes['lif1'].v_threshold[0]:.6f}")
print(f"Match: {np.isclose(nir_graph.nodes['lif1'].v_threshold[0], threshold)}")

print(f"\nFC1 Weights match: {np.allclose(nir_graph.nodes['fc1'].weight, w_fc1_torch)}")
print(f"Recurrent Weights (diagonal) match: {np.allclose(np.diag(nir_graph.nodes['rec1'].weight), w_rec1_torch)}")

print("\n--- LAYER 2 (Leaky) ---")
print(f"SNNTorch Beta: {beta2:.6f}")
print(f"Expected Tau: {tau2_expected:.6f}")
print(f"NIR Tau: {nir_graph.nodes['lif2'].tau[0]:.6f}")
print(f"Match: {np.isclose(nir_graph.nodes['lif2'].tau[0], tau2_expected)}")

print(f"\nSNNTorch Threshold: {threshold:.6f}")
print(f"NIR Threshold: {nir_graph.nodes['lif2'].v_threshold[0]:.6f}")
print(f"Match: {np.isclose(nir_graph.nodes['lif2'].v_threshold[0], threshold)}")

print(f"\nFC2 Weights match: {np.allclose(nir_graph.nodes['fc2'].weight, w_fc2_torch)}")

print("\n--- RESET MECHANISM ---")
print(f"Layer 1 V_reset: {nir_graph.nodes['lif1'].v_reset[0]:.6f} (should be 0.0 for 'zero' reset)")
print(f"Layer 2 V_reset: {nir_graph.nodes['lif2'].v_reset[0]:.6f} (should be 0.0 for 'zero' reset)")

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

all_match = (
    np.isclose(nir_graph.nodes['lif1'].tau[0], tau1_expected) and
    np.isclose(nir_graph.nodes['lif2'].tau[0], tau2_expected) and
    np.isclose(nir_graph.nodes['lif1'].v_threshold[0], threshold) and
    np.isclose(nir_graph.nodes['lif2'].v_threshold[0], threshold) and
    np.allclose(nir_graph.nodes['fc1'].weight, w_fc1_torch) and
    np.allclose(nir_graph.nodes['fc2'].weight, w_fc2_torch) and
    np.allclose(np.diag(nir_graph.nodes['rec1'].weight), w_rec1_torch)
)

if all_match:
    print("✓ SUCCESS: NIR conversion is accurate!")
    print("  All parameters match the original SNNTorch model.")
else:
    print("✗ WARNING: Some parameters don't match!")
    print("  Please review the comparison above.")

print("\n" + "=" * 70)
