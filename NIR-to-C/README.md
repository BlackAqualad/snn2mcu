NIR-to-C Generator — Quick Start and NIR Authoring Guide

Purpose
-------
This guide explains how to write NIR files so they work with the project's NIR→C generator, how to express recurrence in NIR for this workflow, and which features are supported or intentionally unsupported by the generator/runtime.

Who is this for
----------------
- Engineers preparing a NIR network export (from PyTorch, snntorch, or other frameworks) to embed on Cortex-M microcontrollers.
- Developers who will customize the generator to add features (biases, full recurrence, new neuron models).

Quick Principles
----------------
- The generator expects a simple feedforward chain of layers (Input → Affine → LIF/RLeaky → ...). Optional recurrence must be expressed explicitly using an `Affine` node that feeds back into the same LIF layer (affine-recurrent pattern).
- Inputs to the generated runtime are binary spikes (`q7_t`, values 0 or 1). The generator does not perform float→spike conversion at runtime by default.
- Biases are ignored by default. If your network requires biases, either bake them into the weight matrix or extend the generator/runtime.
- The resistance must be 1, as it behaves like a classical SNN.
- V_leak must be 0. 
- To use 1-to-1, just use a diagonal matrix, it will be translated into 1-to-1 connection.

Minimal requirements for a compatible NIR
----------------------------------------
1. Input node
   - Provide one input node. Its `input_type` should be a mapping that contains a NumPy array describing the input shape, for example:
     - `input_node.input_type = {'input': np.array([12])}`
   - The generator reads the first array value to determine the number of inputs.

2. Feedforward weights (Affine nodes)
   - Use `Affine` nodes for weight layers. Each `Affine` must include a `weight` numpy 2D array with shape `(out_features, in_features)`.
   - The generator writes weights to C using an input-major layout computed as `weight.T.flatten()`.

3. Neuron layers (LIF / RLeaky parameters)
   - After each Affine, place a `LIF` or `RLeaky` node that includes these parameters:
     - `beta` (decay) — discrete-time decay multiplier in (0,1].
     - `v_threshold` (threshold) or `threshold` — spike threshold.
     - `v_reset` — reset potential after a spike.
   - Parameters may be uniform (1-element arrays like `np.array([0.9])`) or per-neuron (arrays of size equal to the layer). Uniform parameters produce smaller generated code.

How recurrence is expressed (affine-recurrent workaround)
---------------------------------------------------------
The generator does not treat `RLeaky` as a first-class native recurrent node. Instead, recurrence is expressed by an explicit `Affine` that feeds back into the same layer. Use this pattern:

- Let `lif1` be the LIF layer receiving feedforward inputs.
- Create an `Affine` named with a `rec` suffix (convention) that takes `lif1` outputs and writes back to `lif1` inputs. Example name: `fc1_rec`.
- Provide `fc1_rec.weight` as a 2D NumPy array that is diagonal if you want memory-efficient 1-to-1 recurrence.

Supported recurrence modes:
- 1-to-1 (diagonal) recurrence: supported natively. The generator extracts the diagonal and stores it as a 1D vector (saves memory and simplifies runtime).
- Full all-to-all recurrence: not supported by default. The generator will raise an error if it detects a non-diagonal recurrent matrix unless you modify it.

Why this pattern?
- It gives explicit control and keeps the runtime simple. The generator's update functions are optimized for feedforward+diagonal-recurrent patterns used on embedded devices.

Supported features (short)
--------------------------
- Feedforward fully-connected networks (Affine + LIF).
- Uniform or per-neuron parameters for thresholds, resets, and decays.
- Diagonal (1-to-1) recurrent weights via an `Affine` feedback node (extracted as a 1D vector).
- Q15 fixed-point representation for thresholds and decay values. We use a default scale factor of 60.0.
- Scientific notation weight formatting (4 decimal digits) for cross-compiler stability.

Unsupported / intentionally excluded features
--------------------------------------------
- Bias terms in Affine layers (ignored by generated runtime).
- Other neuron models (custom types beyond LIF/RLeaky) without generator/runtime changes.
- All to all is always the SNNTorch false, but the weights could both be uniform and different per neuron.

Practical examples
------------------
1) Diagonal recurrent layer (recommended memory-efficient recurrence)

- In Python/NIR: create an `Affine` whose weight is a diagonal matrix with shape `(N,N)` and name it `fc1_rec` (or include `rec` in the name). Connect `lif1` output to `fc1_rec` and `fc1_rec` back to `lif1`.
- The generator will extract the diagonal and produce a `recurrent_weights1[N]` vector in C.

2) Feedforward only

- Just chain `Input -> Affine -> LIF -> Affine -> LIF -> Output`.

Customization points (where you can safely change behavior)
-----------------------------------------------------------
- scale factor: change `self.scale_factor` in `nir_to_c_generator.py` to alter Q15 scaling.
- weight formatting: edit `_format_weight()` to adjust precision or notation.
- bias support: add generator code to detect non-zero `bias` arrays and emit bias arrays + accumulation in C.
- recurrence: to enable all-to-all recurrence, remove/alter the diagonal check and change the runtime update functions to perform matrix-vector accumulation for recurrent inputs.

Troubleshooting notes
---------------------
- "Non-diagonal recurrent matrix" error: convert recurrence to diagonal or modify the generator to accept dense recurrence.
- Wrong runtime output: check that weight shapes are `(out,in)` and that you flattened with `weight.T.flatten()`.

---
