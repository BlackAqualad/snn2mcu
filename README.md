# snn2mcu
Automated NIR-to-C workflow for deploying Spiking Neural Networks (SNNs) on low-power microcontrollers. Based on my MSc Thesis at Politecnico di Torino.

The generator V2 is different from the NIR-to-C, in the former, there is the linear support and the tau is calculated from the NIR default export settings (not exponential but linear tau from beta)

For recursion, use the NIR-TO-C folder one (V2 not tested for custom recursive architectures).

Use that when using R-Leaky, which is not supported by the export_nir function.

If there are only Leaky (LIF), use the V2 instead, which supports Linear and Affine and has the correct beta from tau computation compatible with export_nir

The V2 also has minor differences to support big architectures to make them fit into the board.
