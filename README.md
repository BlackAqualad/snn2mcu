# snn2mcu: SNN Deployment on Low-Power Microcontrollers

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Thesis](https://img.shields.io/badge/Thesis-Politecnico%20di%20Torino-blue)](https://webthesis.biblio.polito.it/38593/)

Automated **NIR-to-C** workflow for deploying Spiking Neural Networks (SNNs) on resource-constrained microcontrollers. This project is based on my MSc Thesis at Politecnico di Torino.

---

## 🚀 Overview & Versioning

This repository provides two main versions of the code generator:

* **Generator V2 (Recommended for LIF):**
    * Supports **Linear** and **Affine** layers.
    * Features correct **beta-from-tau** computation (linear tau) compatible with default `export_nir` settings from snnTorch framework.
    * Optimized for **large architectures** to fit memory-constrained boards.
* **Original NIR-to-C (Recommended for Recursion):**
    * Use this version for **R-Leaky** (Recurrent LIF) neurons.
    * Required when using architectures not yet supported by the standard `export_nir` function.

---

## 🛠 How to Use

### 1. Requirements
Ensure you have the following installed:
* Python 3.8+
* `nir` library (`pip install nir`)
* A C compiler for your target MCU (e.g., GCC for ARM)

### 2. Workflow
1.  **Export your model:** Generate a `.nir` file from your SNN framework (e.g., snnTorch).
2.  **Choose your generator:** * For standard LIF models: use the scripts in the `V2` folder.
    * For recurrent models: use the `NIR-to-C` folder.
3.  **Run the translation:** Described in the generator files.
4.  **Deploy:** Include the generated `.h` and `.c` files in your MCU project.

---

## 🎓 Citation

If you use snn2mcu or the NIR-to-C workflow in your research or project, please cite my Master's Thesis. You can use the BibTeX entry below or the "Cite this repository" button in the GitHub sidebar.

**BibTeX:**
```bibtex
@mastersthesis{snn2mcu2025,
  author  = {Simone Delvecchio},
  title   = {Optimization of Spiking Neural Networks execution on low-power microcontrollers},
  school  = {Politecnico di Torino},
  year    = {2025},
  url     = {[https://webthesis.biblio.polito.it/38593/](https://webthesis.biblio.polito.it/38593/)}
}
