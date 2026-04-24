# From χ² to Normalising Flows — ML from Scratch for Physicists

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/prose-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

A step-by-step tutorial series taking you from linear regression all the way to
normalising flows, the generative models now used in gravitational wave parameter
estimation, collider anomaly detection, and cosmological inference using nothing
but familiar physics intuition and a single unifying idea:

$$\theta^* = \underset{\theta}{\arg\min}\; L(\theta)$$

If you have ever minimised a χ² by hand, you already understand the engine.
This series makes that connection explicit and builds on it systematically.

The companion blog lives at **[wesmail.github.io](https://wesmail.github.io/)**.

## Series overview

| Step | Topic | Key concept introduced | File |
|------|-------|----------------------|------|
| 1 | Linear regression | Loss function, MSE, the optimisation loop | `tutorial_01_regression.ipynb` |
| 2 | Classification | Cross-entropy, sigmoid, ROC curve | `tutorial_02_classification.ipynb` |
| 3 | Neural networks | Layers, ReLU, gradient descent, backprop | `tutorial_03_neural_networks.ipynb` |
| 4 | Overfitting & regularisation | Train/val/test split, L2, early stopping | `tutorial_04_overfitting.ipynb` |
| 5 | Density estimation | Negative log-likelihood, KDE, GMM | `tutorial_05_density.ipynb` |
| 6 | Autoencoders | Latent space, reconstruction loss, PyTorch intro | `tutorial_06_autoencoder.ipynb` |
| 7 | Variational Autoencoders (VAE) | ELBO derivation, KL divergence, reparameterisation | `tutorial_07_vae.ipynb` |
| 8 | Normalising Flows | Change of variables, coupling layers, RealNVP | `tutorial_08_flows.ipynb` |

Every notebook is self-contained: it generates its own synthetic dataset, trains a
model, prints a detailed explanation of each step, and saves publication-quality
plots. Running them in order is recommended but not required, each file includes a
recap of the concepts it builds on.

## Physics connections at a glance

The table below maps each tutorial concept to the place you are most likely to
encounter it in a physics context.

| Tutorial concept | Physics / astronomy equivalent |
|-----------------|-------------------------------|
| MSE loss | χ² under Gaussian noise |
| Maximum likelihood | Parameter estimation in Bayesian analyses |
| ROC curve | Detection efficiency vs. false alarm rate |
| Neural network classifier | PID in collider detectors; GW trigger ranking |
| KDE / GMM density | Population models for BBH mass spectra |
| Autoencoder | Anomaly detection in detector data |
| VAE | Latent-space population inference |
| Normalising flow | DINGO, RIFT, CaloFlow, FlowMC, sbi |

## Requirements

The full series uses five libraries. All steps up to and including Step 5 require
only `numpy`, `matplotlib`, `scipy`, and `scikit-learn`. Steps 6–8 additionally
require `torch` (PyTorch CPU is sufficient — no GPU needed for these tutorials).

```
numpy      >= 1.24
matplotlib >= 3.7
scipy      >= 1.10
scikit-learn >= 1.3
torch      >= 2.0
```

## Installation

### Option A: `pip`

It is good practice to work inside a virtual environment so the tutorial
dependencies do not interfere with your system packages.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 2. Install all dependencies
pip install numpy matplotlib scipy scikit-learn torch

# or, to pin exact versions for full reproducibility
pip install -r requirements.txt
```

### Option B: `uv` (recommended for speed)

[uv](https://github.com/astral-sh/uv) is a modern Python package manager written
in Rust. It resolves and installs dependencies significantly faster than pip and
handles virtual environments automatically.

```bash
# 1. Install uv (once, system-wide)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# on Windows (PowerShell):
# irm https://astral.sh/uv/install.ps1 | iex

# 2. Create a virtual environment and install everything in one step
uv venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
uv pip install numpy matplotlib scipy scikit-learn torch

# alternatively, if using a pyproject.toml-based project
uv sync
```

### Verifying the installation

Run the following one-liner to confirm all libraries are importable before
starting:

```bash
python -c "import numpy, matplotlib, scipy, sklearn, torch; print('All good!')"
```


## License

This repository contains two distinct types of content, each under its own
license.

**Code** — all `.py` files and `requirements.txt` — is released under the
**MIT License** (see [`LICENSE`](LICENSE) for the full text). You are free to use,
copy, modify, and distribute the code for any purpose, including commercial use,
as long as you retain the copyright notice.

**Written content** — the prose, explanations, and figures in the companion blog
at [wesmail.github.io](https://wesmail.github.io/) — is released under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**. You are free to
share and adapt the material for any purpose as long as you give appropriate
credit and link back to the original.

```
MIT License

Copyright (c) 2025 Waleed Esmail

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Citation

If you use this material in your research or teaching, a citation is appreciated:

```bibtex
@misc{esmail2025mlfromscratch,
  author = {Esmail, Waleed},
  title  = {From {$\chi^2$} to Normalising Flows: {ML} from Scratch for Physicists},
  year   = {2025},
  url    = {https://wesmail.github.io/contents/blogs/ml-from-scratch/},
  note   = {Blog series and companion code}
}
```

## Acknowledgements

Synthetic datasets throughout this series are inspired by real problems in
gravitational-wave astronomy (LIGO/Virgo), high-energy particle physics (LHC
experiments), and multi-messenger astrophysics. No real experimental data is used.

*Dr. Waleed Esmail — Postdoctoral Researcher, University of Münster*  
*[wesmail.github.io](https://wesmail.github.io/)*