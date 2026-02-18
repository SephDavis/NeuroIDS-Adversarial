# NeuroIDS-Adversarial

**Evading NeuroIDS: Adversarial Attack Analysis and Hardening for Neuromorphic Intrusion Detection**

Toby R. Davis — Department of Computer Science and Engineering, Mississippi State University

[![DOI](https://img.shields.io/badge/NeuroIDS-10.13140%2FRG.2.2.23827.13604-blue)](https://www.researchgate.net/doi/10.13140/RG.2.2.23827.13604)
[![DOI](https://img.shields.io/badge/NeuroIDS--Sat-10.13140%2FRG.2.2.16893.42724-blue)](https://www.researchgate.net/doi/10.13140/RG.2.2.16893.42724)

---

## Overview

NeuroIDS-Adversarial is a research framework for evaluating the adversarial robustness of spiking neural networks (SNNs) used for network intrusion detection. This project extends [NeuroIDS](https://github.com/SephDavis/NeuroIDS) by systematically attacking the neuromorphic classifier, analyzing its vulnerability surface across all five attack categories, and implementing defensive hardening strategies.

**Core question:** Can an adversary craft network traffic that evades a neuromorphic intrusion detection system, and if so, how do we defend against it?

**Key findings:**
- FGSM reduces SNN accuracy from 62.4% to 54.1% at ε=0.20 (7.5pp degradation)
- PGD provides only marginal improvement over FGSM due to SNN stochasticity
- Mimicry attacks evade detection for ~46% of malicious traffic at ε=0.20
- Adversarial attacks disproportionately amplify the existing attack→Normal misclassification bias
- Adversarial training improves *both* clean accuracy and adversarial robustness simultaneously — atypical for deep learning, attributable to regularization on output-layer-only training
- SNN stochasticity provides partial inherent robustness confirmed against an adaptive adversary

## Motivation

Neuromorphic IDS offer dramatic power efficiency advantages (1,500 pJ per inference vs. 45 mJ for conventional CPUs), making them viable for deployment in resource-constrained environments — CubeSat constellations, tactical edge networks, IoT infrastructure. But power efficiency means nothing if an adversary can trivially evade detection.

Prior work on NeuroIDS and NeuroIDS-Sat identified that the dominant failure mode is attack→Normal misclassification. This project quantifies how adversarial perturbation amplifies that failure mode and establishes practical defenses.

## Architecture

The SNN classifier follows the NeuroIDS architecture with output-layer-only training to isolate the adversarial vulnerability of the SNN representation itself:

| Component | Configuration |
|-----------|--------------|
| Encoding | Rate-coded spike trains, r_base=0.1, r_max=0.5 |
| Hidden layer 1 | 96 LIF neurons, τ_m=20ms, V_thresh=0.6 |
| Hidden layer 2 | 48 LIF neurons |
| Output | Linear readout from spike counts → 5 classes |
| Simulation | 75 timesteps, Δt=1ms, τ_ref=3ms |
| Training | Focal loss (γ=2.0), inverse-frequency class weights |
| Parameters | 8,933 total |

Output-layer-only training (frozen hidden weights, learned linear readout) is used deliberately: perturbations that change the spike count feature vector reveal properties of the spiking dynamics rather than the output layer decision boundary.

## Attacks Implemented

### FGSM (Fast Gradient Sign Method)
Single-step perturbation using SPSA-based numerical gradient estimation (N=5 samples, σ=0.01). Standard backpropagation is impossible through the non-differentiable spike generation threshold.

### PGD (Projected Gradient Descent)
Iterative attack with 10 steps, random initialization, step size α=ε/4, projection back to ℓ∞ ε-ball. SPSA ablation (N=5→50) shows PGD separates from FGSM only with higher gradient fidelity, confirming SNN stochasticity as the dominant factor limiting iterative refinement.

### Spike Timing Attack
SNN-specific attack targeting features with high gradient magnitude (above median). Perturbs *when* neurons fire within the 75-timestep window. Results confirm spike count-based readout provides architectural robustness to temporal perturbation — count aggregation acts as a low-pass filter over timing noise.

### Mimicry Attack
Transforms malicious traffic toward the normal class centroid, bounded by ε. Per-class analysis reveals R2L achieves highest evasion (~62%) since these attacks inherently mimic legitimate access patterns, while DoS achieves lowest (~45%) due to larger feature distance from normal traffic.

### Combined Evasion
Multi-stage: FGSM first, then mimicry refinement on still-detected samples.

## Defenses Implemented

### Input Preprocessing
- **Gaussian noise** (σ=0.05) — Randomization to disrupt adversarial structure
- **Feature squeezing** (4-bit) — Quantization to remove sub-threshold perturbations
- **Median filtering** — Smoothing across neighboring features

### SNN-Specific Defenses
- **Temporal averaging** — Averages predictions across M=5 stochastic forward passes, exploiting the fact that adversarial examples optimized against one spike realization transfer poorly to others
- **Spike consistency check** — Flags samples with inconsistent predictions across multiple runs as potentially adversarial

### Adaptive Adversary Evaluation
Temporal averaging is evaluated against an adaptive adversary that averages gradients over M_a=5 stochastic passes before computing perturbations. Results confirm partial but genuine inherent robustness from SNN stochasticity.

### Adversarial Training
Mixed training on 50% clean + 50% FGSM adversarial examples (ε=0.1). Early stopping on combined clean/adversarial validation accuracy. Uniquely for SNNs, this improves both clean and adversarial performance simultaneously.

## Experiment Suite

Running `python main.py` generates data for every table in the paper:

| Function | Paper Table | Description |
|----------|------------|-------------|
| `table_ii_baseline()` | Table II | Per-class precision, recall, F1, support |
| `table_iii_gradient_attacks()` | Table III | FGSM (ε=0.01–0.20) and PGD (ε=0.05–0.15) |
| `table_iv_spsa_ablation()` | Table IV | SPSA sample count ablation (N=5,10,20,50) |
| `table_v_perclass_fgsm()` | Table V | Per-class recall degradation under FGSM |
| `table_vi_misclassification_direction()` | Table VI | Dominant misclassification target per class |
| `table_vii_spike_timing()` | Table VII | Spike timing attack (η=0.05–0.20) |
| `table_viii_mimicry_perclass()` | Table VIII | Mimicry evasion rate by attack class |
| `table_ix_mimicry_vs_eps()` | Table IX | Mimicry evasion vs perturbation budget |
| `table_x_defenses()` | Table X | All defenses: clean and FGSM accuracy |
| `table_xi_adaptive_adversary()` | Table XI | Standard vs adaptive FGSM against temporal averaging |
| `table_xii_adversarial_training()` | Table XII | Adversarial training comparison with baseline |

Additionally generates full 5×5 confusion matrices for baseline and adversarially trained models, with all results exported to `results.json`.

## Dataset

Uses the [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) dataset:

| Category | Train | Test | Train % |
|----------|-------|------|---------|
| Normal | 67,343 | 9,711 | 53.5% |
| DoS | 45,927 | 7,458 | 36.5% |
| Probe | 11,656 | 2,421 | 9.3% |
| R2L | 995 | 2,754 | 0.8% |
| U2R | 52 | 200 | 0.04% |

Minority classes are oversampled to 10% of the majority class count with Gaussian noise augmentation (σ=0.01). Place `KDDTrain+.txt` and `KDDTest+.txt` in `./data/`. If no data files are found, synthetic data is generated as a fallback.

## Quick Start

### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
```

### Installation

```bash
git clone https://github.com/SephDavis/NeuroIDS-Adversarial.git
cd NeuroIDS-Adversarial
pip install -r requirements.txt
```

### Running Experiments

Full run (all tables, ~30–60 min on full test set):
```bash
python main.py
```

Fast iteration (subset of test data, fewer epochs):
```bash
python main.py --test-subset 2000 --epochs 10 --adv-epochs 8
```

Skip the slow SPSA ablation:
```bash
python main.py --experiment quick
```

Individual phases:
```bash
python main.py --experiment train --epochs 20
python main.py --experiment attack
python main.py --experiment defense
```

Force unbuffered output on Windows:
```bash
python -u main.py
```

## Project Structure

```
NeuroIDS-Adversarial/
├── main.py                    # Comprehensive experiment runner (all paper tables)
├── requirements.txt
├── README.md
├── results.json               # Generated experiment results (after running main.py)
├── data/
│   ├── KDDTrain+.txt          # NSL-KDD training set (not included, download separately)
│   └── KDDTest+.txt           # NSL-KDD test set
└── src/
    ├── __init__.py
    ├── snn_model.py           # SNN with LIF neurons, focal loss, spike count classification
    ├── adversarial_attacks.py # FGSM, PGD, spike timing, mimicry, combined evasion
    ├── adversarial_defenses.py # Adversarial training, input denoising, temporal averaging
    └── data_loader.py         # NSL-KDD loading, normalization, balancing, synthetic fallback
```

## Related Work

- **[NeuroIDS](https://www.researchgate.net/doi/10.13140/RG.2.2.23827.13604)** — Neuromorphic intrusion detection achieving 73.4% accuracy on NSL-KDD at 1,620 pJ per inference
- **[NeuroIDS-Sat](https://www.researchgate.net/doi/10.13140/RG.2.2.16893.42724)** — Space-adapted variant for CubeSat constellations with radiation tolerance via triple modular redundancy, 70.1% accuracy at 1.52 mW

## Citation

```bibtex
@article{davis2026neuroids-adversarial,
  title={Evading NeuroIDS: Adversarial Attack Analysis and Hardening
         for Neuromorphic Intrusion Detection},
  author={Davis, Toby R.},
  institution={Mississippi State University},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

This research was supported by the Department of Defense Cyber Service Academy Scholarship program at Mississippi State University.
