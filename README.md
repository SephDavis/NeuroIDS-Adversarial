# NeuroIDS-Adversarial

**Evading NeuroIDS: Adversarial Attack Analysis and Hardening for Neuromorphic Intrusion Detection**

Author: Toby R. Davis
Department of Computer Science and Engineering
Mississippi State University

---

## Overview

NeuroIDS-Adversarial is a research framework that evaluates the adversarial robustness of spiking neural networks (SNNs) used for network intrusion detection. This project extends [NeuroIDS](https://github.com/SephDavis/NeuroIDS) by systematically attacking the neuromorphic classifier, analyzing its vulnerability surface, and implementing defensive hardening strategies.

The core question this work addresses: **Can an adversary craft network traffic that evades a neuromorphic intrusion detection system, and if so, how do we defend against it?**

SNNs present a unique adversarial challenge. Unlike conventional deep neural networks, spike-based classifiers are inherently non-differentiable due to discrete spike generation, which means standard gradient-based attacks (FGSM, PGD) cannot be applied directly. This project adapts these attacks for SNNs using numerical gradient estimation and introduces SNN-specific attack vectors that exploit temporal spike dynamics.

## Motivation

Neuromorphic intrusion detection systems offer significant advantages in power efficiency and real-time processing, but their robustness against adversarial manipulation remains largely unexplored. As these systems move toward deployment in resource-constrained environments (CubeSat constellations, edge networks, IoT infrastructure), understanding their failure modes under adversarial pressure is critical.

This work provides:

- **Offensive analysis** — Multiple attack strategies tailored for spiking neural networks, including a novel spike timing attack that exploits temporal dynamics unique to SNNs
- **Defensive hardening** — Practical defense mechanisms ranging from input preprocessing to adversarial training, with quantified robustness improvements
- **Attack-defense tradeoff analysis** — Evaluation across perturbation magnitudes (ε = 0.01 to 0.20) showing how accuracy degrades under attack and recovers with defenses applied

## Architecture

The SNN classifier follows the NeuroIDS architecture:

- **Encoding layer** — Rate-coded spike encoding converts normalized network features into spike trains
- **Hidden layer 1** — 96 Leaky Integrate-and-Fire (LIF) neurons with membrane time constant τ_m = 20ms
- **Hidden layer 2** — 48 LIF neurons
- **Output layer** — Spike count-based classification across 5 categories (Normal, DoS, Probe, R2L, U2R)
- **Simulation** — 75 timesteps per inference, refractory period τ_ref = 3ms

Training uses focal loss with adaptive class scaling to handle the severe class imbalance in NSL-KDD (U2R has only 52 training samples vs. 67,343 Normal samples).

## Attacks Implemented

### Fast Gradient Sign Method (FGSM)

Single-step perturbation in the direction of the estimated gradient sign. Because SNNs are non-differentiable, we use SPSA-based numerical gradient estimation with random perturbation directions rather than backpropagation.

```
X_adv = X + ε · sign(∇_x L(θ, X, y))
```

### Projected Gradient Descent (PGD)

Iterative attack with random initialization and projection back to the ε-ball at each step. Stronger than FGSM but computationally more expensive. Runs 10 iterations with step size α = ε/4.

```
X_adv^(t+1) = Π_{ε-ball}(X_adv^(t) + α · sign(∇_x L))
```

### Spike Timing Attack

An SNN-specific attack that exploits the temporal sensitivity of spike generation. Small perturbations to features near the firing threshold shift when spikes occur during the 75-timestep simulation window, potentially altering the spike count distribution enough to cause misclassification. This attack targets high-importance features identified through gradient magnitude analysis.

### Mimicry Attack

Transforms malicious traffic to statistically resemble normal traffic while staying within the perturbation budget. Computes the mean and standard deviation of normal traffic features and shifts malicious samples toward that distribution, bounded by ε. Particularly relevant for evading anomaly-based detection.

### Combined Evasion Attack

Multi-stage attack that applies FGSM first, then refines still-detected samples with mimicry. Models a realistic adversary who uses multiple strategies in sequence.

## Defenses Implemented

### Input Preprocessing

- **Gaussian noise injection** — Adds random noise (σ = 0.05) to disrupt adversarial perturbations while preserving clean accuracy
- **Feature squeezing** — Reduces feature precision to discrete bins (4-bit depth), removing small perturbations below the quantization threshold
- **Median filtering** — Replaces each feature with the median of neighboring features, smoothing out localized perturbations

### SNN-Specific Defenses

- **Temporal averaging** — Exploits the stochastic nature of spike generation by averaging predictions across multiple forward passes. Adversarial examples tuned to a single spike pattern become less effective when the pattern varies
- **Spike consistency check** — Runs multiple inference passes and flags samples with inconsistent predictions as potentially adversarial. Clean samples produce stable predictions; adversarial samples near decision boundaries do not

### Adversarial Training

Trains a new model on a mixed dataset of 50% clean and 50% FGSM-generated adversarial examples. Uses early stopping based on a combined metric of clean and adversarial validation accuracy to balance robustness with baseline performance.

## Dataset

This project uses the [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) dataset, a refined version of KDD Cup 1999 that removes duplicate records and provides a more balanced test set.

**Classes:**

| Category | Description | Train Samples | Test Samples |
|----------|-------------|---------------|--------------|
| Normal | Legitimate traffic | 67,343 | 9,711 |
| DoS | Denial of Service | 45,927 | 7,458 |
| Probe | Surveillance/scanning | 11,656 | 2,421 |
| R2L | Remote to Local | 995 | 2,754 |
| U2R | User to Root | 52 | 200 |

To use real data, place `KDDTrain+.txt` and `KDDTest+.txt` in `./data/`. The loader also checks `./data/NSL-KDD/`. If no data files are found, synthetic data is generated as a fallback.

## Quick Start

### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
```

### Installation

```bash
git clone https://github.com/sephdavis/neuroids-adversarial.git
cd neuroids-adversarial
pip install -r requirements.txt
```

### Running Experiments

Run all experiments (training, attacks, defenses):

```bash
python main.py --experiment all --epochs 20
```

Run individual phases:

```bash
python main.py --experiment train --epochs 20
python main.py --experiment attack
python main.py --experiment defense
```

## Project Structure

```
neuroids-adversarial/
├── main.py                      # Experiment runner
├── requirements.txt             # Python dependencies
├── README.md
├── data/                        # NSL-KDD dataset (not included)
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
└── src/
    ├── snn_model.py             # SNN with LIF neurons, focal loss, spike count classification
    ├── adversarial_attacks.py   # FGSM, PGD, spike timing, mimicry, combined evasion
    ├── adversarial_defenses.py  # Adversarial training, input denoising, temporal averaging
    └── data_loader.py           # NSL-KDD loading, normalization, balancing
```

## Related Work

- **NeuroIDS** — Neuromorphic intrusion detection achieving 73.4% accuracy on NSL-KDD with 1,620 pJ per inference
- **NeuroIDS-Sat** — Space-adapted variant for CubeSat constellations with radiation tolerance via triple modular redundancy
- **BlackLock** — Post-quantum encryption system using Ring-LWR for securing neuromorphic IDS communications

## Citation

```bibtex
@article{davis2026neuroids-adversarial,
  title={Evading NeuroIDS: Adversarial Attack Analysis and Hardening 
         for Neuromorphic Intrusion Detection},
  author={Davis, Toby R.},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

This research was supported by the Department of Defense Cyber Service Academy Scholarship program at Mississippi State University.
