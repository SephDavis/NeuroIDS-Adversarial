# NeuroIDS-Adversarial

Adversarial Robustness Analysis of Neuromorphic Intrusion Detection Systems

Author: Toby R. Davis, Mississippi State University

## Quick Start

```bash
pip install -r requirements.txt
python main.py --experiment all --epochs 20
```

## Structure

- `main.py` - Experiment runner
- `src/snn_model.py` - SNN implementation  
- `src/adversarial_attacks.py` - FGSM, PGD, Mimicry attacks
- `src/adversarial_defenses.py` - Defense mechanisms
- `src/data_loader.py` - Data handling

## Attacks Implemented

1. **FGSM** - Fast Gradient Sign Method
2. **PGD** - Projected Gradient Descent  
3. **Spike Timing** - SNN-specific temporal attack
4. **Mimicry** - Traffic disguise attack

## Defenses Implemented

1. Input preprocessing (noise, squeezing, filtering)
2. Temporal averaging (SNN-specific)
3. Adversarial training
