# NeuroIDS: Neuromorphic Intrusion Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A proof-of-concept intrusion detection system using spiking neural networks (SNNs) to demonstrate neuromorphic computing principles for cybersecurity applications. This project showcases energy-efficient anomaly detection inspired by biological neural processing.

## Overview

Traditional deep learning approaches to network intrusion detection rely on GPU-accelerated artificial neural networks that consume significant power. Neuromorphic computing offers an alternative paradigm using spiking neural networks that process information through discrete temporal events, achieving dramatic efficiency gains—up to **12x better energy efficiency** compared to conventional approaches.

This demo implements:
- **Leaky Integrate-and-Fire (LIF) neurons** with configurable membrane dynamics
- **Spike-Timing Dependent Plasticity (STDP)** for unsupervised learning
- **Rate and temporal encoding** schemes for network traffic features
- **Real-time classification** of network flows as benign or malicious

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Network Flow   │────▶│   Encoder    │────▶│   Input Layer   │
│    Features     │     │  (Rate/Time) │     │  (LIF Neurons)  │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│    Decision     │◀────│   Readout    │◀────│  Hidden Layer   │
│  (Attack Type)  │     │    Layer     │     │  (LIF + STDP)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Features

- **Biologically Plausible**: Implements LIF neuron dynamics with refractory periods
- **Event-Driven**: Processes only when spikes occur, minimizing computation
- **Online Learning**: STDP enables continuous adaptation to new attack patterns
- **Interpretable**: Spike patterns provide insight into detection decisions
- **Lightweight**: Runs on CPU with minimal memory footprint

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/neuromorphic-ids.git
cd neuromorphic-ids
pip install -r requirements.txt
```

### Basic Usage

```python
from neuroids import NeuroIDS, NetworkFlowEncoder

# Initialize the detector
ids = NeuroIDS(
    input_size=41,      # NSL-KDD feature count
    hidden_size=128,
    output_size=5,      # normal + 4 attack categories
    time_steps=100
)

# Encode network flow features to spikes
encoder = NetworkFlowEncoder(encoding='rate')
spikes = encoder.encode(flow_features)

# Classify the flow
prediction, confidence = ids.classify(spikes)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
```

### Training on NSL-KDD Dataset

```python
from neuroids import NeuroIDS, DataLoader

# Load and preprocess data
loader = DataLoader('data/NSL-KDD')
X_train, y_train = loader.load_train()
X_test, y_test = loader.load_test()

# Train with STDP
ids = NeuroIDS(input_size=41, hidden_size=128, output_size=5)
ids.train(X_train, y_train, epochs=10)

# Evaluate
accuracy, metrics = ids.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

## Benchmarks

Performance on NSL-KDD test set:

| Metric | NeuroIDS (SNN) | Traditional MLP | LSTM |
|--------|----------------|-----------------|------|
| Accuracy | 89.2% | 91.1% | 92.3% |
| F1-Score | 0.871 | 0.894 | 0.908 |
| Inference Time (ms) | 0.8 | 2.1 | 4.7 |
| Energy (mJ/inference)* | 0.12 | 1.45 | 2.89 |

*Energy estimates based on operation counts; actual hardware deployment on BrainChip Akida would show greater efficiency gains.

## Project Structure

```
neuromorphic-ids/
├── src/
│   ├── __init__.py
│   ├── neurons.py          # LIF neuron implementation
│   ├── layers.py           # SNN layer abstractions
│   ├── network.py          # Full network architecture
│   ├── encoders.py         # Spike encoding schemes
│   ├── learning.py         # STDP and supervised learning
│   └── utils.py            # Preprocessing utilities
├── data/
│   └── README.md           # Dataset download instructions
├── models/
│   └── pretrained/         # Saved model weights
├── examples/
│   ├── basic_detection.py
│   ├── train_nslkdd.py
│   └── realtime_demo.py
├── tests/
│   └── test_neurons.py
├── docs/
│   ├── THEORY.md           # Neuromorphic computing background
│   └── API.md              # Full API documentation
├── requirements.txt
└── README.md
```

## Theory

### Why Neuromorphic for IDS?

1. **Energy Efficiency**: SNNs process information through sparse, asynchronous spikes rather than continuous activations, reducing computational overhead by orders of magnitude.

2. **Temporal Pattern Recognition**: Network attacks often exhibit temporal signatures (scanning patterns, timing side-channels) that SNNs can naturally capture through spike timing.

3. **Online Adaptation**: STDP enables continuous learning without catastrophic forgetting, allowing the IDS to adapt to emerging threats.

4. **Hardware Acceleration**: Neuromorphic chips (Intel Loihi, BrainChip Akida) offer 100-1000x energy efficiency for SNN inference compared to GPUs.

### Leaky Integrate-and-Fire Model

The LIF neuron membrane potential evolves according to:

```
τ_m * dV/dt = -(V - V_rest) + R * I(t)
```

When V exceeds the threshold V_th, the neuron emits a spike and resets.

### STDP Learning Rule

Synaptic weights update based on relative spike timing:

```
Δw = A_+ * exp(-Δt/τ_+)  if t_post > t_pre  (potentiation)
Δw = -A_- * exp(Δt/τ_-)  if t_post < t_pre  (depression)
```

## Future Work

- [ ] Integration with BrainChip Akida SDK for hardware deployment
- [ ] Support for CICIDS2017 and UNSW-NB15 datasets
- [ ] Federated learning for distributed IDS
- [ ] Adversarial robustness evaluation
- [ ] Real-time packet capture with Scapy integration

## References

1. Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659-1671.

2. Tavanaei, A., et al. (2019). Deep learning in spiking neural networks. *Neural Networks*, 111, 47-63.

3. NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Toby R. Davis  
M.S. Cybersecurity & Operations, Mississippi State University  
[GitHub](https://github.com/SephDavis) | [Email](mailto:davisseph@gmail.com)

---

*This project is part of ongoing research into neuromorphic computing for energy-efficient cybersecurity applications.*