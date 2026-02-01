"""
NeuroIDS-Adversarial: Spiking Neural Network Model with Adversarial Robustness
Author: Toby R. Davis
Mississippi State University
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LIFParams:
    """Leaky Integrate-and-Fire neuron parameters"""
    tau_m: float = 20.0      # Membrane time constant (ms)
    tau_ref: float = 3.0     # Refractory period (ms)
    v_rest: float = 0.0      # Resting potential
    v_reset: float = 0.0     # Reset potential
    v_thresh: float = 0.6    # Threshold potential
    r_m: float = 1.0         # Membrane resistance
    dt: float = 1.0          # Timestep (ms)


class SpikingNeuralNetwork:
    """
    Vectorized Spiking Neural Network for Intrusion Detection
    with adversarial robustness features.
    """
    
    def __init__(
        self,
        input_size: int = 41,
        hidden1_size: int = 96,
        hidden2_size: int = 48,
        output_size: int = 5,
        timesteps: int = 75,
        params: Optional[LIFParams] = None,
        seed: int = 42
    ):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.timesteps = timesteps
        self.params = params or LIFParams()
        
        np.random.seed(seed)
        
        # Initialize weights with Xavier initialization
        self.w1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.w2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.w_out = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        
        # Precompute decay factor
        self.decay = np.exp(-self.params.dt / self.params.tau_m)
        
        # Class scaling factors for imbalanced data
        self.class_scales = np.ones(output_size)
        
        # Training statistics
        self.training_history = []
        
    def encode_spikes_batch(self, X: np.ndarray, r_base: float = 0.1, r_max: float = 0.5) -> np.ndarray:
        """
        Rate-coded spike encoding for batch input.
        
        Args:
            X: Input features (batch_size, features)
            r_base: Base firing rate
            r_max: Maximum firing rate
            
        Returns:
            Spike trains (batch_size, timesteps, features)
        """
        batch_size, n_features = X.shape
        
        # Compute firing probabilities
        rates = r_base + X * (r_max - r_base)
        rates = np.clip(rates, 0, 1)
        
        # Generate spikes
        random_vals = np.random.rand(batch_size, self.timesteps, n_features)
        spikes = (random_vals < rates[:, np.newaxis, :]).astype(np.float32)
        
        return spikes
    
    def forward_batch(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Vectorized forward pass through the network.
        
        Args:
            X: Input features (batch_size, features)
            
        Returns:
            output_probs: Class probabilities (batch_size, classes)
            info: Dictionary with spike counts and intermediate values
        """
        batch_size = X.shape[0]
        p = self.params
        
        # Encode input as spikes
        input_spikes = self.encode_spikes_batch(X)
        
        # Initialize membrane potentials
        v1 = np.full((batch_size, self.hidden1_size), p.v_rest)
        v2 = np.full((batch_size, self.hidden2_size), p.v_rest)
        
        # Initialize refractory counters
        ref1 = np.zeros((batch_size, self.hidden1_size))
        ref2 = np.zeros((batch_size, self.hidden2_size))
        
        # Spike accumulators
        spike_count1 = np.zeros((batch_size, self.hidden1_size))
        spike_count2 = np.zeros((batch_size, self.hidden2_size))
        
        # Simulate network dynamics
        for t in range(self.timesteps):
            # Layer 1: Input -> Hidden1
            current1 = input_spikes[:, t, :] @ self.w1
            
            # Update membrane potential (only for non-refractory neurons)
            active1 = (ref1 <= 0)
            v1 = np.where(
                active1,
                p.v_rest + (v1 - p.v_rest) * self.decay + p.r_m * current1,
                v1
            )
            
            # Generate spikes
            spikes1 = (v1 >= p.v_thresh).astype(np.float32)
            spike_count1 += spikes1
            
            # Reset spiking neurons
            v1 = np.where(spikes1 > 0, p.v_reset, v1)
            ref1 = np.where(spikes1 > 0, p.tau_ref, ref1 - p.dt)
            
            # Layer 2: Hidden1 -> Hidden2
            current2 = spikes1 @ self.w2
            
            active2 = (ref2 <= 0)
            v2 = np.where(
                active2,
                p.v_rest + (v2 - p.v_rest) * self.decay + p.r_m * current2,
                v2
            )
            
            spikes2 = (v2 >= p.v_thresh).astype(np.float32)
            spike_count2 += spikes2
            
            v2 = np.where(spikes2 > 0, p.v_reset, v2)
            ref2 = np.where(spikes2 > 0, p.tau_ref, ref2 - p.dt)
        
        # Output layer: spike counts as features
        logits = spike_count2 @ self.w_out * self.class_scales
        
        # Softmax with numerical stability
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        info = {
            'spike_count1': spike_count1,
            'spike_count2': spike_count2,
            'logits': logits,
            'total_spikes': np.mean(spike_count1.sum(axis=1) + spike_count2.sum(axis=1))
        }
        
        return probs, info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions for input samples."""
        probs, _ = self.forward_batch(X)
        return np.argmax(probs, axis=1)
    
    def compute_loss(self, probs: np.ndarray, y: np.ndarray, 
                     class_weights: Optional[np.ndarray] = None,
                     focal_gamma: float = 2.0) -> float:
        """
        Compute focal loss for handling class imbalance.
        
        Args:
            probs: Predicted probabilities
            y: True labels (integer)
            class_weights: Optional per-class weights
            focal_gamma: Focal loss focusing parameter
        """
        batch_size = len(y)
        
        # Get probability of true class
        p_true = probs[np.arange(batch_size), y]
        p_true = np.clip(p_true, 1e-7, 1 - 1e-7)
        
        # Focal loss
        focal_weight = (1 - p_true) ** focal_gamma
        loss = -focal_weight * np.log(p_true)
        
        if class_weights is not None:
            loss *= class_weights[y]
        
        return np.mean(loss)
    
    def train_step(self, X: np.ndarray, y: np.ndarray, 
                   learning_rate: float = 0.02,
                   class_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Single training step with gradient descent on output weights.
        
        Args:
            X: Input features (batch_size, features)
            y: True labels (batch_size,)
            learning_rate: Learning rate
            class_weights: Optional per-class weights
            
        Returns:
            Dictionary with loss and accuracy
        """
        batch_size = len(y)
        
        # Forward pass
        probs, info = self.forward_batch(X)
        
        # Compute loss
        loss = self.compute_loss(probs, y, class_weights)
        
        # Create one-hot targets
        y_onehot = np.zeros((batch_size, self.output_size))
        y_onehot[np.arange(batch_size), y] = 1
        
        # Gradient on output weights (spike_count2.T @ (probs - y_onehot))
        grad = info['spike_count2'].T @ (probs - y_onehot) / batch_size
        
        # Apply class weights to gradient
        if class_weights is not None:
            weight_factor = class_weights[y].reshape(-1, 1)
            grad = info['spike_count2'].T @ ((probs - y_onehot) * weight_factor) / batch_size
        
        # Update weights
        self.w_out -= learning_rate * grad
        
        # Compute accuracy
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'total_spikes': info['total_spikes']
        }
    
    def update_class_scales(self, y_pred: np.ndarray, y_true: np.ndarray,
                           target_dist: np.ndarray, alpha: float = 0.05):
        """
        Adaptively adjust class scaling factors based on prediction distribution.
        """
        pred_counts = np.bincount(y_pred, minlength=self.output_size)
        pred_dist = pred_counts / len(y_pred)
        
        for k in range(self.output_size):
            if pred_dist[k] > 1.5 * target_dist[k]:
                self.class_scales[k] *= (1 - alpha)
            elif pred_dist[k] < 0.5 * target_dist[k]:
                self.class_scales[k] *= (1 + alpha)
    
    def get_spike_features(self, X: np.ndarray) -> np.ndarray:
        """Extract spike count features for a batch of inputs."""
        _, info = self.forward_batch(X)
        return info['spike_count2']
    
    def save_weights(self, filepath: str):
        """Save model weights to file."""
        np.savez(filepath, w1=self.w1, w2=self.w2, w_out=self.w_out,
                 class_scales=self.class_scales)
    
    def load_weights(self, filepath: str):
        """Load model weights from file."""
        data = np.load(filepath)
        self.w1 = data['w1']
        self.w2 = data['w2']
        self.w_out = data['w_out']
        self.class_scales = data['class_scales']


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              n_classes: int = 5) -> Dict:
    """Compute precision, recall, F1 for each class."""
    metrics = {}
    
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(y_true == c)
        }
    
    return metrics
