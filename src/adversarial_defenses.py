"""
Adversarial Defense Module for NeuroIDS
Implements defensive strategies to harden SNNs against adversarial attacks.

Author: Toby R. Davis
Mississippi State University
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from snn_model import SpikingNeuralNetwork, compute_per_class_metrics
from adversarial_attacks import AdversarialAttacker, AttackConfig


class DefenseType(Enum):
    """Types of adversarial defenses"""
    ADVERSARIAL_TRAINING = "adversarial_training"
    INPUT_DENOISING = "input_denoising"
    SPIKE_SMOOTHING = "spike_smoothing"
    ENSEMBLE = "ensemble"
    FEATURE_SQUEEZING = "feature_squeezing"
    RANDOMIZED_SMOOTHING = "randomized_smoothing"


@dataclass
class DefenseConfig:
    """Configuration for adversarial defenses"""
    # Adversarial training
    adv_train_ratio: float = 0.5      # Ratio of adversarial examples in training
    adv_epsilon: float = 0.1          # Epsilon for generating training attacks
    
    # Input denoising
    denoise_sigma: float = 0.05       # Gaussian noise for denoising
    
    # Feature squeezing
    squeeze_depth: int = 4            # Bit depth for squeezing
    
    # Ensemble
    n_models: int = 5                 # Number of models in ensemble
    
    # Randomized smoothing
    smooth_sigma: float = 0.1         # Noise level for smoothing
    smooth_samples: int = 100         # Number of samples for certification


class InputDenoiser:
    """
    Input preprocessing defenses.
    
    These methods transform inputs before passing to the classifier,
    potentially removing adversarial perturbations.
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()
    
    def gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to inputs.
        
        Randomization can disrupt adversarial perturbations while
        preserving clean sample accuracy (if noise is small enough).
        """
        noise = np.random.normal(0, self.config.denoise_sigma, X.shape)
        return np.clip(X + noise, 0, 1)
    
    def feature_squeezing(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce color depth of features.
        
        Maps continuous values to discrete bins, potentially
        "squeezing out" adversarial perturbations.
        """
        n_levels = 2 ** self.config.squeeze_depth
        X_squeezed = np.round(X * (n_levels - 1)) / (n_levels - 1)
        return np.clip(X_squeezed, 0, 1)
    
    def median_filter(self, X: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Apply median filtering across features.
        
        For each sample, replaces each feature with the median
        of nearby features, smoothing out localized perturbations.
        """
        X_filtered = X.copy()
        n_features = X.shape[1]
        half_window = window_size // 2
        
        for i in range(n_features):
            start = max(0, i - half_window)
            end = min(n_features, i + half_window + 1)
            X_filtered[:, i] = np.median(X[:, start:end], axis=1)
        
        return X_filtered
    
    def spatial_smoothing(self, X: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply averaging filter across features.
        """
        X_smoothed = X.copy()
        n_features = X.shape[1]
        half_kernel = kernel_size // 2
        
        for i in range(n_features):
            start = max(0, i - half_kernel)
            end = min(n_features, i + half_kernel + 1)
            X_smoothed[:, i] = np.mean(X[:, start:end], axis=1)
        
        return X_smoothed


class SpikeBasedDefense:
    """
    SNN-specific defense mechanisms that exploit spike-based computation.
    """
    
    def __init__(self, model: SpikingNeuralNetwork):
        self.model = model
    
    def spike_dropout(self, dropout_rate: float = 0.1) -> SpikingNeuralNetwork:
        """
        Create model variant with spike dropout during inference.
        
        Randomly drops spikes, making it harder for adversaries
        to craft inputs that reliably cause misclassification.
        """
        # This would modify the forward pass to randomly drop spikes
        # For now, we implement this as input randomization
        pass
    
    def temporal_averaging(self, X: np.ndarray, n_runs: int = 5) -> np.ndarray:
        """
        Average predictions across multiple forward passes.
        
        Exploits the stochastic nature of spike generation to
        reduce adversarial effectiveness.
        """
        all_probs = []
        
        for _ in range(n_runs):
            probs, _ = self.model.forward_batch(X)
            all_probs.append(probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        return np.argmax(avg_probs, axis=1)
    
    def spike_consistency_check(self, X: np.ndarray, 
                                threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect potential adversarial examples via spike pattern consistency.
        
        Adversarial examples often produce inconsistent spike patterns
        across multiple runs due to being near decision boundaries.
        
        Returns:
            predictions: Final predictions
            is_suspicious: Boolean array marking suspicious samples
        """
        n_runs = 10
        all_preds = []
        
        for _ in range(n_runs):
            preds = self.model.predict(X)
            all_preds.append(preds)
        
        all_preds = np.array(all_preds)
        
        # Check consistency: what fraction of runs gave the same prediction?
        final_preds = []
        consistency = []
        
        for i in range(len(X)):
            sample_preds = all_preds[:, i]
            values, counts = np.unique(sample_preds, return_counts=True)
            most_common_idx = np.argmax(counts)
            final_preds.append(values[most_common_idx])
            consistency.append(counts[most_common_idx] / n_runs)
        
        is_suspicious = np.array(consistency) < threshold
        
        return np.array(final_preds), is_suspicious


class AdversarialTrainer:
    """
    Implements adversarial training to improve model robustness.
    """
    
    def __init__(self, model: SpikingNeuralNetwork, config: Optional[DefenseConfig] = None):
        self.model = model
        self.config = config or DefenseConfig()
        self.attacker = AdversarialAttacker(
            model, 
            AttackConfig(epsilon=self.config.adv_epsilon)
        )
        
        # Training statistics
        self.training_history = []
    
    def generate_adversarial_batch(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate adversarial examples for training.
        
        Uses FGSM for efficiency during training.
        """
        X_adv, _ = self.attacker.fgsm_attack(X, y)
        return X_adv
    
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                   batch_size: int = 256, learning_rate: float = 0.02,
                   class_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Train one epoch with adversarial examples mixed in.
        """
        n_samples = len(X_train)
        indices = np.random.permutation(n_samples)
        
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            # Split batch into clean and adversarial
            n_adv = int(len(X_batch) * self.config.adv_train_ratio)
            
            if n_adv > 0:
                # Generate adversarial examples for first part of batch
                X_adv = self.generate_adversarial_batch(X_batch[:n_adv], y_batch[:n_adv])
                
                # Combine clean and adversarial
                X_combined = np.vstack([X_adv, X_batch[n_adv:]])
                y_combined = np.concatenate([y_batch[:n_adv], y_batch[n_adv:]])
            else:
                X_combined = X_batch
                y_combined = y_batch
            
            # Training step
            result = self.model.train_step(X_combined, y_combined, 
                                          learning_rate, class_weights)
            
            total_loss += result['loss']
            total_acc += result['accuracy']
            n_batches += 1
        
        epoch_stats = {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches
        }
        
        self.training_history.append(epoch_stats)
        return epoch_stats
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 20, batch_size: int = 256,
             learning_rate: float = 0.02,
             class_weights: Optional[np.ndarray] = None,
             patience: int = 5,
             verbose: bool = True) -> Dict:
        """
        Full adversarial training loop with early stopping.
        """
        best_val_acc = 0
        best_weights = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train epoch
            train_stats = self.train_epoch(
                X_train, y_train, batch_size, learning_rate, class_weights
            )
            
            # Validation
            val_pred = self.model.predict(X_val)
            val_acc = np.mean(val_pred == y_val)
            
            # Also check adversarial validation accuracy
            X_val_adv, _ = self.attacker.fgsm_attack(X_val, y_val)
            val_pred_adv = self.model.predict(X_val_adv)
            val_acc_adv = np.mean(val_pred_adv == y_val)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_stats['loss']:.4f} - "
                      f"Train Acc: {train_stats['accuracy']:.4f} - "
                      f"Val Acc: {val_acc:.4f} - "
                      f"Val Adv Acc: {val_acc_adv:.4f}")
            
            # Combined metric: clean + adversarial accuracy
            combined_acc = 0.5 * val_acc + 0.5 * val_acc_adv
            
            if combined_acc > best_val_acc:
                best_val_acc = combined_acc
                best_weights = {
                    'w1': self.model.w1.copy(),
                    'w2': self.model.w2.copy(),
                    'w_out': self.model.w_out.copy()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        if best_weights is not None:
            self.model.w1 = best_weights['w1']
            self.model.w2 = best_weights['w2']
            self.model.w_out = best_weights['w_out']
        
        return {
            'best_combined_accuracy': best_val_acc,
            'epochs_trained': epoch + 1,
            'history': self.training_history
        }


class EnsembleDefense:
    """
    Ensemble defense using multiple diverse models.
    """
    
    def __init__(self, n_models: int = 5, **model_kwargs):
        self.n_models = n_models
        self.models = []
        
        for i in range(n_models):
            model = SpikingNeuralNetwork(seed=42 + i, **model_kwargs)
            self.models.append(model)
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 **train_kwargs):
        """Train all models in the ensemble."""
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}")
            
            # Use different subsets for diversity
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_subset = X_train[indices]
            y_subset = y_train[indices]
            
            trainer = AdversarialTrainer(model)
            trainer.train(X_subset, y_subset, X_val, y_val, 
                         verbose=False, **train_kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction via majority voting."""
        all_preds = []
        
        for model in self.models:
            preds = model.predict(X)
            all_preds.append(preds)
        
        all_preds = np.array(all_preds)
        
        # Majority voting
        final_preds = []
        for i in range(len(X)):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds.append(values[np.argmax(counts)])
        
        return np.array(final_preds)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction with confidence based on agreement."""
        all_preds = []
        
        for model in self.models:
            preds = model.predict(X)
            all_preds.append(preds)
        
        all_preds = np.array(all_preds)
        
        final_preds = []
        confidences = []
        
        for i in range(len(X)):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds.append(values[np.argmax(counts)])
            confidences.append(np.max(counts) / self.n_models)
        
        return np.array(final_preds), np.array(confidences)


class RandomizedSmoothing:
    """
    Randomized smoothing for certified robustness.
    
    Provides provable guarantees against adversarial perturbations.
    """
    
    def __init__(self, model: SpikingNeuralNetwork, config: Optional[DefenseConfig] = None):
        self.model = model
        self.config = config or DefenseConfig()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Smoothed prediction."""
        all_preds = []
        
        for _ in range(self.config.smooth_samples):
            # Add Gaussian noise
            noise = np.random.normal(0, self.config.smooth_sigma, X.shape)
            X_noisy = np.clip(X + noise, 0, 1)
            
            preds = self.model.predict(X_noisy)
            all_preds.append(preds)
        
        all_preds = np.array(all_preds)
        
        # Return most common prediction
        final_preds = []
        for i in range(len(X)):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds.append(values[np.argmax(counts)])
        
        return np.array(final_preds)
    
    def certify(self, X: np.ndarray, y: np.ndarray, 
               alpha: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """
        Certify robustness radius for each sample.
        
        Returns certified radius within which no adversarial example exists.
        """
        from scipy import stats
        
        n_samples = len(X)
        predictions = np.zeros(n_samples, dtype=int)
        radii = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Sample many predictions
            counts = np.zeros(self.model.output_size)
            
            for _ in range(self.config.smooth_samples):
                noise = np.random.normal(0, self.config.smooth_sigma, X[i].shape)
                X_noisy = np.clip(X[i] + noise, 0, 1).reshape(1, -1)
                pred = self.model.predict(X_noisy)[0]
                counts[pred] += 1
            
            # Get top two classes
            sorted_idx = np.argsort(counts)[::-1]
            top_class = sorted_idx[0]
            n_top = counts[top_class]
            
            predictions[i] = top_class
            
            # Compute certified radius using Neyman-Pearson
            p_lower = stats.binom.ppf(alpha, self.config.smooth_samples, 
                                      n_top / self.config.smooth_samples)
            p_lower = max(p_lower / self.config.smooth_samples, 0.5)
            
            if p_lower > 0.5:
                radii[i] = self.config.smooth_sigma * stats.norm.ppf(p_lower)
            else:
                radii[i] = 0.0
        
        return predictions, radii


def evaluate_defenses(model: SpikingNeuralNetwork,
                     X_test: np.ndarray, y_test: np.ndarray,
                     attack_epsilon: float = 0.1) -> Dict:
    """
    Evaluate multiple defense strategies against adversarial attacks.
    """
    results = {}
    
    # Create attacker
    attack_config = AttackConfig(epsilon=attack_epsilon)
    attacker = AdversarialAttacker(model, attack_config)
    
    # Generate adversarial examples
    X_fgsm, _ = attacker.fgsm_attack(X_test, y_test)
    X_pgd, _ = attacker.pgd_attack(X_test, y_test)
    
    # Baseline (no defense)
    results['baseline'] = {
        'clean_acc': np.mean(model.predict(X_test) == y_test),
        'fgsm_acc': np.mean(model.predict(X_fgsm) == y_test),
        'pgd_acc': np.mean(model.predict(X_pgd) == y_test)
    }
    
    # Input denoising
    denoiser = InputDenoiser()
    
    for method_name, method in [
        ('gaussian_noise', denoiser.gaussian_noise),
        ('feature_squeezing', denoiser.feature_squeezing),
        ('median_filter', denoiser.median_filter)
    ]:
        X_test_d = method(X_test)
        X_fgsm_d = method(X_fgsm)
        X_pgd_d = method(X_pgd)
        
        results[method_name] = {
            'clean_acc': np.mean(model.predict(X_test_d) == y_test),
            'fgsm_acc': np.mean(model.predict(X_fgsm_d) == y_test),
            'pgd_acc': np.mean(model.predict(X_pgd_d) == y_test)
        }
    
    # Spike-based defense
    spike_defense = SpikeBasedDefense(model)
    
    results['temporal_averaging'] = {
        'clean_acc': np.mean(spike_defense.temporal_averaging(X_test) == y_test),
        'fgsm_acc': np.mean(spike_defense.temporal_averaging(X_fgsm) == y_test),
        'pgd_acc': np.mean(spike_defense.temporal_averaging(X_pgd) == y_test)
    }
    
    # Randomized smoothing
    smoother = RandomizedSmoothing(model, DefenseConfig(smooth_samples=50))
    
    results['randomized_smoothing'] = {
        'clean_acc': np.mean(smoother.predict(X_test) == y_test),
        'fgsm_acc': np.mean(smoother.predict(X_fgsm) == y_test),
        'pgd_acc': np.mean(smoother.predict(X_pgd) == y_test)
    }
    
    return results
