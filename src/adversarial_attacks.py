"""
Adversarial Attack Module for NeuroIDS
Implements multiple attack strategies against spiking neural networks.

Author: Toby R. Davis
Mississippi State University
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from snn_model import SpikingNeuralNetwork


class AttackType(Enum):
    """Types of adversarial attacks"""
    FGSM = "fgsm"                    # Fast Gradient Sign Method
    PGD = "pgd"                      # Projected Gradient Descent
    SPIKE_TIMING = "spike_timing"   # Spike timing manipulation
    FEATURE_PERTURBATION = "feature_perturbation"
    MIMICRY = "mimicry"             # Traffic mimicry attack
    EVASION = "evasion"             # Combined evasion attack


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks"""
    epsilon: float = 0.1            # Perturbation magnitude
    alpha: float = 0.01             # Step size for iterative attacks
    num_iter: int = 10              # Number of iterations for PGD
    targeted: bool = False          # Whether attack is targeted
    target_class: int = 0           # Target class for targeted attacks
    clip_min: float = 0.0           # Minimum feature value
    clip_max: float = 1.0           # Maximum feature value
    

class AdversarialAttacker:
    """
    Generates adversarial examples against spiking neural networks.
    
    Implements multiple attack strategies tailored for SNNs:
    1. FGSM: Fast gradient-based perturbation
    2. PGD: Iterative projected gradient descent
    3. Spike Timing: Manipulates temporal spike patterns
    4. Feature Perturbation: Strategic feature modification
    5. Mimicry: Makes malicious traffic resemble normal
    """
    
    def __init__(self, model: SpikingNeuralNetwork, config: Optional[AttackConfig] = None):
        self.model = model
        self.config = config or AttackConfig()
        
        # Feature importance scores (computed during analysis)
        self.feature_importance = None
        
        # Attack success statistics
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'avg_perturbation': 0.0
        }
    
    def estimate_gradient(self, X: np.ndarray, y: np.ndarray, 
                          delta: float = 0.01) -> np.ndarray:
        """
        Estimate gradient of loss w.r.t. input using finite differences.
        
        SNNs are non-differentiable due to spike generation, so we use
        numerical gradient estimation.
        
        Args:
            X: Input samples (batch_size, features)
            y: True labels
            delta: Perturbation size for finite differences
            
        Returns:
            Estimated gradients (batch_size, features)
        """
        batch_size, n_features = X.shape
        gradients = np.zeros_like(X)
        
        # Get baseline loss
        probs_base, _ = self.model.forward_batch(X)
        loss_base = self._compute_attack_loss(probs_base, y)
        
        # Estimate gradient for each feature
        for f in range(n_features):
            X_plus = X.copy()
            X_plus[:, f] += delta
            X_plus = np.clip(X_plus, self.config.clip_min, self.config.clip_max)
            
            probs_plus, _ = self.model.forward_batch(X_plus)
            loss_plus = self._compute_attack_loss(probs_plus, y)
            
            gradients[:, f] = (loss_plus - loss_base) / delta
        
        return gradients
    
    def estimate_gradient_fast(self, X: np.ndarray, y: np.ndarray,
                               n_samples: int = 3, sigma: float = 0.01) -> np.ndarray:
        """
        Fast gradient estimation using random perturbations (SPSA-like).
        
        Much faster than full finite differences for high-dimensional inputs.
        """
        batch_size, n_features = X.shape
        gradients = np.zeros_like(X)
        
        for _ in range(n_samples):
            # Random perturbation direction
            delta = np.random.choice([-1, 1], size=X.shape) * sigma
            
            X_plus = np.clip(X + delta, self.config.clip_min, self.config.clip_max)
            X_minus = np.clip(X - delta, self.config.clip_min, self.config.clip_max)
            
            probs_plus, _ = self.model.forward_batch(X_plus)
            probs_minus, _ = self.model.forward_batch(X_minus)
            
            loss_plus = self._compute_attack_loss(probs_plus, y)
            loss_minus = self._compute_attack_loss(probs_minus, y)
            
            # SPSA gradient estimate
            gradients += (loss_plus - loss_minus)[:, np.newaxis] / (2 * delta)
        
        return gradients / n_samples
    
    def _compute_attack_loss(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute per-sample loss for attack optimization.
        
        For untargeted attacks: maximize loss (minimize correct class prob)
        For targeted attacks: minimize loss (maximize target class prob)
        """
        batch_size = len(y)
        
        if self.config.targeted:
            # Targeted: minimize probability of target class (we'll negate later)
            target = np.full(batch_size, self.config.target_class)
            loss = -np.log(probs[np.arange(batch_size), target] + 1e-7)
        else:
            # Untargeted: minimize probability of true class
            loss = -np.log(probs[np.arange(batch_size), y] + 1e-7)
        
        return loss
    
    def fgsm_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Fast Gradient Sign Method attack.
        
        Single-step attack that perturbs in the direction of the gradient sign.
        
        Args:
            X: Clean input samples
            y: True labels
            
        Returns:
            X_adv: Adversarial examples
            info: Attack statistics
        """
        # Estimate gradient
        grad = self.estimate_gradient_fast(X, y)
        
        # Compute perturbation
        if self.config.targeted:
            # For targeted attack, move against gradient (minimize target loss)
            perturbation = -self.config.epsilon * np.sign(grad)
        else:
            # For untargeted attack, move along gradient (maximize loss)
            perturbation = self.config.epsilon * np.sign(grad)
        
        # Apply perturbation
        X_adv = X + perturbation
        X_adv = np.clip(X_adv, self.config.clip_min, self.config.clip_max)
        
        # Compute attack success
        pred_clean = self.model.predict(X)
        pred_adv = self.model.predict(X_adv)
        
        if self.config.targeted:
            success_rate = np.mean(pred_adv == self.config.target_class)
        else:
            success_rate = np.mean(pred_adv != y)
        
        info = {
            'attack_type': 'FGSM',
            'epsilon': self.config.epsilon,
            'success_rate': success_rate,
            'avg_perturbation': np.mean(np.abs(perturbation)),
            'max_perturbation': np.max(np.abs(perturbation))
        }
        
        self._update_stats(success_rate, np.mean(np.abs(perturbation)))
        
        return X_adv, info
    
    def pgd_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Projected Gradient Descent attack.
        
        Iterative attack with projection back to epsilon-ball.
        
        Args:
            X: Clean input samples
            y: True labels
            
        Returns:
            X_adv: Adversarial examples
            info: Attack statistics
        """
        X_adv = X.copy()
        
        # Random initialization within epsilon ball
        X_adv += np.random.uniform(
            -self.config.epsilon, self.config.epsilon, X.shape
        )
        X_adv = np.clip(X_adv, self.config.clip_min, self.config.clip_max)
        
        for i in range(self.config.num_iter):
            # Estimate gradient
            grad = self.estimate_gradient_fast(X_adv, y)
            
            # Gradient step
            if self.config.targeted:
                X_adv = X_adv - self.config.alpha * np.sign(grad)
            else:
                X_adv = X_adv + self.config.alpha * np.sign(grad)
            
            # Project back to epsilon ball
            perturbation = X_adv - X
            perturbation = np.clip(perturbation, -self.config.epsilon, self.config.epsilon)
            X_adv = X + perturbation
            
            # Clip to valid range
            X_adv = np.clip(X_adv, self.config.clip_min, self.config.clip_max)
        
        # Compute attack success
        pred_adv = self.model.predict(X_adv)
        
        if self.config.targeted:
            success_rate = np.mean(pred_adv == self.config.target_class)
        else:
            success_rate = np.mean(pred_adv != y)
        
        info = {
            'attack_type': 'PGD',
            'epsilon': self.config.epsilon,
            'num_iter': self.config.num_iter,
            'success_rate': success_rate,
            'avg_perturbation': np.mean(np.abs(X_adv - X)),
            'max_perturbation': np.max(np.abs(X_adv - X))
        }
        
        self._update_stats(success_rate, np.mean(np.abs(X_adv - X)))
        
        return X_adv, info
    
    def spike_timing_attack(self, X: np.ndarray, y: np.ndarray,
                           timing_noise: float = 0.05) -> Tuple[np.ndarray, Dict]:
        """
        Spike timing manipulation attack.
        
        Exploits the temporal nature of SNNs by perturbing features
        in ways that affect spike timing without changing overall magnitude.
        
        This is SNN-specific: small changes to feature values can shift
        when spikes occur, potentially causing misclassification.
        """
        batch_size, n_features = X.shape
        
        # Analyze which features most affect spike timing
        if self.feature_importance is None:
            self._compute_feature_importance(X, y)
        
        # Perturb high-importance features with timing-aware noise
        X_adv = X.copy()
        
        # Add structured noise that affects spike timing
        # Features near threshold boundaries are most sensitive
        for i in range(batch_size):
            for f in range(n_features):
                if self.feature_importance[f] > np.median(self.feature_importance):
                    # Add noise that pushes value toward/away from spike threshold
                    noise = np.random.uniform(-timing_noise, timing_noise)
                    X_adv[i, f] = np.clip(X_adv[i, f] + noise, 
                                         self.config.clip_min, self.config.clip_max)
        
        # Compute attack success
        pred_adv = self.model.predict(X_adv)
        success_rate = np.mean(pred_adv != y)
        
        info = {
            'attack_type': 'Spike_Timing',
            'timing_noise': timing_noise,
            'success_rate': success_rate,
            'avg_perturbation': np.mean(np.abs(X_adv - X))
        }
        
        self._update_stats(success_rate, np.mean(np.abs(X_adv - X)))
        
        return X_adv, info
    
    def mimicry_attack(self, X_malicious: np.ndarray, X_normal: np.ndarray,
                       y_malicious: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Traffic mimicry attack.
        
        Modifies malicious traffic to have statistical properties
        similar to normal traffic while preserving malicious functionality.
        
        Args:
            X_malicious: Malicious samples to disguise
            X_normal: Normal samples to mimic
            y_malicious: True labels of malicious samples
            
        Returns:
            X_adv: Adversarial (mimicked) samples
            info: Attack statistics
        """
        # Compute statistics of normal traffic
        normal_mean = np.mean(X_normal, axis=0)
        normal_std = np.std(X_normal, axis=0) + 1e-7
        
        # Compute statistics of malicious traffic
        mal_mean = np.mean(X_malicious, axis=0)
        mal_std = np.std(X_malicious, axis=0) + 1e-7
        
        # Transform malicious samples to match normal distribution
        # while staying within epsilon bound
        X_normalized = (X_malicious - mal_mean) / mal_std
        X_mimicked = X_normalized * normal_std + normal_mean
        
        # Bound the perturbation
        perturbation = X_mimicked - X_malicious
        perturbation = np.clip(perturbation, -self.config.epsilon, self.config.epsilon)
        X_adv = X_malicious + perturbation
        X_adv = np.clip(X_adv, self.config.clip_min, self.config.clip_max)
        
        # Compute attack success (malicious classified as normal)
        pred_adv = self.model.predict(X_adv)
        success_rate = np.mean(pred_adv == 0)  # 0 is Normal class
        
        info = {
            'attack_type': 'Mimicry',
            'success_rate': success_rate,
            'avg_perturbation': np.mean(np.abs(X_adv - X_malicious)),
            'samples_evaded': np.sum(pred_adv == 0)
        }
        
        self._update_stats(success_rate, np.mean(np.abs(X_adv - X_malicious)))
        
        return X_adv, info
    
    def combined_evasion_attack(self, X: np.ndarray, y: np.ndarray,
                                X_normal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Combined evasion attack using multiple strategies.
        
        Applies FGSM, then refines with mimicry if normal samples available.
        """
        # Stage 1: FGSM attack
        X_adv, fgsm_info = self.fgsm_attack(X, y)
        
        # Stage 2: If we have normal samples, apply mimicry refinement
        if X_normal is not None:
            # Only refine samples that weren't already evaded
            pred_stage1 = self.model.predict(X_adv)
            still_detected = (pred_stage1 != 0)  # Not classified as Normal
            
            if np.sum(still_detected) > 0:
                X_detected = X_adv[still_detected]
                y_detected = y[still_detected]
                
                # Apply mimicry to still-detected samples
                X_mimicked, _ = self.mimicry_attack(X_detected, X_normal, y_detected)
                X_adv[still_detected] = X_mimicked
        
        # Final evaluation
        pred_final = self.model.predict(X_adv)
        success_rate = np.mean(pred_final != y)
        
        info = {
            'attack_type': 'Combined_Evasion',
            'fgsm_success': fgsm_info['success_rate'],
            'final_success_rate': success_rate,
            'avg_perturbation': np.mean(np.abs(X_adv - X))
        }
        
        return X_adv, info
    
    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                   n_samples: int = 100):
        """
        Compute feature importance based on gradient magnitudes.
        """
        # Use subset for efficiency
        idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X_sample = X[idx]
        y_sample = y[idx]
        
        grad = self.estimate_gradient_fast(X_sample, y_sample)
        self.feature_importance = np.mean(np.abs(grad), axis=0)
    
    def _update_stats(self, success_rate: float, avg_perturbation: float):
        """Update running attack statistics."""
        n = self.attack_stats['total_attacks']
        self.attack_stats['total_attacks'] += 1
        self.attack_stats['successful_attacks'] += success_rate
        
        # Running average of perturbation
        self.attack_stats['avg_perturbation'] = (
            (n * self.attack_stats['avg_perturbation'] + avg_perturbation) / (n + 1)
        )
    
    def get_attack_summary(self) -> Dict:
        """Return summary of attack statistics."""
        total = self.attack_stats['total_attacks']
        if total == 0:
            return {'message': 'No attacks performed yet'}
        
        return {
            'total_attacks': total,
            'avg_success_rate': self.attack_stats['successful_attacks'] / total,
            'avg_perturbation': self.attack_stats['avg_perturbation']
        }


def evaluate_robustness(model: SpikingNeuralNetwork, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       epsilons: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict:
    """
    Comprehensive robustness evaluation across multiple epsilon values.
    
    Args:
        model: Trained SNN model
        X_test: Test features
        y_test: Test labels
        epsilons: List of perturbation magnitudes to test
        
    Returns:
        Dictionary with robustness metrics
    """
    results = {
        'clean_accuracy': np.mean(model.predict(X_test) == y_test),
        'epsilon_results': {}
    }
    
    for eps in epsilons:
        config = AttackConfig(epsilon=eps)
        attacker = AdversarialAttacker(model, config)
        
        # Test each attack type
        eps_results = {}
        
        # FGSM
        X_fgsm, fgsm_info = attacker.fgsm_attack(X_test, y_test)
        eps_results['fgsm'] = {
            'success_rate': fgsm_info['success_rate'],
            'adversarial_accuracy': 1 - fgsm_info['success_rate']
        }
        
        # PGD
        X_pgd, pgd_info = attacker.pgd_attack(X_test, y_test)
        eps_results['pgd'] = {
            'success_rate': pgd_info['success_rate'],
            'adversarial_accuracy': 1 - pgd_info['success_rate']
        }
        
        # Spike Timing
        X_spike, spike_info = attacker.spike_timing_attack(X_test, y_test)
        eps_results['spike_timing'] = {
            'success_rate': spike_info['success_rate'],
            'adversarial_accuracy': 1 - spike_info['success_rate']
        }
        
        results['epsilon_results'][eps] = eps_results
    
    return results
