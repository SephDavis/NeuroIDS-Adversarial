#!/usr/bin/env python3
"""
NeuroIDS-Adversarial: Comprehensive Experiment Runner
Generates all data for every table in the revised paper.

Author: Toby R. Davis, Mississippi State University

Tables produced:
  Table II   - Baseline SNN Performance
  Table III  - FGSM/PGD Attack Results
  Table IV   - SPSA Sample Count Ablation
  Table V    - Per-Class Recall Under FGSM
  Table VI   - Dominant Misclassification Under FGSM
  Table VII  - Spike Timing Attack Results
  Table VIII - Mimicry Evasion by Class
  Table IX   - Mimicry Evasion vs Epsilon
  Table X    - Defense Performance Comparison
  Table XI   - Temporal Averaging vs Adaptive Adversary
  Table XII  - Adversarial Training: SNN vs Conventional
"""

import argparse
import time
import sys
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np

from snn_model import SpikingNeuralNetwork, compute_per_class_metrics
from adversarial_attacks import AdversarialAttacker, AttackConfig
from adversarial_defenses import (
    AdversarialTrainer, DefenseConfig, InputDenoiser,
    SpikeBasedDefense, RandomizedSmoothing
)
from data_loader import (
    load_nslkdd, create_balanced_dataset,
    compute_class_weights, train_val_split, CLASS_NAMES
)


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_table(headers, rows, title=""):
    """Pretty-print a table."""
    if title:
        print(f"\n  {title}")
        print(f"  {'-' * 60}")
    
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    
    header_str = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(f"  {header_str}")
    print(f"  {'─' * len(header_str)}")
    
    for row in rows:
        row_str = "  ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths))
        print(f"  {row_str}")


def train_baseline(X_train, y_train, X_val, y_val, epochs=20, verbose=True):
    """Train baseline model with early stopping."""
    print_header("Training Baseline NeuroIDS (Output-Layer-Only)")
    
    model = SpikingNeuralNetwork(
        input_size=41, hidden1_size=96, hidden2_size=48,
        output_size=5, timesteps=75, seed=42
    )
    class_weights = compute_class_weights(y_train)
    batch_size, lr, patience = 256, 0.02, 5
    best_val_acc, patience_counter, best_weights = 0, 0, None
    best_epoch = 0
    
    print(f"  Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    start = time.time()
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(y_train))
        X_shuf, y_shuf = X_train[perm], y_train[perm]
        total_loss, total_acc, n_batch = 0, 0, 0
        
        for i in range(0, len(y_train), batch_size):
            result = model.train_step(
                X_shuf[i:i+batch_size], y_shuf[i:i+batch_size],
                lr, class_weights
            )
            total_loss += result['loss']
            total_acc += result['accuracy']
            n_batch += 1
        
        val_pred = model.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        
        if verbose:
            print(f"  Epoch {epoch+1:2d} | Loss: {total_loss/n_batch:.4f} | "
                  f"Train: {total_acc/n_batch:.4f} | Val: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_weights = {
                'w1': model.w1.copy(), 'w2': model.w2.copy(),
                'w_out': model.w_out.copy(),
                'class_scales': model.class_scales.copy()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    if best_weights:
        model.w1 = best_weights['w1']
        model.w2 = best_weights['w2']
        model.w_out = best_weights['w_out']
        model.class_scales = best_weights['class_scales']
    
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s | Best val: {best_val_acc:.4f} at epoch {best_epoch}")
    return model, best_val_acc


def table_ii_baseline(model, X_test, y_test):
    """Table II: Baseline SNN Performance on NSL-KDD Test Set."""
    print_header("TABLE II: Baseline SNN Performance")
    
    pred = model.predict(X_test)
    acc = np.mean(pred == y_test)
    metrics = compute_per_class_metrics(y_test, pred)
    
    headers = ["Class", "Precision", "Recall", "F1", "Support"]
    rows = []
    for c, name in enumerate(CLASS_NAMES):
        if c in metrics:
            m = metrics[c]
            rows.append([name, f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                        f"{m['f1']:.3f}", str(m['support'])])
    rows.append(["Overall Acc", "", f"{acc:.3f}", "", str(len(y_test))])
    
    print_table(headers, rows)
    return acc, metrics


def table_iii_gradient_attacks(model, X_test, y_test):
    """Table III: FGSM and PGD Attack Results."""
    print_header("TABLE III: Adversarial Attack Results (Accuracy Under Attack)")
    
    fgsm_epsilons = [0.01, 0.05, 0.10, 0.15, 0.20]
    pgd_epsilons = [0.05, 0.10, 0.15]
    
    results = {}
    headers = ["Epsilon", "FGSM Acc", "PGD Acc", "FGSM Misc Rate"]
    rows = []
    
    clean_acc = np.mean(model.predict(X_test) == y_test)
    rows.append(["0.00 (clean)", f"{clean_acc:.3f}", f"{clean_acc:.3f}", "---"])
    results['clean'] = clean_acc
    
    for eps in fgsm_epsilons:
        # FGSM
        cfg = AttackConfig(epsilon=eps)
        attacker = AdversarialAttacker(model, cfg)
        X_fgsm, fgsm_info = attacker.fgsm_attack(X_test, y_test)
        fgsm_acc = np.mean(model.predict(X_fgsm) == y_test)
        
        # PGD (only at select epsilons)
        pgd_str = "---"
        if eps in pgd_epsilons:
            cfg_pgd = AttackConfig(epsilon=eps, num_iter=10, alpha=eps/4)
            attacker_pgd = AdversarialAttacker(model, cfg_pgd)
            X_pgd, pgd_info = attacker_pgd.pgd_attack(X_test, y_test)
            pgd_acc = np.mean(model.predict(X_pgd) == y_test)
            pgd_str = f"{pgd_acc:.3f}"
            results[f'pgd_{eps}'] = pgd_acc
        
        misc_rate = 1.0 - fgsm_acc
        rows.append([f"{eps:.2f}", f"{fgsm_acc:.3f}", pgd_str, f"{misc_rate:.3f}"])
        results[f'fgsm_{eps}'] = fgsm_acc
    
    print_table(headers, rows)
    return results


def table_iv_spsa_ablation(model, X_test, y_test, eps=0.10):
    """Table IV: Effect of SPSA Sample Count on Attack Strength."""
    print_header("TABLE IV: SPSA Sample Count Ablation (eps=0.10)")
    
    spsa_counts = [5, 10, 20, 50]
    headers = ["SPSA Samples N", "FGSM Acc", "PGD Acc"]
    rows = []
    results = {}
    
    for n in spsa_counts:
        # FGSM with varying SPSA samples
        cfg = AttackConfig(epsilon=eps)
        attacker = AdversarialAttacker(model, cfg)
        
        # Override SPSA sample count in gradient estimation
        X_fgsm = _fgsm_with_spsa_n(model, X_test, y_test, eps, n)
        fgsm_acc = np.mean(model.predict(X_fgsm) == y_test)
        
        # PGD with varying SPSA samples
        X_pgd = _pgd_with_spsa_n(model, X_test, y_test, eps, n, num_iter=10)
        pgd_acc = np.mean(model.predict(X_pgd) == y_test)
        
        label = f"{n}" + (" (default)" if n == 5 else "")
        rows.append([label, f"{fgsm_acc:.3f}", f"{pgd_acc:.3f}"])
        results[n] = {'fgsm': fgsm_acc, 'pgd': pgd_acc}
    
    print_table(headers, rows)
    return results


def _spsa_gradient(model, X, y, n_samples=5, sigma=0.01):
    """Compute SPSA gradient with configurable sample count."""
    batch_size, n_features = X.shape
    gradients = np.zeros_like(X)
    
    for _ in range(n_samples):
        delta = np.random.choice([-1, 1], size=X.shape) * sigma
        X_plus = np.clip(X + delta, 0, 1)
        X_minus = np.clip(X - delta, 0, 1)
        
        probs_plus, _ = model.forward_batch(X_plus)
        probs_minus, _ = model.forward_batch(X_minus)
        
        # Loss: negative log prob of true class
        loss_plus = -np.log(probs_plus[np.arange(len(y)), y] + 1e-7)
        loss_minus = -np.log(probs_minus[np.arange(len(y)), y] + 1e-7)
        
        gradients += (loss_plus - loss_minus)[:, np.newaxis] / (2 * delta)
    
    return gradients / n_samples


def _fgsm_with_spsa_n(model, X, y, eps, n_samples):
    """FGSM attack with configurable SPSA sample count."""
    grad = _spsa_gradient(model, X, y, n_samples=n_samples)
    X_adv = X + eps * np.sign(grad)
    return np.clip(X_adv, 0, 1)


def _pgd_with_spsa_n(model, X, y, eps, n_samples, num_iter=10):
    """PGD attack with configurable SPSA sample count."""
    alpha = eps / 4
    X_adv = X.copy()
    X_adv += np.random.uniform(-eps, eps, X.shape)
    X_adv = np.clip(X_adv, 0, 1)
    
    for _ in range(num_iter):
        grad = _spsa_gradient(model, X_adv, y, n_samples=n_samples)
        X_adv = X_adv + alpha * np.sign(grad)
        perturbation = np.clip(X_adv - X, -eps, eps)
        X_adv = np.clip(X + perturbation, 0, 1)
    
    return X_adv


def table_v_perclass_fgsm(model, X_test, y_test):
    """Table V: Per-Class Recall Under FGSM Attack."""
    print_header("TABLE V: Per-Class Recall Under FGSM Attack")
    
    epsilons = [0.05, 0.10, 0.20]
    
    # Clean recall
    pred_clean = model.predict(X_test)
    clean_metrics = compute_per_class_metrics(y_test, pred_clean)
    
    headers = ["Class", "Clean"] + [f"eps={e}" for e in epsilons]
    rows = []
    results = {'clean': {}, 'attacked': {}}
    
    # Compute attacked recalls
    attacked_metrics = {}
    for eps in epsilons:
        cfg = AttackConfig(epsilon=eps)
        attacker = AdversarialAttacker(model, cfg)
        X_fgsm, _ = attacker.fgsm_attack(X_test, y_test)
        pred_adv = model.predict(X_fgsm)
        attacked_metrics[eps] = compute_per_class_metrics(y_test, pred_adv)
    
    for c, name in enumerate(CLASS_NAMES):
        row = [name]
        clean_recall = clean_metrics[c]['recall'] if c in clean_metrics else 0
        row.append(f"{clean_recall:.3f}")
        results['clean'][name] = clean_recall
        results['attacked'][name] = {}
        
        for eps in epsilons:
            adv_recall = attacked_metrics[eps][c]['recall'] if c in attacked_metrics[eps] else 0
            row.append(f"{adv_recall:.3f}")
            results['attacked'][name][eps] = adv_recall
        
        rows.append(row)
    
    print_table(headers, rows)
    return results


def table_vi_misclassification_direction(model, X_test, y_test, eps=0.10):
    """Table VI: Dominant Misclassification Under FGSM."""
    print_header("TABLE VI: Dominant Misclassification Direction (FGSM eps=0.10)")
    
    cfg = AttackConfig(epsilon=eps)
    attacker = AdversarialAttacker(model, cfg)
    X_fgsm, _ = attacker.fgsm_attack(X_test, y_test)
    pred_adv = model.predict(X_fgsm)
    
    headers = ["True Class", "Top Misclass Target", "% of Errors"]
    rows = []
    results = {}
    
    for c, name in enumerate(CLASS_NAMES):
        mask = (y_test == c)
        if np.sum(mask) == 0:
            continue
        
        preds_for_class = pred_adv[mask]
        wrong_mask = preds_for_class != c
        n_errors = np.sum(wrong_mask)
        
        if n_errors == 0:
            rows.append([name, "N/A", "0.0%"])
            results[name] = {'target': 'N/A', 'pct': 0.0}
            continue
        
        wrong_preds = preds_for_class[wrong_mask]
        error_classes, error_counts = np.unique(wrong_preds, return_counts=True)
        top_idx = np.argmax(error_counts)
        top_class = CLASS_NAMES[error_classes[top_idx]]
        top_pct = error_counts[top_idx] / n_errors * 100
        
        rows.append([name, top_class, f"{top_pct:.1f}%"])
        results[name] = {'target': top_class, 'pct': top_pct}
    
    print_table(headers, rows)
    return results


def table_vii_spike_timing(model, X_test, y_test):
    """Table VII: Spike Timing Attack Results."""
    print_header("TABLE VII: Spike Timing Attack Results")
    
    etas = [0.05, 0.10, 0.15, 0.20]
    
    clean_acc = np.mean(model.predict(X_test) == y_test)
    
    headers = ["Timing Noise η", "Accuracy", "Δ from Clean"]
    rows = [["0.00 (clean)", f"{clean_acc:.3f}", "---"]]
    results = {'clean': clean_acc}
    
    for eta in etas:
        cfg = AttackConfig(epsilon=eta)
        attacker = AdversarialAttacker(model, cfg)
        X_spike, info = attacker.spike_timing_attack(X_test, y_test, timing_noise=eta)
        spike_acc = np.mean(model.predict(X_spike) == y_test)
        delta = spike_acc - clean_acc
        
        rows.append([f"{eta:.2f}", f"{spike_acc:.3f}", f"{delta:+.1%}"])
        results[eta] = spike_acc
    
    print_table(headers, rows)
    return results


def table_viii_mimicry_perclass(model, X_test, y_test, X_normal, eps=0.2):
    """Table VIII: Mimicry Attack Evasion Rate by Class."""
    print_header("TABLE VIII: Mimicry Evasion by Attack Class (eps=0.2)")
    
    headers = ["Class", "Samples", "Evaded", "Evasion %"]
    rows = []
    results = {}
    
    total_samples = 0
    total_evaded = 0
    
    for c in range(1, 5):  # Skip Normal (class 0)
        name = CLASS_NAMES[c]
        mask = (y_test == c)
        X_class = X_test[mask]
        y_class = y_test[mask]
        n_samples = len(X_class)
        
        if n_samples == 0:
            continue
        
        cfg = AttackConfig(epsilon=eps)
        attacker = AdversarialAttacker(model, cfg)
        X_mim, info = attacker.mimicry_attack(X_class, X_normal, y_class)
        
        pred_mim = model.predict(X_mim)
        n_evaded = np.sum(pred_mim == 0)  # Classified as Normal
        evasion_rate = n_evaded / n_samples
        
        rows.append([name, str(n_samples), str(n_evaded), f"{evasion_rate:.1%}"])
        results[name] = {'samples': n_samples, 'evaded': n_evaded, 'rate': evasion_rate}
        
        total_samples += n_samples
        total_evaded += n_evaded
    
    overall_rate = total_evaded / total_samples if total_samples > 0 else 0
    rows.append(["All Attacks", str(total_samples), str(total_evaded), f"{overall_rate:.1%}"])
    results['All'] = {'samples': total_samples, 'evaded': total_evaded, 'rate': overall_rate}
    
    print_table(headers, rows)
    return results


def table_ix_mimicry_vs_eps(model, X_test, y_test, X_normal):
    """Table IX: Mimicry Attack Evasion Rate vs Perturbation Budget."""
    print_header("TABLE IX: Mimicry Evasion vs Epsilon")
    
    epsilons = [0.05, 0.10, 0.15, 0.20]
    mal_mask = y_test != 0
    X_mal = X_test[mal_mask]
    y_mal = y_test[mal_mask]
    
    headers = ["Epsilon", "Evasion Rate"]
    rows = []
    results = {}
    
    for eps in epsilons:
        cfg = AttackConfig(epsilon=eps)
        attacker = AdversarialAttacker(model, cfg)
        X_mim, info = attacker.mimicry_attack(X_mal, X_normal, y_mal)
        
        pred_mim = model.predict(X_mim)
        evasion = np.mean(pred_mim == 0)
        
        rows.append([f"{eps:.2f}", f"{evasion:.1%}"])
        results[eps] = evasion
    
    print_table(headers, rows)
    return results


def table_x_defenses(model, X_test, y_test, eps=0.10):
    """Table X: Defense Performance Comparison."""
    print_header("TABLE X: Defense Performance (FGSM eps=0.10)")
    
    # Generate FGSM adversarial examples
    cfg = AttackConfig(epsilon=eps)
    attacker = AdversarialAttacker(model, cfg)
    X_fgsm, _ = attacker.fgsm_attack(X_test, y_test)
    
    headers = ["Defense", "Clean Acc", "FGSM Acc"]
    rows = []
    results = {}
    
    # Baseline
    clean_acc = np.mean(model.predict(X_test) == y_test)
    fgsm_acc = np.mean(model.predict(X_fgsm) == y_test)
    rows.append(["Baseline (no defense)", f"{clean_acc:.3f}", f"{fgsm_acc:.3f}"])
    results['baseline'] = {'clean': clean_acc, 'fgsm': fgsm_acc}
    
    # Gaussian Noise
    denoiser = InputDenoiser()
    gn_clean = np.mean(model.predict(denoiser.gaussian_noise(X_test)) == y_test)
    gn_fgsm = np.mean(model.predict(denoiser.gaussian_noise(X_fgsm)) == y_test)
    rows.append(["Gaussian Noise", f"{gn_clean:.3f}", f"{gn_fgsm:.3f}"])
    results['gaussian'] = {'clean': gn_clean, 'fgsm': gn_fgsm}
    
    # Feature Squeezing
    fs_clean = np.mean(model.predict(denoiser.feature_squeezing(X_test)) == y_test)
    fs_fgsm = np.mean(model.predict(denoiser.feature_squeezing(X_fgsm)) == y_test)
    rows.append(["Feature Squeezing", f"{fs_clean:.3f}", f"{fs_fgsm:.3f}"])
    results['squeeze'] = {'clean': fs_clean, 'fgsm': fs_fgsm}
    
    # Temporal Averaging
    spike_def = SpikeBasedDefense(model)
    ta_clean = np.mean(spike_def.temporal_averaging(X_test) == y_test)
    ta_fgsm = np.mean(spike_def.temporal_averaging(X_fgsm) == y_test)
    rows.append(["Temporal Averaging", f"{ta_clean:.3f}", f"{ta_fgsm:.3f}"])
    results['temporal_avg'] = {'clean': ta_clean, 'fgsm': ta_fgsm}
    
    print_table(headers, rows)
    return results, X_fgsm


def table_xi_adaptive_adversary(model, X_test, y_test, eps=0.10):
    """Table XI: Temporal Averaging vs Adaptive Adversary."""
    print_header("TABLE XI: Temporal Averaging vs Adaptive Adversary (eps=0.10)")
    
    spike_def = SpikeBasedDefense(model)
    
    # No defense, no attack
    clean_acc = np.mean(model.predict(X_test) == y_test)
    
    # No defense, standard FGSM
    cfg = AttackConfig(epsilon=eps)
    attacker = AdversarialAttacker(model, cfg)
    X_fgsm, _ = attacker.fgsm_attack(X_test, y_test)
    nodef_fgsm_acc = np.mean(model.predict(X_fgsm) == y_test)
    
    # Temporal averaging, standard FGSM
    ta_std_acc = np.mean(spike_def.temporal_averaging(X_fgsm) == y_test)
    
    # Temporal averaging, adaptive FGSM (average gradients over M_a passes)
    M_a = 5
    grad_sum = np.zeros_like(X_test, dtype=np.float64)
    for _ in range(M_a):
        grad = _spsa_gradient(model, X_test, y_test, n_samples=5)
        grad_sum += grad
    grad_avg = grad_sum / M_a
    
    X_adaptive = X_test + eps * np.sign(grad_avg)
    X_adaptive = np.clip(X_adaptive, 0, 1)
    ta_adaptive_acc = np.mean(spike_def.temporal_averaging(X_adaptive) == y_test)
    
    headers = ["Configuration", "Accuracy", "Δ from Clean"]
    rows = [
        ["No defense, no attack", f"{clean_acc:.3f}", "---"],
        ["No defense, FGSM", f"{nodef_fgsm_acc:.3f}",
         f"{nodef_fgsm_acc - clean_acc:+.1%}"],
        ["Temp avg, standard FGSM", f"{ta_std_acc:.3f}",
         f"{ta_std_acc - clean_acc:+.1%}"],
        ["Temp avg, adaptive FGSM", f"{ta_adaptive_acc:.3f}",
         f"{ta_adaptive_acc - clean_acc:+.1%}"],
    ]
    
    results = {
        'clean': clean_acc,
        'nodef_fgsm': nodef_fgsm_acc,
        'ta_std': ta_std_acc,
        'ta_adaptive': ta_adaptive_acc
    }
    
    print_table(headers, rows)
    return results


def table_xii_adversarial_training(model_baseline, X_train, y_train, X_val, y_val,
                                     X_test, y_test, epochs=15):
    """Table X row for adversarial training + Table XII comparison."""
    print_header("Adversarial Training")
    
    # Train adversarially hardened model
    adv_model = SpikingNeuralNetwork(
        input_size=41, hidden1_size=96, hidden2_size=48,
        output_size=5, timesteps=75, seed=123
    )
    
    trainer = AdversarialTrainer(
        adv_model,
        DefenseConfig(adv_train_ratio=0.5, adv_epsilon=0.1)
    )
    
    class_weights = compute_class_weights(y_train)
    
    train_result = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=256,
        learning_rate=0.02,
        class_weights=class_weights,
        patience=5, verbose=True
    )
    
    # Evaluate
    adv_clean_acc = np.mean(adv_model.predict(X_test) == y_test)
    
    # Generate FGSM against the adversarially trained model
    cfg = AttackConfig(epsilon=0.1)
    attacker = AdversarialAttacker(adv_model, cfg)
    X_fgsm_new, _ = attacker.fgsm_attack(X_test, y_test)
    adv_fgsm_acc = np.mean(adv_model.predict(X_fgsm_new) == y_test)
    
    gap = adv_clean_acc - adv_fgsm_acc
    
    print_header("TABLE XII: Adversarial Training Results")
    
    # Compare with baseline
    baseline_clean = np.mean(model_baseline.predict(X_test) == y_test)
    cfg_b = AttackConfig(epsilon=0.1)
    att_b = AdversarialAttacker(model_baseline, cfg_b)
    X_fgsm_b, _ = att_b.fgsm_attack(X_test, y_test)
    baseline_fgsm = np.mean(model_baseline.predict(X_fgsm_b) == y_test)
    baseline_gap = baseline_clean - baseline_fgsm
    
    headers = ["Model", "Clean Acc", "FGSM Acc", "Gap (pp)"]
    rows = [
        ["Baseline SNN", f"{baseline_clean:.3f}", f"{baseline_fgsm:.3f}",
         f"{baseline_gap*100:.1f}"],
        ["SNN + Adv Train", f"{adv_clean_acc:.3f}", f"{adv_fgsm_acc:.3f}",
         f"{gap*100:.1f}"],
    ]
    
    print_table(headers, rows)
    
    # Per-class metrics for adversarially trained model
    pred_adv_clean = adv_model.predict(X_test)
    adv_metrics = compute_per_class_metrics(y_test, pred_adv_clean)
    
    print("\n  Adversarially Trained Model - Per-Class Performance:")
    for c, name in enumerate(CLASS_NAMES):
        if c in adv_metrics:
            m = adv_metrics[c]
            print(f"    {name:<10} Prec: {m['precision']:.3f}  "
                  f"Recall: {m['recall']:.3f}  F1: {m['f1']:.3f}")
    
    return {
        'adv_clean': adv_clean_acc,
        'adv_fgsm': adv_fgsm_acc,
        'gap': gap,
        'baseline_clean': baseline_clean,
        'baseline_fgsm': baseline_fgsm,
        'baseline_gap': baseline_gap
    }, adv_model


def confusion_matrix_analysis(model, X_test, y_test, label="Baseline"):
    """Full confusion matrix for the model."""
    print_header(f"Confusion Matrix ({label})")
    
    pred = model.predict(X_test)
    n_classes = 5
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_c in range(n_classes):
        for pred_c in range(n_classes):
            cm[true_c, pred_c] = np.sum((y_test == true_c) & (pred == pred_c))
    
    # Print
    header = f"  {'':>8}" + "".join(f"{CLASS_NAMES[c]:>8}" for c in range(n_classes))
    print(header)
    print(f"  {'─' * len(header)}")
    for true_c in range(n_classes):
        row = f"  {CLASS_NAMES[true_c]:>8}"
        for pred_c in range(n_classes):
            row += f"{cm[true_c, pred_c]:>8}"
        print(row)
    
    return cm


def save_results(all_results, filepath="results.json"):
    """Save all results to JSON for later reference."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    converted = convert(all_results)
    with open(filepath, 'w') as f:
        json.dump(converted, f, indent=2)
    print(f"\n  Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="NeuroIDS-Adversarial Experiments")
    parser.add_argument('--experiment', default='all',
                       choices=['all', 'train', 'attack', 'defense', 'quick'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--adv-epochs', type=int, default=15)
    parser.add_argument('--test-subset', type=int, default=0,
                       help="Use subset of test data (0=full)")
    parser.add_argument('--save', type=str, default='results.json',
                       help="Save results to JSON file")
    args = parser.parse_args()
    
    print_header("NeuroIDS-Adversarial: Comprehensive Experiment Suite")
    print(f"  Experiment: {args.experiment}")
    print(f"  Epochs: {args.epochs} (baseline), {args.adv_epochs} (adversarial)")
    
    all_results = {}
    
    # ─── Load Data ───
    print_header("Loading Data")
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    X_train_raw, y_train_raw, X_test, y_test = load_nslkdd(data_dir)
    
    X_train_bal, y_train_bal = create_balanced_dataset(X_train_raw, y_train_raw)
    X_tr, y_tr, X_val, y_val = train_val_split(X_train_bal, y_train_bal)
    X_normal = X_test[y_test == 0]
    
    print(f"  Train: {len(y_tr)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Normal test samples: {len(X_normal)}")
    
    # Optional: use test subset for speed
    if args.test_subset > 0:
        idx = np.random.RandomState(42).choice(len(y_test), args.test_subset, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
        X_normal = X_test[y_test == 0]
        print(f"  Using test subset: {len(y_test)} samples")
    
    # ─── Train Baseline ───
    model, best_val_acc = train_baseline(X_tr, y_tr, X_val, y_val, args.epochs)
    all_results['best_val_acc'] = best_val_acc
    
    # ─── Table II: Baseline Performance ───
    clean_acc, baseline_metrics = table_ii_baseline(model, X_test, y_test)
    all_results['table_ii'] = {'accuracy': clean_acc}
    
    # ─── Confusion Matrix ───
    cm = confusion_matrix_analysis(model, X_test, y_test, "Baseline")
    all_results['confusion_matrix'] = cm
    
    if args.experiment in ['all', 'attack', 'quick']:
        # ─── Table III: Gradient Attacks ───
        all_results['table_iii'] = table_iii_gradient_attacks(model, X_test, y_test)
        
        # ─── Table V: Per-Class FGSM ───
        all_results['table_v'] = table_v_perclass_fgsm(model, X_test, y_test)
        
        # ─── Table VI: Misclassification Direction ───
        all_results['table_vi'] = table_vi_misclassification_direction(model, X_test, y_test)
        
        # ─── Table VII: Spike Timing ───
        all_results['table_vii'] = table_vii_spike_timing(model, X_test, y_test)
        
        # ─── Table VIII: Mimicry Per-Class ───
        all_results['table_viii'] = table_viii_mimicry_perclass(model, X_test, y_test, X_normal)
        
        # ─── Table IX: Mimicry vs Epsilon ───
        all_results['table_ix'] = table_ix_mimicry_vs_eps(model, X_test, y_test, X_normal)
    
    if args.experiment in ['all', 'attack'] and args.experiment != 'quick':
        # ─── Table IV: SPSA Ablation (slower) ───
        all_results['table_iv'] = table_iv_spsa_ablation(model, X_test, y_test)
    
    if args.experiment in ['all', 'defense', 'quick']:
        # ─── Table X: Defenses ───
        defense_results, X_fgsm = table_x_defenses(model, X_test, y_test)
        all_results['table_x'] = defense_results
        
        # ─── Table XI: Adaptive Adversary ───
        all_results['table_xi'] = table_xi_adaptive_adversary(model, X_test, y_test)
        
        # ─── Table XII: Adversarial Training ───
        adv_results, adv_model = table_xii_adversarial_training(
            model, X_tr, y_tr, X_val, y_val, X_test, y_test,
            epochs=args.adv_epochs
        )
        all_results['table_xii'] = adv_results
        
        # Adversarially trained model confusion matrix
        confusion_matrix_analysis(adv_model, X_test, y_test, "Adversarially Trained")
    
    # ─── Save Results ───
    save_results(all_results, args.save)
    
    print_header("ALL EXPERIMENTS COMPLETE")
    print(f"  Baseline accuracy: {clean_acc:.4f}")
    print(f"  Results saved to: {args.save}")


if __name__ == '__main__':
    main()
