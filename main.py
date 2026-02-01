#!/usr/bin/env python3
"""
NeuroIDS-Adversarial: Main Experiment Runner
Author: Toby R. Davis, Mississippi State University
"""

import argparse
import time
import sys
import os

# Get the directory where main.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))

import numpy as np

from snn_model import SpikingNeuralNetwork, compute_per_class_metrics
from adversarial_attacks import AdversarialAttacker, AttackConfig
from adversarial_defenses import (
    AdversarialTrainer, DefenseConfig, InputDenoiser,
    SpikeBasedDefense, RandomizedSmoothing
)
from data_loader import (
    load_nslkdd, generate_synthetic_nslkdd, create_balanced_dataset,
    compute_class_weights, train_val_split
)

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_metrics(metrics):
    print(f"\n{'Class':<10} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 50)
    for c, name in enumerate(CLASS_NAMES):
        if c in metrics:
            m = metrics[c]
            print(f"{name:<10} {m['precision']:<10.3f} {m['recall']:<10.3f} "
                  f"{m['f1']:<10.3f} {m['support']:<10}")


def train_model(X_train, y_train, X_val, y_val, epochs=20, verbose=True):
    print_header("Training Baseline NeuroIDS")
    
    model = SpikingNeuralNetwork(input_size=41, hidden1_size=96, hidden2_size=48,
                                 output_size=5, timesteps=75)
    class_weights = compute_class_weights(y_train)
    batch_size, lr, patience = 256, 0.02, 5
    best_val_acc, patience_counter, best_weights = 0, 0, None
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}")
    start = time.time()
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(y_train))
        X_shuf, y_shuf = X_train[perm], y_train[perm]
        total_loss, total_acc, n_batch = 0, 0, 0
        
        for i in range(0, len(y_train), batch_size):
            result = model.train_step(X_shuf[i:i+batch_size], y_shuf[i:i+batch_size], lr, class_weights)
            total_loss += result['loss']
            total_acc += result['accuracy']
            n_batch += 1
        
        val_acc = np.mean(model.predict(X_val) == y_val)
        if verbose:
            print(f"Epoch {epoch+1:2d} - Loss: {total_loss/n_batch:.4f} - "
                  f"Train: {total_acc/n_batch:.4f} - Val: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {'w1': model.w1.copy(), 'w2': model.w2.copy(), 'w_out': model.w_out.copy()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_weights:
        model.w1, model.w2, model.w_out = best_weights['w1'], best_weights['w2'], best_weights['w_out']
    print(f"Done in {time.time()-start:.2f}s, Best Val: {best_val_acc:.4f}")
    return model


def run_attacks(model, X_test, y_test, X_normal):
    print_header("Adversarial Attack Analysis")
    print(f"Clean Accuracy: {np.mean(model.predict(X_test) == y_test):.4f}\n")
    
    for name, eps_list in [("FGSM", [0.01, 0.05, 0.1, 0.15, 0.2]), ("PGD", [0.05, 0.1, 0.15])]:
        print(f"--- {name} Attack ---")
        for eps in eps_list:
            cfg = AttackConfig(epsilon=eps, num_iter=10 if name=="PGD" else 1, alpha=eps/4)
            attacker = AdversarialAttacker(model, cfg)
            X_adv, info = attacker.fgsm_attack(X_test, y_test) if name=="FGSM" else attacker.pgd_attack(X_test, y_test)
            print(f"  eps={eps:.2f}: Success={info['success_rate']:.4f}, Adv_Acc={np.mean(model.predict(X_adv)==y_test):.4f}")
    
    print("\n--- Mimicry Attack ---")
    mal_mask = y_test != 0
    attacker = AdversarialAttacker(model, AttackConfig(epsilon=0.2))
    _, info = attacker.mimicry_attack(X_test[mal_mask], X_normal, y_test[mal_mask])
    print(f"Evasion: {info['success_rate']:.4f} ({info['samples_evaded']}/{np.sum(mal_mask)})")


def run_defenses(model, X_train, y_train, X_val, y_val, X_test, y_test):
    print_header("Defense Analysis")
    
    attacker = AdversarialAttacker(model, AttackConfig(epsilon=0.1))
    X_fgsm, _ = attacker.fgsm_attack(X_test, y_test)
    
    print(f"\n{'Defense':<22} {'Clean':<10} {'FGSM':<10}")
    print("-" * 42)
    print(f"{'Baseline':<22} {np.mean(model.predict(X_test)==y_test):<10.4f} "
          f"{np.mean(model.predict(X_fgsm)==y_test):<10.4f}")
    
    denoiser = InputDenoiser()
    for name, fn in [('Gaussian Noise', denoiser.gaussian_noise), ('Feature Squeeze', denoiser.feature_squeezing)]:
        c = np.mean(model.predict(fn(X_test)) == y_test)
        f = np.mean(model.predict(fn(X_fgsm)) == y_test)
        print(f"{name:<22} {c:<10.4f} {f:<10.4f}")
    
    spike_def = SpikeBasedDefense(model)
    c = np.mean(spike_def.temporal_averaging(X_test) == y_test)
    f = np.mean(spike_def.temporal_averaging(X_fgsm) == y_test)
    print(f"{'Temporal Averaging':<22} {c:<10.4f} {f:<10.4f}")
    
    print("\n--- Adversarial Training ---")
    adv_model = SpikingNeuralNetwork(input_size=41, hidden1_size=96, hidden2_size=48,
                                     output_size=5, timesteps=75, seed=123)
    trainer = AdversarialTrainer(adv_model, DefenseConfig(adv_train_ratio=0.5, adv_epsilon=0.1))
    trainer.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=256,
                 class_weights=compute_class_weights(y_train), patience=5, verbose=True)
    
    X_fgsm_new, _ = AdversarialAttacker(adv_model, AttackConfig(epsilon=0.1)).fgsm_attack(X_test, y_test)
    print(f"\n{'Adv Training':<22} {np.mean(adv_model.predict(X_test)==y_test):<10.4f} "
          f"{np.mean(adv_model.predict(X_fgsm_new)==y_test):<10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='all', choices=['all', 'train', 'attack', 'defense'])
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    print_header("NeuroIDS-Adversarial")
    
    # Use the script directory to find data
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    print(f"Looking for data in: {data_dir}")
    
    # Try to load real NSL-KDD data
    X_train, y_train, X_test, y_test = load_nslkdd(data_dir)
    
    X_train_bal, y_train_bal = create_balanced_dataset(X_train, y_train)
    X_tr, y_tr, X_val, y_val = train_val_split(X_train_bal, y_train_bal)
    X_normal = X_test[y_test == 0]
    
    model = train_model(X_tr, y_tr, X_val, y_val, args.epochs)
    
    if args.experiment in ['all', 'train']:
        print_header("Test Evaluation")
        pred = model.predict(X_test)
        print(f"Accuracy: {np.mean(pred == y_test):.4f}")
        print_metrics(compute_per_class_metrics(y_test, pred))
    
    if args.experiment in ['all', 'attack']:
        run_attacks(model, X_test, y_test, X_normal)
    
    if args.experiment in ['all', 'defense']:
        run_defenses(model, X_tr, y_tr, X_val, y_val, X_test, y_test)
    
    print_header("Complete")


if __name__ == '__main__':
    main()