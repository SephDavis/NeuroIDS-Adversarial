"""
Data Loading and Preprocessing for NSL-KDD Dataset
Author: Toby R. Davis
Mississippi State University
"""

import numpy as np
import os
from typing import Tuple, Dict, Optional


CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

ATTACK_CATEGORIES = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}

ATTACK_MAPPING = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos',
    'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l', 'phf': 'r2l',
    'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
}


def load_nslkdd(data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NSL-KDD dataset from files or generate synthetic data."""
    
    possible_paths = [
        (os.path.join(data_dir, 'KDDTrain+.txt'), os.path.join(data_dir, 'KDDTest+.txt')),
        (os.path.join(data_dir, 'NSL-KDD', 'KDDTrain+.txt'), os.path.join(data_dir, 'NSL-KDD', 'KDDTest+.txt')),
        (os.path.join(data_dir, 'nsl-kdd', 'KDDTrain+.txt'), os.path.join(data_dir, 'nsl-kdd', 'KDDTest+.txt')),
        ('./KDDTrain+.txt', './KDDTest+.txt'),
    ]
    
    for train_file, test_file in possible_paths:
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"Loading NSL-KDD from: {train_file}")
            X_train, y_train = _load_nslkdd_file(train_file)
            X_test, y_test = _load_nslkdd_file(test_file)
            return X_train, y_train, X_test, y_test
    
    print("NSL-KDD files not found. Generating synthetic data...")
    print("(Place KDDTrain+.txt and KDDTest+.txt in ./data/ or ./data/NSL-KDD/)")
    return generate_synthetic_nslkdd()


def _load_nslkdd_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single NSL-KDD file."""
    
    protocols = {'tcp': 0, 'udp': 1, 'icmp': 2}
    services = {}
    flags = {}
    
    X_list = []
    y_list = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 42:
                continue
            
            features = []
            
            try:
                features.append(float(parts[0]))
            except:
                features.append(0.0)
            
            proto = parts[1].lower()
            if proto not in protocols:
                protocols[proto] = len(protocols)
            features.append(float(protocols[proto]))
            
            service = parts[2].lower()
            if service not in services:
                services[service] = len(services)
            features.append(float(services[service]))
            
            flag = parts[3].lower()
            if flag not in flags:
                flags[flag] = len(flags)
            features.append(float(flags[flag]))
            
            for i in range(4, 41):
                try:
                    features.append(float(parts[i]))
                except:
                    features.append(0.0)
            
            X_list.append(features)
            
            attack_type = parts[41].lower().replace('.', '').strip()
            category = ATTACK_MAPPING.get(attack_type, 'normal')
            y_list.append(ATTACK_CATEGORIES[category])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    X = normalize_features(X)
    
    print(f"  Loaded {len(y)} samples")
    return X, y


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 1] range using min-max scaling."""
    X_min = np.min(X, axis=0, keepdims=True)
    X_max = np.max(X, axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range
    return np.clip(X_norm, 0, 1).astype(np.float32)


def generate_synthetic_nslkdd(n_train: int = 125973, n_test: int = 22544,
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """Generate synthetic NSL-KDD-like data with well-separated classes."""
    np.random.seed(seed)
    n_features = 41

    train_dist = np.array([0.535, 0.365, 0.093, 0.008, 0.0004])
    test_dist = np.array([0.431, 0.331, 0.107, 0.122, 0.009])

    # Class centroids - distinct feature patterns for each class
    centroids = np.zeros((5, n_features))
    
    # Normal: low activity, logged in, stable connections
    centroids[0] = [0.15, 0.3, 0.25, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1,
                    0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.8, 0.2,
                    0.1, 0.4, 0.4, 0.8, 0.2, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05]
    
    # DoS: high bytes, high counts, flooding patterns
    centroids[1] = [0.05, 0.8, 0.5, 0.7, 0.95, 0.95, 0.0, 0.1, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.95, 0.95, 0.7, 0.7, 0.1, 0.1, 0.95, 0.05,
                    0.02, 0.95, 0.95, 0.95, 0.05, 0.9, 0.02, 0.7, 0.7, 0.1, 0.1]
    
    # Probe: scanning, many services, low bytes
    centroids[2] = [0.02, 0.5, 0.9, 0.3, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.8, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.9,
                    0.8, 0.3, 0.1, 0.1, 0.9, 0.1, 0.8, 0.1, 0.1, 0.7, 0.7]
    
    # R2L: failed logins, compromised, medium activity
    centroids[3] = [0.3, 0.3, 0.4, 0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5,
                    0.6, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0,
                    0.0, 0.3, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.5, 0.5,
                    0.3, 0.2, 0.2, 0.5, 0.5, 0.2, 0.3, 0.1, 0.1, 0.2, 0.2]
    
    # U2R: root access, shells, privilege escalation
    centroids[4] = [0.4, 0.2, 0.3, 0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 0.7,
                    0.3, 0.9, 0.6, 0.8, 0.5, 0.7, 0.3, 0.4, 0.5, 0.0,
                    0.0, 0.0, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.7, 0.3,
                    0.2, 0.1, 0.1, 0.7, 0.3, 0.1, 0.2, 0.05, 0.05, 0.1, 0.1]

    def generate_samples(n_samples, class_dist):
        X_list = []
        y_list = []
        
        for c in range(5):
            count = max(1, int(np.round(class_dist[c] * n_samples)))
            
            # Generate samples around class centroid
            noise_scale = 0.08 if c < 2 else 0.12
            samples = np.tile(centroids[c], (count, 1))
            samples += np.random.randn(count, n_features) * noise_scale
            
            # Add class-specific variations
            if c == 1:  # DoS
                samples[:, 22:24] += np.random.rand(count, 2) * 0.2
            elif c == 2:  # Probe
                samples[:, 2] += np.random.rand(count) * 0.15
            elif c == 3:  # R2L
                samples[:, 10:12] += np.random.rand(count, 2) * 0.2
            elif c == 4:  # U2R
                samples[:, 12:15] += np.random.rand(count, 3) * 0.3
            
            samples = np.clip(samples, 0, 1)
            X_list.append(samples)
            y_list.append(np.full(count, c, dtype=np.int32))
        
        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list)
        
        # Shuffle
        perm = np.random.permutation(len(y))
        return X[perm], y[perm]

    X_train, y_train = generate_samples(n_train, train_dist)
    X_test, y_test = generate_samples(n_test, test_dist)
    
    return X_train, y_train, X_test, y_test


def create_balanced_dataset(X: np.ndarray, y: np.ndarray,
                           minority_target: int = 6734,
                           majority_cap: int = 67343,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create balanced dataset matching NeuroIDS-Sat oversampling strategy."""
    np.random.seed(seed)
    
    X_balanced = []
    y_balanced = []
    
    majority_count = np.max([np.sum(y == c) for c in range(5)])
    target_minority = max(minority_target, int(0.1 * majority_count))
    
    for c in range(5):
        mask = (y == c)
        X_c = X[mask]
        n_c = len(X_c)
        
        if n_c == 0:
            continue
        
        if n_c < target_minority:
            indices = np.random.choice(n_c, target_minority, replace=True)
            X_sampled = X_c[indices].copy()
            noise = np.random.normal(0, 0.01, X_sampled.shape)
            X_sampled = np.clip(X_sampled + noise, 0, 1)
        elif n_c > majority_cap:
            indices = np.random.choice(n_c, majority_cap, replace=False)
            X_sampled = X_c[indices]
        else:
            X_sampled = X_c
        
        X_balanced.append(X_sampled)
        y_balanced.append(np.full(len(X_sampled), c))
    
    X_balanced = np.vstack(X_balanced).astype(np.float32)
    y_balanced = np.concatenate(y_balanced).astype(np.int32)
    
    perm = np.random.permutation(len(y_balanced))
    return X_balanced[perm], y_balanced[perm]


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse frequency class weights."""
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    
    weights = np.ones(5)
    for c, count in zip(classes, counts):
        weights[c] = n_samples / (n_classes * count)
    
    return weights


def get_class_distribution(y: np.ndarray) -> Dict[str, int]:
    """Get class distribution as dictionary."""
    classes, counts = np.unique(y, return_counts=True)
    return {CLASS_NAMES[c]: int(count) for c, count in zip(classes, counts)}


def train_val_split(X: np.ndarray, y: np.ndarray,
                   val_ratio: float = 0.2,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """Stratified train/validation split."""
    np.random.seed(seed)
    
    train_idx = []
    val_idx = []
    
    for c in range(5):
        c_idx = np.where(y == c)[0]
        np.random.shuffle(c_idx)
        
        n_val = int(len(c_idx) * val_ratio)
        val_idx.extend(c_idx[:n_val])
        train_idx.extend(c_idx[n_val:])
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]