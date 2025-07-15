"""
EMG Dataset Loader and Preprocessing
Handles EMG signal data for training and validation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
import pickle
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings


class EMGDataset(Dataset):
    """
    Dataset class for EMG signal classification
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 labels: np.ndarray,
                 transform=None,
                 feature_scaler=None):
        """
        Initialize EMG dataset
        
        Args:
            features: Feature array of shape (n_samples, 72)
            labels: Label array of shape (n_samples,)
            transform: Optional transform to apply to features
            feature_scaler: Optional scaler for feature normalization
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform
        self.feature_scaler = feature_scaler
        
        # Apply feature scaling if provided
        if self.feature_scaler is not None:
            self.features = self.feature_scaler.transform(self.features)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get dataset item
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Apply transform if provided
        if self.transform:
            feature = self.transform(feature)
        
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class EMGDataProcessor:
    """
    Data processor for EMG signals
    """
    
    def __init__(self, 
                 normalize_features: bool = True,
                 augment_data: bool = True,
                 noise_std: float = 0.01):
        """
        Initialize data processor
        
        Args:
            normalize_features: Whether to normalize features
            augment_data: Whether to apply data augmentation
            noise_std: Standard deviation for noise augmentation
        """
        self.normalize_features = normalize_features
        self.augment_data = augment_data
        self.noise_std = noise_std
        
        self.feature_scaler = StandardScaler() if normalize_features else None
        self.label_encoder = LabelEncoder()
    
    def generate_synthetic_data(self, 
                               n_samples: int = 1000,
                               n_features: int = 72,
                               n_classes: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic EMG data for testing
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features per sample
            n_classes: Number of classes
            
        Returns:
            Tuple of (features, labels)
        """
        np.random.seed(42)
        
        # Generate features with class-specific patterns
        features = []
        labels = []
        
        for class_id in range(n_classes):
            samples_per_class = n_samples // n_classes
            
            # Create class-specific base pattern
            base_pattern = np.random.randn(n_features) * 0.5
            base_pattern[class_id * 9:(class_id + 1) * 9] += 2.0  # Stronger signal for specific channels
            
            for _ in range(samples_per_class):
                # Add noise and variation
                sample = base_pattern + np.random.randn(n_features) * 0.3
                
                # Add some channel-specific patterns
                for channel in range(8):
                    channel_start = channel * 9
                    channel_end = channel_start + 9
                    
                    # Time domain features (first 5)
                    sample[channel_start:channel_start+5] *= (1.0 + 0.2 * np.sin(class_id))
                    
                    # Frequency domain features (last 4)
                    sample[channel_start+5:channel_end] *= (1.0 + 0.3 * np.cos(class_id))
                
                features.append(sample)
                labels.append(class_id)
        
        # Shuffle data
        indices = np.random.permutation(len(features))
        features = np.array(features)[indices]
        labels = np.array(labels)[indices]
        
        return features, labels
    
    def augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to features
        
        Args:
            features: Input features
            
        Returns:
            Augmented features
        """
        if not self.augment_data:
            return features
        
        augmented = features.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, features.shape)
        augmented += noise
        
        # Scale variation (±10%)
        scale_factor = np.random.uniform(0.9, 1.1, (features.shape[0], 1))
        augmented *= scale_factor
        
        # Channel-wise offset
        offset = np.random.normal(0, 0.05, (features.shape[0], 8))
        offset_expanded = np.repeat(offset, 9, axis=1)
        augmented += offset_expanded
        
        return augmented
    
    def prepare_dataset(self, 
                       features: np.ndarray,
                       labels: np.ndarray,
                       test_size: float = 0.2,
                       val_size: float = 0.1) -> Dict[str, EMGDataset]:
        """
        Prepare train/validation/test datasets
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Dictionary containing train/val/test datasets
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels_encoded, test_size=test_size, stratify=labels_encoded, random_state=42
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )
        
        # Fit scaler on training data
        if self.feature_scaler is not None:
            self.feature_scaler.fit(X_train)
        
        # Apply data augmentation to training data
        X_train_aug = self.augment_features(X_train)
        
        # Create datasets
        datasets = {
            'train': EMGDataset(X_train_aug, y_train, feature_scaler=self.feature_scaler),
            'val': EMGDataset(X_val, y_val, feature_scaler=self.feature_scaler),
            'test': EMGDataset(X_test, y_test, feature_scaler=self.feature_scaler)
        }
        
        return datasets
    
    def create_dataloaders(self, 
                          datasets: Dict[str, EMGDataset],
                          batch_size: int = 32,
                          num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders
        
        Args:
            datasets: Dictionary of datasets
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {}
        
        for split, dataset in datasets.items():
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == 'train')
            )
        
        return dataloaders
    
    def save_preprocessor(self, filepath: str):
        """
        Save preprocessor state
        
        Args:
            filepath: Path to save preprocessor
        """
        state = {
            'feature_scaler': self.feature_scaler,
            'label_encoder': self.label_encoder,
            'normalize_features': self.normalize_features,
            'augment_data': self.augment_data,
            'noise_std': self.noise_std
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_preprocessor(self, filepath: str):
        """
        Load preprocessor state
        
        Args:
            filepath: Path to load preprocessor from
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.feature_scaler = state['feature_scaler']
        self.label_encoder = state['label_encoder']
        self.normalize_features = state['normalize_features']
        self.augment_data = state['augment_data']
        self.noise_std = state['noise_std']


def load_emg_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EMG data from file
    
    Args:
        data_path: Path to data file
        
    Returns:
        Tuple of (features, labels)
    """
    if not os.path.exists(data_path):
        warnings.warn(f"Data file not found: {data_path}")
        return None, None
    
    # Support multiple file formats
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        features = data['features']
        labels = data['labels']
    elif data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            features = data['features']
            labels = data['labels']
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return features, labels


def save_emg_data(features: np.ndarray, 
                  labels: np.ndarray, 
                  filepath: str):
    """
    Save EMG data to file
    
    Args:
        features: Feature array
        labels: Label array  
        filepath: Output file path
    """
    np.savez_compressed(filepath, features=features, labels=labels)


def print_dataset_info(datasets: Dict[str, EMGDataset]):
    """
    Print dataset information
    
    Args:
        datasets: Dictionary of datasets
    """
    print("Dataset Information:")
    print("=" * 50)
    
    for split, dataset in datasets.items():
        print(f"{split.upper()} SET:")
        print(f"  Samples: {len(dataset)}")
        print(f"  Features: {dataset.features.shape[1]}")
        print(f"  Classes: {len(np.unique(dataset.labels))}")
        print(f"  Feature range: [{dataset.features.min():.3f}, {dataset.features.max():.3f}]")
        print(f"  Label distribution: {np.bincount(dataset.labels)}")
        print()


if __name__ == "__main__":
    # Test dataset creation
    print("Testing EMG Dataset Creation")
    print("=" * 50)
    
    # Create processor
    processor = EMGDataProcessor(
        normalize_features=True,
        augment_data=True,
        noise_std=0.01
    )
    
    # Generate synthetic data
    features, labels = processor.generate_synthetic_data(
        n_samples=2000,
        n_features=72,
        n_classes=8
    )
    
    print(f"Generated data shape: {features.shape}")
    print(f"Generated labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Prepare datasets
    datasets = processor.prepare_dataset(features, labels)
    print_dataset_info(datasets)
    
    # Create dataloaders
    dataloaders = processor.create_dataloaders(datasets, batch_size=16)
    
    # Test dataloader
    train_loader = dataloaders['train']
    for batch_features, batch_labels in train_loader:
        print(f"Batch features shape: {batch_features.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Feature range: [{batch_features.min():.3f}, {batch_features.max():.3f}]")
        print(f"Labels in batch: {batch_labels.unique()}")
        break