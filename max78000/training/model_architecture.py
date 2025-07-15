"""
EMG Classification Model Architecture for MAX78000
Compatible with ai8x-synthesis framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class EMGClassificationModel(nn.Module):
    """
    EMG Classification Model optimized for MAX78000 hardware
    
    Features:
    - 8-bit quantization compatible
    - Minimal memory footprint
    - Efficient convolution operations
    - 8-class gesture recognition
    """
    
    def __init__(self, 
                 input_size: int = 72,  # 8 channels × 9 features
                 num_classes: int = 8,
                 dropout_rate: float = 0.3):
        super(EMGClassificationModel, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Reshape input features to 2D for convolution
        # 8 channels × 9 features → 8×9 "image"
        self.feature_reshape = nn.Unflatten(1, (8, 9))
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 2), padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(16, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 72)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape to 2D feature map
        x = x.view(-1, self.input_size)
        x = self.feature_reshape(x)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class EMGLightweightModel(nn.Module):
    """
    Lightweight EMG model for resource-constrained deployment
    Optimized for MAX78000 constraints
    """
    
    def __init__(self, 
                 input_size: int = 72,
                 num_classes: int = 8,
                 hidden_dim: int = 64):
        super(EMGLightweightModel, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Simple fully connected architecture
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 72)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = x.view(-1, self.input_size)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EMGQuantizedModel(nn.Module):
    """
    Quantization-aware training model for MAX78000
    8-bit integer operations
    """
    
    def __init__(self, 
                 input_size: int = 72,
                 num_classes: int = 8):
        super(EMGQuantizedModel, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Model layers
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(32, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization"""
        x = self.quant(x)
        
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse layers for quantization"""
        # No fusion needed for this simple model
        pass


def create_model(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create EMG classification models
    
    Args:
        model_type: Type of model ('standard', 'lightweight', 'quantized')
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    if model_type == "standard":
        return EMGClassificationModel(**kwargs)
    elif model_type == "lightweight":
        return EMGLightweightModel(**kwargs)
    elif model_type == "quantized":
        return EMGQuantizedModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 72)):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Model size estimation
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"Model size: {size_mb:.2f} MB")


# Class definitions for reference
CLASS_NAMES = [
    "Rest",
    "Grasp", 
    "Release",
    "Rotate CW",
    "Rotate CCW",
    "Flex",
    "Extend",
    "Point"
]

# Feature configuration
FEATURE_CONFIG = {
    'channels': 8,
    'time_features': 5,  # MAV, WL, ZC, SSC, RMS
    'freq_features': 4,  # Frequency domain features
    'total_features': 72  # 8 channels × 9 features
}


if __name__ == "__main__":
    # Test model creation
    print("Testing EMG Classification Models")
    print("=" * 50)
    
    # Standard model
    model_std = create_model("standard")
    model_summary(model_std)
    print()
    
    # Lightweight model
    model_light = create_model("lightweight")
    model_summary(model_light)
    print()
    
    # Test forward pass
    test_input = torch.randn(4, 72)  # Batch of 4 samples
    
    with torch.no_grad():
        output_std = model_std(test_input)
        output_light = model_light(test_input)
        
        print(f"Standard model output shape: {output_std.shape}")
        print(f"Lightweight model output shape: {output_light.shape}")
        
        # Apply softmax to get probabilities
        probs_std = F.softmax(output_std, dim=1)
        probs_light = F.softmax(output_light, dim=1)
        
        print(f"Standard model predictions: {torch.argmax(probs_std, dim=1)}")
        print(f"Lightweight model predictions: {torch.argmax(probs_light, dim=1)}")