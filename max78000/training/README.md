# EMG Classification Training for MAX78000

This directory contains the training infrastructure for EMG gesture classification models optimized for deployment on the MAX78000 microcontroller.

## Overview

The training pipeline includes:
- **PyTorch model architectures** optimized for MAX78000 constraints
- **Dataset processing** with augmentation and normalization
- **Training script** with validation and early stopping
- **Model conversion** to MAX78000 format using ai8x-synthesis
- **Quantization** for 8-bit inference

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# Train standard model with synthetic data
python train.py --model standard --epochs 50 --batch_size 32

# Train lightweight model
python train.py --model lightweight --epochs 100 --batch_size 64

# Train with custom data
python train.py --model standard --data_path /path/to/your/data.npz --epochs 100
```

### 3. Convert Model for MAX78000

```bash
# Convert trained model to MAX78000 format
python convert_to_max78000.py \
    --model checkpoints/best_model.pth \
    --output-dir generated \
    --formats onnx weights inference
```

## File Structure

```
training/
├── model_architecture.py    # PyTorch model definitions
├── dataset.py              # Dataset loading and preprocessing
├── train.py                # Training script with validation
├── convert_to_max78000.py  # Model conversion for MAX78000
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── generated/             # Generated files for MAX78000
    ├── emg_weights.h      # Weight header file
    ├── emg_inference.c    # Inference code
    ├── emg_model.onnx     # ONNX model
    └── conversion_info.json # Conversion metadata
```

## Model Architectures

### Standard Model (EMGClassificationModel)
- **Input**: 72 features (8 channels × 9 features)
- **Architecture**: CNN with 2D convolutions + fully connected layers
- **Parameters**: ~50K parameters
- **Best for**: Accuracy-focused applications

### Lightweight Model (EMGLightweightModel)
- **Input**: 72 features
- **Architecture**: Fully connected layers only
- **Parameters**: ~5K parameters
- **Best for**: Resource-constrained deployment

### Quantized Model (EMGQuantizedModel)
- **Input**: 72 features
- **Architecture**: Quantization-aware training
- **Parameters**: ~10K parameters (8-bit)
- **Best for**: MAX78000 deployment

## Training Configuration

### Command Line Arguments

```bash
python train.py --help
```

**Key parameters:**
- `--model`: Model type (standard, lightweight, quantized)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--patience`: Early stopping patience (default: 10)
- `--data_path`: Path to training data file

### Data Format

Training data should be in `.npz` format with:
- `features`: Shape (n_samples, 72) - EMG features
- `labels`: Shape (n_samples,) - Class labels (0-7)

### Feature Format

The 72 input features are organized as:
```
Channel 0: [MAV, WL, ZC, SSC, RMS, F1, F2, F3, F4]  # 9 features
Channel 1: [MAV, WL, ZC, SSC, RMS, F1, F2, F3, F4]  # 9 features
...
Channel 7: [MAV, WL, ZC, SSC, RMS, F1, F2, F3, F4]  # 9 features
```

Where:
- **MAV**: Mean Absolute Value
- **WL**: Waveform Length
- **ZC**: Zero Crossings
- **SSC**: Slope Sign Changes
- **RMS**: Root Mean Square
- **F1-F4**: Frequency domain features

## Class Labels

The model classifies 8 gesture classes:
0. Rest
1. Grasp
2. Release
3. Rotate CW
4. Rotate CCW
5. Flex
6. Extend
7. Point

## Training Output

Training produces:
- **Checkpoints**: Model weights saved during training
- **Logs**: TensorBoard logs for monitoring
- **Plots**: Training curves and confusion matrices
- **Preprocessor**: Feature scaling parameters

### Directory Structure After Training

```
logs/emg_training_standard_20250715_120000/
├── events.out.tfevents.xxx    # TensorBoard logs
├── confusion_matrix.png       # Confusion matrix plot
├── training_history.png       # Training curves
└── training_history.json      # Training metrics

checkpoints/emg_training_standard_20250715_120000/
├── best_model.pth            # Best model weights
├── checkpoint_epoch_X.pth    # Periodic checkpoints
└── preprocessor.pkl          # Data preprocessing state
```

## Model Conversion

### Prerequisites

1. Install ai8x-synthesis from Maxim's repository:
```bash
git clone https://github.com/MaximIntegratedAI/ai8x-synthesis.git
cd ai8x-synthesis
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export AI8X_SYNTHESIS_PATH=/path/to/ai8x-synthesis
```

### Conversion Process

```bash
# Convert model to multiple formats
python convert_to_max78000.py \
    --model checkpoints/best_model.pth \
    --preprocessor checkpoints/preprocessor.pkl \
    --output-dir generated \
    --formats onnx weights inference makefile
```

### Generated Files

- **emg_weights.h**: C header with quantized weights
- **emg_inference.c**: C inference functions
- **emg_model.onnx**: ONNX model for validation
- **ai8x_rules.mk**: Makefile rules for synthesis
- **conversion_info.json**: Conversion metadata

## Integration with MAX78000

### 1. Copy Generated Files

```bash
cp generated/emg_weights.h ../src/
cp generated/emg_inference.c ../src/
```

### 2. Update MAX78000 Project

Add to your MAX78000 project:
```c
#include "emg_weights.h"
#include "emg_inference.h"
```

### 3. Initialize CNN

```c
// Initialize CNN accelerator
emg_cnn_init();

// Run inference
int8_t features[72] = { /* feature data */ };
int32_t output[8];
emg_cnn_infer(features, output);

// Get predicted class
int predicted_class = emg_get_predicted_class(output);
const char* class_name = emg_get_class_name(predicted_class);
```

## Performance Optimization

### Model Size Reduction
- Use lightweight model architecture
- Apply quantization-aware training
- Prune unnecessary weights

### Memory Optimization
- Batch normalization folding
- Weight sharing
- Activation quantization

### Inference Speed
- Layer fusion
- Hardware acceleration
- Parallel processing

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow training**: Enable GPU acceleration
3. **Poor accuracy**: Increase model capacity or training epochs
4. **Conversion errors**: Check ai8x-synthesis installation

### Debug Commands

```bash
# Check model architecture
python -c "from model_architecture import create_model; print(create_model('standard'))"

# Test dataset loading
python -c "from dataset import EMGDataProcessor; p = EMGDataProcessor(); print('Dataset OK')"

# Validate conversion
python convert_to_max78000.py --model checkpoints/best_model.pth --formats onnx
```

## Performance Metrics

### Expected Performance
- **Validation Accuracy**: 85-95%
- **Inference Time**: <1ms on MAX78000
- **Model Size**: <100KB for weights
- **Power Consumption**: <1mW during inference

### Benchmarking

```bash
# Run performance benchmark
python -c "
from model_architecture import create_model
import torch
model = create_model('standard')
input_tensor = torch.randn(1, 72)
with torch.no_grad():
    output = model(input_tensor)
print(f'Output shape: {output.shape}')
"
```

## Contributing

When adding new features:
1. Update model architectures in `model_architecture.py`
2. Add corresponding conversion logic in `convert_to_max78000.py`
3. Test with sample data
4. Update documentation

## References

- [MAX78000 User Guide](https://www.maximintegrated.com/en/products/microcontrollers/MAX78000.html)
- [ai8x-synthesis Documentation](https://github.com/MaximIntegratedAI/ai8x-synthesis)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [EMG Signal Processing](https://doi.org/10.1109/TBME.2016.2628362)