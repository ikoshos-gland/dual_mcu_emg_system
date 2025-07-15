# Dual-Microcontroller 8-Channel EMG System

## System Architecture
- **STM32 NUCLEO-H7S3L8**: Signal processing, filtering, and feature extraction
- **MAX78000**: AI inference and classification
- **ADS1299**: 8-channel 24-bit EMG ADC

## Project Structure
```
dual_mcu_emg_system/
├── stm32_h7s3l8/          # STM32 H7S3L8 firmware
│   ├── src/               # Source files
│   ├── inc/               # Header files
│   ├── drivers/           # Hardware drivers
│   └── examples/          # Example implementations
├── max78000/              # MAX78000 AI inference
│   ├── src/               # Inference source code
│   ├── models/            # Neural network models
│   ├── training/          # Training scripts
│   └── deployment/        # Deployment utilities
├── shared/                # Shared components
│   ├── protocols/         # Communication protocols
│   ├── utils/             # Utility functions
│   └── config/            # Configuration files
├── docs/                  # Documentation
└── tests/                 # Test suites
```

## System Flow
1. **ADS1299** acquires 8-channel EMG data at 16kSPS
2. **STM32 H7S3L8** processes signals with filtering and feature extraction
3. **MAX78000** performs AI inference on extracted features
4. **Control output** generates actuator commands based on classifications

## Performance Targets
- **Sampling Rate**: 16kSPS × 8 channels
- **Processing Latency**: <10ms (STM32)
- **Inference Latency**: <1ms (MAX78000)
- **Total System Latency**: <20ms
- **Classification Accuracy**: >90%