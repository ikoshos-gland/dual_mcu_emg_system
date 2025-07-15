# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dual-microcontroller EMG signal processing system that uses STM32 NUCLEO-H7S3L8 for real-time signal processing and MAX78000 for AI inference. The system processes 8-channel EMG signals at 16kSPS through a complete pipeline: ADS1299 ADC → STM32 processing → MAX78000 inference → control output.

## Build System

### STM32 H7S3L8 Firmware
```bash
# Navigate to STM32 project directory
cd stm32_h7s3l8/

# Build all targets (ELF, HEX, BIN)
make all

# Clean build artifacts
make clean

# Flash to target hardware
make flash

# Start debug session with OpenOCD + GDB
make debug

# Show project configuration
make info
```

### MAX78000 AI Inference
```bash
# Navigate to MAX78000 project directory
cd max78000/

# Build MAX78000 firmware
make all

# Clean build artifacts
make clean

# Flash to MAX78000 hardware
make flash

# Start debug session
make debug

# Show build configuration
make info

# Generate CNN model (requires ai8x-synthesis)
make generate-cnn-model
```

### Testing
```bash
# Run unit tests from STM32 directory
make test

# Or run tests directly from tests directory
cd tests/ && make run
```

### Documentation
```bash
# Generate Doxygen documentation
make documentation
```

## Architecture Overview

### Signal Processing Pipeline
The system implements a complete real-time EMG processing pipeline:

1. **ADS1299 Driver** (`ads1299_driver.c/h`): Manages 8-channel 24-bit ADC via SPI with DMA
2. **EMG Processing** (`emg_processing.c/h`): Real-time filtering, windowing, and feature extraction
3. **Communication Protocol** (`shared/protocols/mcu_communication.h`): Inter-MCU packet protocol
4. **STM32 Main Controller** (`stm32_h7s3l8/src/main.c`): System orchestration and callback management
5. **MAX78000 CNN Engine** (`max78000/src/emg_cnn.c/h`): Hardware-accelerated AI inference
6. **MAX78000 Communication** (`max78000/src/emg_communication.c/h`): SPI slave interface
7. **MAX78000 Main Application** (`max78000/src/emg_main.c`): AI inference pipeline

### Key Components

#### Data Flow Architecture
- **Input**: ADS1299 DRDY interrupt triggers data acquisition
- **Processing**: Windowed signal processing with 50% overlap (512 samples)
- **Output**: 100Hz feature packets to MAX78000 via SPI
- **Callbacks**: Event-driven architecture connects all components

#### Memory Management
- **Circular Buffers**: Used for continuous data acquisition without blocking
- **DMA Transfers**: All high-speed SPI operations use DMA for efficiency
- **ARM DSP**: Optimized filtering using ARM CMSIS-DSP library

#### Communication Protocol
- **Packet Structure**: Header (8B) + Payload (≤128B) + CRC32 (4B)
- **Feature Data**: 72 bytes per packet (8 channels × 9 features)
- **Error Handling**: Sequence numbers, acknowledgments, and retransmission

### Configuration System

#### Hardware Configuration (`main.h`)
- **Clock System**: 600MHz Cortex-M7 with 300MHz AHB
- **SPI1**: ADS1299 communication at 20MHz
- **SPI2**: MAX78000 communication at 10MHz
- **DMA**: HPDMA1 channels for all SPI transfers

#### Signal Processing Parameters
- **Sampling Rate**: 16kSPS per channel
- **Window Size**: 512 samples (32ms at 16kSPS)
- **Filters**: Bandpass (20-450Hz), notch (50Hz), comb filter
- **Features**: Time domain (MAV, WL, ZC, SSC, RMS) + frequency domain

## Development Workflow

### Working with STM32 Code
- **HAL Configuration**: `stm32_hal_config.c` contains all peripheral initialization
- **Driver Development**: Follow existing patterns in `ads1299_driver.c`
- **Processing Extensions**: Add new algorithms to `emg_processing.c`
- **Error Handling**: Use `EMG_StatusTypeDef` enum for consistent error reporting

### Working with MAX78000 Code
- **CNN Implementation**: `emg_cnn.c` provides hardware-accelerated inference
- **Model Integration**: Replace `emg_weights.h` with ai8x-synthesis generated weights
- **Communication**: `emg_communication.c` handles SPI slave protocol
- **Application Logic**: `emg_main.c` orchestrates the inference pipeline

### State Management
The system uses a state machine approach:
- **SYSTEM_INIT**: Hardware initialization phase
- **SYSTEM_READY**: Idle state, ready for commands
- **SYSTEM_RUNNING**: Active processing with continuous data flow
- **SYSTEM_ERROR**: Error state with recovery mechanisms

### Callback Architecture
Components communicate through callbacks:
- **ADS1299 Data Ready**: Triggers EMG processing
- **Feature Ready**: Sends data to MAX78000
- **Inference Complete**: MAX78000 CNN completion interrupt
- **Communication Events**: Handle inter-MCU protocol

### MAX78000 AI Inference System
- **Location**: `max78000/` directory (fully implemented)
- **CNN Accelerator**: Hardware-accelerated inference using 64-processor CNN engine
- **Model Format**: 8-bit quantized neural networks with 442KB weight memory
- **Inference Speed**: <1ms latency with ultra-low power consumption
- **Communication**: SPI slave interface with robust packet protocol

## Performance Characteristics

### Real-time Requirements
- **Sampling**: 16kSPS × 8 channels = 128k samples/second
- **Processing Latency**: <10ms for feature extraction (STM32)
- **Inference Latency**: <1ms for AI classification (MAX78000)
- **Communication**: 100Hz feature updates between MCUs
- **Total Latency**: <20ms end-to-end (achieved)

### Resource Usage
- **Memory**: Optimized for 620KB RAM (STM32H7S3L8)
- **CPU**: ARM DSP optimizations for filtering operations
- **DMA**: All high-bandwidth transfers use DMA to reduce CPU load

### Neural Network Training and Deployment

#### Training Environment Setup
```bash
# Install Python dependencies
cd max78000/training/
pip install -r requirements.txt

# Train a model
python train.py --model standard --epochs 100 --batch_size 32

# Convert for MAX78000 deployment
python convert_to_max78000.py --model checkpoints/best_model.pth --output-dir generated
```

#### Model Development Workflow
1. **Data Collection**: Gather EMG data with proper feature extraction
2. **Model Training**: Use PyTorch training scripts in `max78000/training/`
3. **Conversion**: Convert to MAX78000 format using ai8x-synthesis
4. **Integration**: Copy generated files to MAX78000 project
5. **Testing**: Validate inference accuracy and performance

#### Training Output Structure
```
max78000/training/
├── checkpoints/           # Trained model weights
├── logs/                 # TensorBoard training logs
├── generated/            # MAX78000 converted files
│   ├── emg_weights.h     # Quantized weights
│   ├── emg_inference.c   # Inference functions
│   └── emg_model.onnx    # ONNX model
```

## Troubleshooting

### Common Build Issues
- **Missing ARM GCC**: Install arm-none-eabi-gcc toolchain
- **HAL Library**: STM32H7RSxx HAL drivers must be in Drivers/ directory
- **Linker Script**: Ensure STM32H7S3L8Hx_FLASH.ld is present

### Hardware Debugging
- **ST-Link**: Use `make flash` for programming
- **OpenOCD**: Debug session with `make debug`
- **Serial Output**: UART debugging available on designated pins

### Signal Processing Issues
- **Filter Stability**: Check coefficient tables in `emg_processing.c`
- **Buffer Overflows**: Monitor circular buffer management
- **Timing**: Verify interrupt priorities and DMA configuration

### Neural Network Issues
- **Model Conversion**: Ensure ai8x-synthesis is properly installed
- **Weight Loading**: Check generated `emg_weights.h` file integrity
- **Inference Errors**: Verify input feature format and scaling
- **Performance**: Monitor inference time and memory usage