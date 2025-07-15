# EMG System Testing and Integration Scripts

This directory contains comprehensive testing and integration scripts for the dual-MCU EMG system.

## Scripts Overview

### 1. `system_integration.py`
**End-to-End System Integration Tester**

Tests the complete dual-MCU system with real-time data flow from STM32 to MAX78000.

**Features:**
- Real-time data monitoring from both microcontrollers
- Performance metrics collection
- Communication protocol validation
- Live status display with classification results
- Automated test report generation

**Usage:**
```bash
# Basic integration test
python system_integration.py --duration 300

# Custom serial ports
python system_integration.py --stm32-port COM3 --max78000-port COM4

# With real-time plotting
python system_integration.py --duration 600 --plot
```

### 2. `performance_monitor.py`
**Real-Time System Performance Monitor**

Monitors system performance metrics including CPU, memory, and EMG-specific metrics.

**Features:**
- Real-time performance monitoring
- Live console display
- Performance threshold alerts
- Historical data visualization
- Automated performance reports

**Usage:**
```bash
# Start monitoring with live display
python performance_monitor.py --live-display

# Monitor for specific duration
python performance_monitor.py --duration 3600 --plot

# Custom monitoring interval
python performance_monitor.py --interval 0.5 --live-display
```

### 3. `system_test.py`
**Comprehensive System Testing Framework**

Complete automated testing suite covering all system components.

**Features:**
- Hardware connection tests
- Communication protocol validation
- Signal processing verification
- ML inference accuracy testing
- Performance benchmarking
- Integration testing
- Stress testing
- HTML and JSON report generation

**Usage:**
```bash
# Run all tests
python system_test.py

# Run specific test category
python system_test.py --category hardware_tests

# Custom configuration
python system_test.py --config my_test_config.json
```

## Test Configuration

### Default Test Configuration (`test_config.json`)

```json
{
  "hardware": {
    "stm32_port": "COM3",
    "max78000_port": "COM4",
    "baudrate": 115200
  },
  "signal_processing": {
    "sampling_rate": 16000,
    "channels": 8,
    "feature_count": 72
  },
  "ml_inference": {
    "model_classes": 8,
    "confidence_threshold": 0.7,
    "max_inference_time": 5000
  },
  "performance": {
    "max_cpu_usage": 80.0,
    "max_memory_usage": 80.0
  }
}
```

## Testing Workflow

### 1. Hardware Verification
```bash
# Test hardware connections
python system_test.py --category hardware_tests
```

### 2. Communication Testing
```bash
# Test inter-MCU communication
python system_test.py --category communication_tests
```

### 3. Signal Processing Validation
```bash
# Test filtering and feature extraction
python system_test.py --category signal_processing_tests
```

### 4. ML Inference Testing
```bash
# Test model loading and inference
python system_test.py --category ml_inference_tests
```

### 5. Performance Testing
```bash
# Test system performance
python system_test.py --category performance_tests
```

### 6. Integration Testing
```bash
# Test end-to-end system
python system_integration.py --duration 300
```

### 7. Stress Testing
```bash
# Test under high load
python system_test.py --category stress_tests
```

## Test Categories

### Hardware Tests
- **Hardware Connections**: Verify STM32 and MAX78000 connections
- **ADS1299 Communication**: Test ADC communication interface

### Communication Tests
- **Inter-MCU Communication**: Validate SPI communication protocol
- **Feature Data Transmission**: Test feature packet transmission

### Signal Processing Tests
- **Signal Filtering**: Verify bandpass and notch filtering
- **Feature Extraction**: Test time and frequency domain features

### ML Inference Tests
- **Model Loading**: Verify neural network model loading
- **Inference Accuracy**: Test classification accuracy
- **Inference Timing**: Measure inference performance

### Performance Tests
- **System Performance**: Monitor CPU and memory usage
- **Real-time Performance**: Verify real-time constraints

### Integration Tests
- **End-to-End Latency**: Measure total system latency
- **Continuous Operation**: Test long-term reliability

### Stress Tests
- **High Load Performance**: Test under maximum load
- **Error Recovery**: Test error handling mechanisms

## Test Results

### Output Files
- `test_report_YYYYMMDD_HHMMSS.json` - JSON test results
- `test_report_YYYYMMDD_HHMMSS.html` - HTML test report
- `integration_report_YYYYMMDD_HHMMSS.json` - Integration test results
- `performance_report_YYYYMMDD_HHMMSS.json` - Performance metrics
- `system_plots_YYYYMMDD_HHMMSS.png` - Performance visualizations

### Test Report Structure
```json
{
  "timestamp": "2025-01-15T12:00:00",
  "test_suite_version": "1.0.0",
  "summary": {
    "total_tests": 15,
    "passed_tests": 14,
    "failed_tests": 1,
    "execution_time": 125.6
  },
  "test_results": {
    "hardware_tests": [...],
    "communication_tests": [...],
    "signal_processing_tests": [...],
    "ml_inference_tests": [...],
    "performance_tests": [...],
    "integration_tests": [...],
    "stress_tests": [...]
  }
}
```

## Performance Metrics

### System Metrics
- **CPU Usage**: Host system CPU utilization
- **Memory Usage**: Host system memory utilization
- **Network Throughput**: Data transfer rates
- **Disk I/O**: File system access

### EMG System Metrics
- **Feature Rate**: Features extracted per second (~100 Hz)
- **Classification Rate**: Classifications per second (~1-10 Hz)
- **Inference Time**: Neural network inference time (<1 ms)
- **Communication Errors**: Inter-MCU communication failures
- **Data Throughput**: Total data processed per second
- **MCU CPU Usage**: Estimated microcontroller CPU usage

### Performance Thresholds
- **CPU Warning**: 80% usage
- **CPU Critical**: 95% usage
- **Memory Warning**: 80% usage
- **Memory Critical**: 95% usage
- **Inference Time Warning**: 5 ms
- **Inference Time Critical**: 10 ms
- **Feature Rate Minimum**: 90 Hz
- **Classification Rate Minimum**: 1 Hz

## Troubleshooting

### Common Issues

1. **Serial Port Access**
   - Ensure correct COM port configuration
   - Check device manager for port assignments
   - Verify no other applications using ports

2. **Connection Timeouts**
   - Increase connection timeout in config
   - Check USB cable connections
   - Verify microcontroller firmware

3. **Test Failures**
   - Check hardware connections
   - Verify firmware versions
   - Review test configuration

### Debug Commands

```bash
# Check system status
python -c "from system_integration import SystemIntegrationTester; t = SystemIntegrationTester(); print(t.connect_devices())"

# Test performance monitoring
python -c "from performance_monitor import SystemPerformanceMonitor; m = SystemPerformanceMonitor(); m.start_monitoring(); import time; time.sleep(5); m.stop_monitoring()"

# Validate test configuration
python -c "from system_test import EMGSystemTestSuite; s = EMGSystemTestSuite(); print(s.config)"
```

## Continuous Integration

### Automated Testing
```bash
# Run automated test suite
python system_test.py --config ci_config.json > test_results.log 2>&1

# Check exit code
if [ $? -eq 0 ]; then
  echo "All tests passed"
else
  echo "Tests failed"
  exit 1
fi
```

### Performance Monitoring
```bash
# Long-term monitoring
python performance_monitor.py --duration 86400 --log-dir /var/log/emg_system/
```

## Dependencies

### Required Python Packages
```bash
pip install numpy matplotlib psutil pyserial
```

### Optional Packages
```bash
pip install scipy scikit-learn torch tensorboard
```

## Configuration

### Environment Variables
- `EMG_STM32_PORT`: Default STM32 serial port
- `EMG_MAX78000_PORT`: Default MAX78000 serial port
- `EMG_TEST_CONFIG`: Path to test configuration file
- `EMG_RESULTS_DIR`: Test results directory

### Hardware Requirements
- STM32 NUCLEO-H7S3L8 development board
- MAX78000 evaluation kit
- ADS1299 evaluation board
- USB connections for serial communication

### Software Requirements
- Python 3.7+
- Serial drivers for microcontrollers
- OpenOCD for debugging (optional)

## Best Practices

### Test Development
1. **Write tests first**: Define test cases before implementation
2. **Use descriptive names**: Clear test and function names
3. **Test edge cases**: Include boundary conditions
4. **Mock external dependencies**: Use simulation for hardware tests
5. **Document test requirements**: Clear test specifications

### Performance Testing
1. **Baseline measurements**: Establish performance baselines
2. **Consistent environments**: Use same hardware/software setup
3. **Multiple runs**: Average results across multiple test runs
4. **Resource monitoring**: Track CPU, memory, and I/O usage
5. **Automated alerts**: Set up threshold-based alerts

### Integration Testing
1. **End-to-end scenarios**: Test complete user workflows
2. **Real-time constraints**: Verify timing requirements
3. **Error conditions**: Test error handling and recovery
4. **Long-term stability**: Extended operation testing
5. **Load testing**: Test under various load conditions

## Contributing

When adding new tests:
1. Follow existing test patterns
2. Add comprehensive documentation
3. Include error handling
4. Update test configuration schema
5. Add to appropriate test category
6. Test on target hardware

## References

- [EMG System Documentation](../README.md)
- [Hardware Setup Guide](../docs/hardware_setup.md)
- [Software Architecture](../docs/software_architecture.md)
- [Performance Specifications](../docs/performance_specs.md)