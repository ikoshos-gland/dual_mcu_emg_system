#!/usr/bin/env python3
"""
End-to-End System Integration Script
Tests and validates the complete dual-MCU EMG system
"""

import os
import sys
import time
import json
import serial
import struct
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging

# Add training directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'max78000', 'training'))

try:
    from dataset import EMGDataProcessor
    from model_architecture import CLASS_NAMES
except ImportError:
    print("Warning: Could not import training modules. Some features may be limited.")
    CLASS_NAMES = ["Rest", "Grasp", "Release", "Rotate CW", "Rotate CCW", "Flex", "Extend", "Point"]


class SystemIntegrationTester:
    """
    Complete system integration tester for dual-MCU EMG system
    """
    
    def __init__(self, 
                 stm32_port: str = "COM3",
                 max78000_port: str = "COM4",
                 baudrate: int = 115200,
                 log_dir: str = "integration_logs"):
        """
        Initialize system integration tester
        
        Args:
            stm32_port: STM32 serial port
            max78000_port: MAX78000 serial port  
            baudrate: Serial communication baudrate
            log_dir: Directory for logging
        """
        self.stm32_port = stm32_port
        self.max78000_port = max78000_port
        self.baudrate = baudrate
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Serial connections
        self.stm32_serial = None
        self.max78000_serial = None
        
        # System state
        self.is_running = False
        self.data_buffer = deque(maxlen=1000)
        self.classification_buffer = deque(maxlen=100)
        self.performance_metrics = {
            'feature_packets': 0,
            'classifications': 0,
            'communication_errors': 0,
            'inference_times': [],
            'system_uptime': 0
        }
        
        # Threading
        self.stm32_thread = None
        self.max78000_thread = None
        self.monitor_thread = None
        
        # Configuration
        self.config = {
            'sampling_rate': 16000,
            'window_size': 512,
            'feature_rate': 100,
            'classification_threshold': 0.7,
            'test_duration': 300  # 5 minutes
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"integration_test_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("System Integration Tester Initialized")
    
    def connect_devices(self) -> bool:
        """
        Connect to both microcontrollers
        
        Returns:
            True if both connections successful
        """
        try:
            # Connect to STM32
            self.logger.info(f"Connecting to STM32 on {self.stm32_port}")
            self.stm32_serial = serial.Serial(
                port=self.stm32_port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            # Connect to MAX78000
            self.logger.info(f"Connecting to MAX78000 on {self.max78000_port}")
            self.max78000_serial = serial.Serial(
                port=self.max78000_port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            # Wait for initialization
            time.sleep(2.0)
            
            # Test connections
            if self.test_stm32_connection() and self.test_max78000_connection():
                self.logger.info("Both devices connected successfully")
                return True
            else:
                self.logger.error("Device connection test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def test_stm32_connection(self) -> bool:
        """Test STM32 connection"""
        try:
            # Send status command
            self.stm32_serial.write(b"STATUS\\n")
            response = self.stm32_serial.readline().decode('utf-8').strip()
            
            if "STM32" in response:
                self.logger.info("STM32 connection verified")
                return True
            else:
                self.logger.warning(f"STM32 unexpected response: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"STM32 connection test failed: {e}")
            return False
    
    def test_max78000_connection(self) -> bool:
        """Test MAX78000 connection"""
        try:
            # Send status command
            self.max78000_serial.write(b"STATUS\\n")
            response = self.max78000_serial.readline().decode('utf-8').strip()
            
            if "MAX78000" in response:
                self.logger.info("MAX78000 connection verified")
                return True
            else:
                self.logger.warning(f"MAX78000 unexpected response: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"MAX78000 connection test failed: {e}")
            return False
    
    def start_data_acquisition(self):
        """Start EMG data acquisition on STM32"""
        try:
            self.logger.info("Starting EMG data acquisition")
            self.stm32_serial.write(b"START_EMG\\n")
            
            # Wait for confirmation
            response = self.stm32_serial.readline().decode('utf-8').strip()
            if "STARTED" in response:
                self.logger.info("EMG acquisition started successfully")
                return True
            else:
                self.logger.error(f"Failed to start EMG acquisition: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Start acquisition failed: {e}")
            return False
    
    def stop_data_acquisition(self):
        """Stop EMG data acquisition"""
        try:
            self.logger.info("Stopping EMG data acquisition")
            self.stm32_serial.write(b"STOP_EMG\\n")
            
            # Wait for confirmation
            response = self.stm32_serial.readline().decode('utf-8').strip()
            if "STOPPED" in response:
                self.logger.info("EMG acquisition stopped successfully")
                return True
            else:
                self.logger.error(f"Failed to stop EMG acquisition: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Stop acquisition failed: {e}")
            return False
    
    def stm32_data_thread(self):
        """Thread for reading STM32 data"""
        self.logger.info("STM32 data thread started")
        
        while self.is_running:
            try:
                if self.stm32_serial and self.stm32_serial.in_waiting > 0:
                    line = self.stm32_serial.readline().decode('utf-8').strip()
                    self.parse_stm32_data(line)
                    
                time.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                self.logger.error(f"STM32 data thread error: {e}")
                self.performance_metrics['communication_errors'] += 1
    
    def max78000_data_thread(self):
        """Thread for reading MAX78000 data"""
        self.logger.info("MAX78000 data thread started")
        
        while self.is_running:
            try:
                if self.max78000_serial and self.max78000_serial.in_waiting > 0:
                    line = self.max78000_serial.readline().decode('utf-8').strip()
                    self.parse_max78000_data(line)
                    
                time.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                self.logger.error(f"MAX78000 data thread error: {e}")
                self.performance_metrics['communication_errors'] += 1
    
    def parse_stm32_data(self, data: str):
        """Parse STM32 serial data"""
        try:
            if data.startswith("FEATURE:"):
                # Parse feature data
                feature_str = data[8:]  # Remove "FEATURE:" prefix
                features = [float(x) for x in feature_str.split(',')]
                
                if len(features) == 72:
                    timestamp = time.time()
                    self.data_buffer.append({
                        'timestamp': timestamp,
                        'features': features,
                        'type': 'feature'
                    })
                    self.performance_metrics['feature_packets'] += 1
                    
            elif data.startswith("STATUS:"):
                # Parse status data
                status_str = data[7:]
                self.logger.info(f"STM32 Status: {status_str}")
                
            elif data.startswith("ERROR:"):
                # Parse error data
                error_str = data[6:]
                self.logger.error(f"STM32 Error: {error_str}")
                self.performance_metrics['communication_errors'] += 1
                
        except Exception as e:
            self.logger.error(f"STM32 data parsing error: {e}")
    
    def parse_max78000_data(self, data: str):
        """Parse MAX78000 serial data"""
        try:
            if data.startswith("CLASSIFICATION:"):
                # Parse classification result
                result_str = data[15:]  # Remove "CLASSIFICATION:" prefix
                parts = result_str.split(',')
                
                if len(parts) >= 3:
                    class_id = int(parts[0])
                    confidence = float(parts[1])
                    inference_time = float(parts[2])
                    
                    timestamp = time.time()
                    self.classification_buffer.append({
                        'timestamp': timestamp,
                        'class_id': class_id,
                        'confidence': confidence,
                        'inference_time': inference_time,
                        'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Unknown"
                    })
                    
                    self.performance_metrics['classifications'] += 1
                    self.performance_metrics['inference_times'].append(inference_time)
                    
                    if confidence >= self.config['classification_threshold']:
                        self.logger.info(f"Classification: {CLASS_NAMES[class_id]} ({confidence:.2f}), {inference_time:.1f}μs")
                    
            elif data.startswith("STATUS:"):
                # Parse status data
                status_str = data[7:]
                self.logger.info(f"MAX78000 Status: {status_str}")
                
            elif data.startswith("ERROR:"):
                # Parse error data
                error_str = data[6:]
                self.logger.error(f"MAX78000 Error: {error_str}")
                self.performance_metrics['communication_errors'] += 1
                
        except Exception as e:
            self.logger.error(f"MAX78000 data parsing error: {e}")
    
    def system_monitor_thread(self):
        """Thread for monitoring system performance"""
        self.logger.info("System monitor thread started")
        
        start_time = time.time()
        last_report_time = start_time
        
        while self.is_running:
            current_time = time.time()
            self.performance_metrics['system_uptime'] = current_time - start_time
            
            # Report every 30 seconds
            if current_time - last_report_time >= 30.0:
                self.print_performance_report()
                last_report_time = current_time
            
            time.sleep(1.0)
    
    def print_performance_report(self):
        """Print system performance report"""
        uptime = self.performance_metrics['system_uptime']
        
        self.logger.info("=== SYSTEM PERFORMANCE REPORT ===")
        self.logger.info(f"Uptime: {uptime:.1f}s")
        self.logger.info(f"Feature Packets: {self.performance_metrics['feature_packets']}")
        self.logger.info(f"Classifications: {self.performance_metrics['classifications']}")
        self.logger.info(f"Communication Errors: {self.performance_metrics['communication_errors']}")
        
        if self.performance_metrics['inference_times']:
            inference_times = self.performance_metrics['inference_times']
            self.logger.info(f"Avg Inference Time: {np.mean(inference_times):.1f}μs")
            self.logger.info(f"Max Inference Time: {np.max(inference_times):.1f}μs")
        
        # Feature rate
        if uptime > 0:
            feature_rate = self.performance_metrics['feature_packets'] / uptime
            classification_rate = self.performance_metrics['classifications'] / uptime
            self.logger.info(f"Feature Rate: {feature_rate:.1f} Hz")
            self.logger.info(f"Classification Rate: {classification_rate:.1f} Hz")
        
        self.logger.info("=== END REPORT ===")
    
    def run_integration_test(self, duration: int = 300):
        """
        Run complete integration test
        
        Args:
            duration: Test duration in seconds
        """
        self.logger.info(f"Starting {duration}s integration test")
        
        # Connect to devices
        if not self.connect_devices():
            self.logger.error("Failed to connect to devices")
            return False
        
        # Start data acquisition
        if not self.start_data_acquisition():
            self.logger.error("Failed to start data acquisition")
            return False
        
        # Start threads
        self.is_running = True
        self.stm32_thread = threading.Thread(target=self.stm32_data_thread)
        self.max78000_thread = threading.Thread(target=self.max78000_data_thread)
        self.monitor_thread = threading.Thread(target=self.system_monitor_thread)
        
        self.stm32_thread.start()
        self.max78000_thread.start()
        self.monitor_thread.start()
        
        # Run test
        try:
            self.logger.info("Integration test running...")
            time.sleep(duration)
            
        except KeyboardInterrupt:
            self.logger.info("Test interrupted by user")
        
        finally:
            # Stop test
            self.is_running = False
            self.stop_data_acquisition()
            
            # Wait for threads
            if self.stm32_thread:
                self.stm32_thread.join(timeout=5.0)
            if self.max78000_thread:
                self.max78000_thread.join(timeout=5.0)
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            
            # Close connections
            if self.stm32_serial:
                self.stm32_serial.close()
            if self.max78000_serial:
                self.max78000_serial.close()
            
            # Generate final report
            self.generate_test_report()
            
            self.logger.info("Integration test completed")
            return True
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.log_dir, f"integration_report_{timestamp}.json")
        
        # Calculate statistics
        inference_times = self.performance_metrics['inference_times']
        
        report = {
            'test_info': {
                'timestamp': timestamp,
                'duration': self.performance_metrics['system_uptime'],
                'config': self.config
            },
            'performance_metrics': {
                'feature_packets': self.performance_metrics['feature_packets'],
                'classifications': self.performance_metrics['classifications'],
                'communication_errors': self.performance_metrics['communication_errors'],
                'feature_rate': self.performance_metrics['feature_packets'] / self.performance_metrics['system_uptime'] if self.performance_metrics['system_uptime'] > 0 else 0,
                'classification_rate': self.performance_metrics['classifications'] / self.performance_metrics['system_uptime'] if self.performance_metrics['system_uptime'] > 0 else 0
            },
            'inference_statistics': {
                'count': len(inference_times),
                'mean_time': np.mean(inference_times) if inference_times else 0,
                'std_time': np.std(inference_times) if inference_times else 0,
                'min_time': np.min(inference_times) if inference_times else 0,
                'max_time': np.max(inference_times) if inference_times else 0,
                'percentile_95': np.percentile(inference_times, 95) if inference_times else 0
            },
            'classification_distribution': self.get_classification_distribution()
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Test report saved to {report_file}")
        
        # Print summary
        self.logger.info("=== FINAL TEST SUMMARY ===")
        self.logger.info(f"Test Duration: {report['test_info']['duration']:.1f}s")
        self.logger.info(f"Feature Packets: {report['performance_metrics']['feature_packets']}")
        self.logger.info(f"Classifications: {report['performance_metrics']['classifications']}")
        self.logger.info(f"Communication Errors: {report['performance_metrics']['communication_errors']}")
        self.logger.info(f"Feature Rate: {report['performance_metrics']['feature_rate']:.1f} Hz")
        self.logger.info(f"Classification Rate: {report['performance_metrics']['classification_rate']:.1f} Hz")
        
        if inference_times:
            self.logger.info(f"Avg Inference Time: {report['inference_statistics']['mean_time']:.1f}μs")
            self.logger.info(f"95th Percentile: {report['inference_statistics']['percentile_95']:.1f}μs")
        
        self.logger.info("=== END SUMMARY ===")
        
        return report
    
    def get_classification_distribution(self) -> Dict:
        """Get classification distribution statistics"""
        class_counts = {}
        confidence_stats = {}
        
        for entry in self.classification_buffer:
            class_name = entry['class_name']
            confidence = entry['confidence']
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                confidence_stats[class_name] = []
            
            class_counts[class_name] += 1
            confidence_stats[class_name].append(confidence)
        
        # Calculate confidence statistics
        for class_name in confidence_stats:
            confidences = confidence_stats[class_name]
            confidence_stats[class_name] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        return {
            'class_counts': class_counts,
            'confidence_stats': confidence_stats
        }
    
    def plot_real_time_data(self, save_plots: bool = True):
        """Generate real-time data visualization"""
        if not self.data_buffer or not self.classification_buffer:
            self.logger.warning("No data available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Feature data timeline
        feature_times = [entry['timestamp'] for entry in self.data_buffer]
        feature_values = [np.mean(entry['features']) for entry in self.data_buffer]
        
        axes[0, 0].plot(feature_times, feature_values)
        axes[0, 0].set_title('Feature Data Timeline')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Mean Feature Value')
        axes[0, 0].grid(True)
        
        # Plot 2: Classification distribution
        class_counts = self.get_classification_distribution()['class_counts']
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            axes[0, 1].bar(classes, counts)
            axes[0, 1].set_title('Classification Distribution')
            axes[0, 1].set_xlabel('Gesture Class')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Inference time distribution
        inference_times = self.performance_metrics['inference_times']
        if inference_times:
            axes[1, 0].hist(inference_times, bins=30, alpha=0.7)
            axes[1, 0].set_title('Inference Time Distribution')
            axes[1, 0].set_xlabel('Time (μs)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # Plot 4: System performance over time
        uptime = self.performance_metrics['system_uptime']
        axes[1, 1].text(0.1, 0.8, f"Uptime: {uptime:.1f}s", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Feature Packets: {self.performance_metrics['feature_packets']}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Classifications: {self.performance_metrics['classifications']}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Errors: {self.performance_metrics['communication_errors']}", transform=axes[1, 1].transAxes)
        
        if inference_times:
            axes[1, 1].text(0.1, 0.4, f"Avg Inference: {np.mean(inference_times):.1f}μs", transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_title('System Performance Summary')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(self.log_dir, f"system_plots_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plots saved to {plot_file}")
        
        plt.show()


def main():
    """Main function for integration testing"""
    parser = argparse.ArgumentParser(description='EMG System Integration Tester')
    parser.add_argument('--stm32-port', type=str, default='COM3',
                       help='STM32 serial port')
    parser.add_argument('--max78000-port', type=str, default='COM4',
                       help='MAX78000 serial port')
    parser.add_argument('--baudrate', type=int, default=115200,
                       help='Serial baudrate')
    parser.add_argument('--duration', type=int, default=300,
                       help='Test duration in seconds')
    parser.add_argument('--log-dir', type=str, default='integration_logs',
                       help='Log directory')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots after test')
    
    args = parser.parse_args()
    
    # Create tester
    tester = SystemIntegrationTester(
        stm32_port=args.stm32_port,
        max78000_port=args.max78000_port,
        baudrate=args.baudrate,
        log_dir=args.log_dir
    )
    
    # Run integration test
    success = tester.run_integration_test(duration=args.duration)
    
    # Generate plots if requested
    if args.plot:
        tester.plot_real_time_data(save_plots=True)
    
    if success:
        print("Integration test completed successfully!")
        sys.exit(0)
    else:
        print("Integration test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()