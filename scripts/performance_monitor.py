#!/usr/bin/env python3
"""
System Performance Monitor for Dual-MCU EMG System
Real-time monitoring and analysis of system performance
"""

import os
import sys
import time
import json
import psutil
import threading
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import logging


class SystemPerformanceMonitor:
    """
    Real-time system performance monitor
    """
    
    def __init__(self, 
                 monitor_interval: float = 1.0,
                 history_size: int = 3600,
                 log_dir: str = "performance_logs"):
        """
        Initialize performance monitor
        
        Args:
            monitor_interval: Monitoring interval in seconds
            history_size: Size of performance history buffer
            log_dir: Directory for performance logs
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Performance data buffers
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.network_history = deque(maxlen=history_size)
        self.disk_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # System metrics
        self.system_metrics = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_available': 0,
            'network_bytes_sent': 0,
            'network_bytes_recv': 0,
            'disk_read_bytes': 0,
            'disk_write_bytes': 0,
            'processes': 0,
            'uptime': 0.0
        }
        
        # EMG system specific metrics
        self.emg_metrics = {
            'feature_rate': 0.0,
            'classification_rate': 0.0,
            'inference_time': 0.0,
            'communication_errors': 0,
            'data_throughput': 0.0,
            'stm32_cpu_usage': 0.0,
            'max78000_cpu_usage': 0.0,
            'memory_usage': 0.0
        }
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'inference_time_warning': 5000.0,  # 5ms
            'inference_time_critical': 10000.0,  # 10ms
            'feature_rate_min': 90.0,  # 90Hz minimum
            'classification_rate_min': 1.0  # 1Hz minimum
        }
        
        # Threading
        self.is_monitoring = False
        self.monitor_thread = None
        self.alert_thread = None
        
        # Alerts
        self.alert_history = deque(maxlen=100)
        self.alert_callbacks = []
        
        # Start time
        self.start_time = time.time()
        
        # Network baseline
        self.network_baseline = psutil.net_io_counters()
        self.disk_baseline = psutil.disk_io_counters()
    
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"performance_monitor_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("System Performance Monitor Initialized")
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            self.system_metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.system_metrics['memory_percent'] = memory.percent
            self.system_metrics['memory_available'] = memory.available
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.system_metrics['network_bytes_sent'] = net_io.bytes_sent - self.network_baseline.bytes_sent
            self.system_metrics['network_bytes_recv'] = net_io.bytes_recv - self.network_baseline.bytes_recv
            
            # Disk metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.system_metrics['disk_read_bytes'] = disk_io.read_bytes - self.disk_baseline.read_bytes
                self.system_metrics['disk_write_bytes'] = disk_io.write_bytes - self.disk_baseline.write_bytes
            
            # Process count
            self.system_metrics['processes'] = len(psutil.pids())
            
            # System uptime
            self.system_metrics['uptime'] = time.time() - self.start_time
            
            # Store in history
            current_time = time.time()
            self.timestamp_history.append(current_time)
            self.cpu_history.append(self.system_metrics['cpu_percent'])
            self.memory_history.append(self.system_metrics['memory_percent'])
            self.network_history.append(self.system_metrics['network_bytes_sent'] + self.system_metrics['network_bytes_recv'])
            self.disk_history.append(self.system_metrics['disk_read_bytes'] + self.system_metrics['disk_write_bytes'])
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def collect_emg_metrics(self):
        """Collect EMG system specific metrics"""
        try:
            # This would be populated by the integration system
            # For now, we'll use placeholder values
            
            # Feature extraction rate (should be ~100Hz)
            self.emg_metrics['feature_rate'] = np.random.normal(100.0, 2.0)
            
            # Classification rate (variable based on activity)
            self.emg_metrics['classification_rate'] = np.random.normal(5.0, 1.0)
            
            # Inference time (should be <1ms)
            self.emg_metrics['inference_time'] = np.random.normal(800.0, 100.0)  # microseconds
            
            # Communication errors (should be minimal)
            self.emg_metrics['communication_errors'] = 0
            
            # Data throughput (features per second)
            self.emg_metrics['data_throughput'] = self.emg_metrics['feature_rate'] * 72  # 72 features per packet
            
            # MCU CPU usage estimates
            self.emg_metrics['stm32_cpu_usage'] = np.random.normal(60.0, 5.0)
            self.emg_metrics['max78000_cpu_usage'] = np.random.normal(30.0, 3.0)
            
            # Memory usage
            self.emg_metrics['memory_usage'] = np.random.normal(45.0, 5.0)
            
        except Exception as e:
            self.logger.error(f"Error collecting EMG metrics: {e}")
    
    def check_performance_thresholds(self):
        """Check performance against thresholds and generate alerts"""
        alerts = []
        
        # CPU threshold check
        if self.system_metrics['cpu_percent'] > self.thresholds['cpu_critical']:
            alerts.append(('CRITICAL', 'CPU', f"CPU usage critical: {self.system_metrics['cpu_percent']:.1f}%"))
        elif self.system_metrics['cpu_percent'] > self.thresholds['cpu_warning']:
            alerts.append(('WARNING', 'CPU', f"CPU usage high: {self.system_metrics['cpu_percent']:.1f}%"))
        
        # Memory threshold check
        if self.system_metrics['memory_percent'] > self.thresholds['memory_critical']:
            alerts.append(('CRITICAL', 'MEMORY', f"Memory usage critical: {self.system_metrics['memory_percent']:.1f}%"))
        elif self.system_metrics['memory_percent'] > self.thresholds['memory_warning']:
            alerts.append(('WARNING', 'MEMORY', f"Memory usage high: {self.system_metrics['memory_percent']:.1f}%"))
        
        # EMG system threshold checks
        if self.emg_metrics['inference_time'] > self.thresholds['inference_time_critical']:
            alerts.append(('CRITICAL', 'INFERENCE', f"Inference time critical: {self.emg_metrics['inference_time']:.1f}μs"))
        elif self.emg_metrics['inference_time'] > self.thresholds['inference_time_warning']:
            alerts.append(('WARNING', 'INFERENCE', f"Inference time high: {self.emg_metrics['inference_time']:.1f}μs"))
        
        if self.emg_metrics['feature_rate'] < self.thresholds['feature_rate_min']:
            alerts.append(('WARNING', 'FEATURES', f"Feature rate low: {self.emg_metrics['feature_rate']:.1f}Hz"))
        
        if self.emg_metrics['classification_rate'] < self.thresholds['classification_rate_min']:
            alerts.append(('WARNING', 'CLASSIFICATION', f"Classification rate low: {self.emg_metrics['classification_rate']:.1f}Hz"))
        
        # Process alerts
        for level, category, message in alerts:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'category': category,
                'message': message
            }
            
            self.alert_history.append(alert)
            
            if level == 'CRITICAL':
                self.logger.critical(message)
            else:
                self.logger.warning(message)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("Performance monitoring started")
        
        while self.is_monitoring:
            try:
                # Collect metrics
                self.collect_system_metrics()
                self.collect_emg_metrics()
                
                # Check thresholds
                self.check_performance_thresholds()
                
                # Sleep until next interval
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(1.0)
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_monitoring:
            self.logger.warning("Monitoring not started")
            return
        
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def add_alert_callback(self, callback):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            'system': self.system_metrics.copy(),
            'emg': self.emg_metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        if not self.cpu_history:
            return {}
        
        summary = {
            'monitoring_duration': time.time() - self.start_time,
            'data_points': len(self.cpu_history),
            'cpu_stats': {
                'current': self.system_metrics['cpu_percent'],
                'average': np.mean(self.cpu_history),
                'max': np.max(self.cpu_history),
                'min': np.min(self.cpu_history),
                'std': np.std(self.cpu_history)
            },
            'memory_stats': {
                'current': self.system_metrics['memory_percent'],
                'average': np.mean(self.memory_history),
                'max': np.max(self.memory_history),
                'min': np.min(self.memory_history),
                'std': np.std(self.memory_history)
            },
            'emg_stats': {
                'feature_rate': self.emg_metrics['feature_rate'],
                'classification_rate': self.emg_metrics['classification_rate'],
                'inference_time': self.emg_metrics['inference_time'],
                'data_throughput': self.emg_metrics['data_throughput']
            },
            'alert_summary': {
                'total_alerts': len(self.alert_history),
                'critical_alerts': len([a for a in self.alert_history if a['level'] == 'CRITICAL']),
                'warning_alerts': len([a for a in self.alert_history if a['level'] == 'WARNING'])
            }
        }
        
        return summary
    
    def save_performance_report(self, filename: Optional[str] = None):
        """Save performance report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.log_dir, f"performance_report_{timestamp}.json")
        
        report = {
            'report_info': {
                'timestamp': datetime.now().isoformat(),
                'monitoring_duration': time.time() - self.start_time,
                'monitor_interval': self.monitor_interval
            },
            'current_metrics': self.get_current_metrics(),
            'performance_summary': self.get_performance_summary(),
            'thresholds': self.thresholds,
            'alert_history': list(self.alert_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report saved to {filename}")
        return filename
    
    def plot_performance_graphs(self, save_plots: bool = True):
        """Generate performance visualization plots"""
        if not self.cpu_history:
            self.logger.warning("No performance data available for plotting")
            return
        
        # Create time axis
        if len(self.timestamp_history) > 0:
            times = [(t - self.timestamp_history[0]) / 60.0 for t in self.timestamp_history]  # Convert to minutes
        else:
            times = list(range(len(self.cpu_history)))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Performance Monitor', fontsize=16)
        
        # CPU usage plot
        axes[0, 0].plot(times, self.cpu_history, 'b-', linewidth=1)
        axes[0, 0].axhline(y=self.thresholds['cpu_warning'], color='orange', linestyle='--', alpha=0.7, label='Warning')
        axes[0, 0].axhline(y=self.thresholds['cpu_critical'], color='red', linestyle='--', alpha=0.7, label='Critical')
        axes[0, 0].set_title('CPU Usage')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 100)
        
        # Memory usage plot
        axes[0, 1].plot(times, self.memory_history, 'g-', linewidth=1)
        axes[0, 1].axhline(y=self.thresholds['memory_warning'], color='orange', linestyle='--', alpha=0.7, label='Warning')
        axes[0, 1].axhline(y=self.thresholds['memory_critical'], color='red', linestyle='--', alpha=0.7, label='Critical')
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Memory Usage (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 100)
        
        # Network throughput plot
        if len(self.network_history) > 1:
            # Calculate network rate
            network_rates = []
            for i in range(1, len(self.network_history)):
                rate = (self.network_history[i] - self.network_history[i-1]) / self.monitor_interval
                network_rates.append(rate / 1024)  # Convert to KB/s
            
            axes[1, 0].plot(times[1:], network_rates, 'r-', linewidth=1)
            axes[1, 0].set_title('Network Throughput')
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('Network Rate (KB/s)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # EMG performance summary
        axes[1, 1].text(0.1, 0.9, 'EMG System Performance', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f'Feature Rate: {self.emg_metrics["feature_rate"]:.1f} Hz', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Classification Rate: {self.emg_metrics["classification_rate"]:.1f} Hz', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Inference Time: {self.emg_metrics["inference_time"]:.1f} μs', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Data Throughput: {self.emg_metrics["data_throughput"]:.0f} features/s', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'STM32 CPU: {self.emg_metrics["stm32_cpu_usage"]:.1f}%', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f'MAX78000 CPU: {self.emg_metrics["max78000_cpu_usage"]:.1f}%', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f'Memory Usage: {self.emg_metrics["memory_usage"]:.1f}%', transform=axes[1, 1].transAxes)
        
        # Alert summary
        alert_summary = self.get_performance_summary().get('alert_summary', {})
        axes[1, 1].text(0.1, 0.1, f'Alerts: {alert_summary.get("total_alerts", 0)} ({alert_summary.get("critical_alerts", 0)} critical)', 
                       transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(self.log_dir, f"performance_plots_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Performance plots saved to {plot_file}")
        
        plt.show()
    
    def print_live_status(self):
        """Print live status to console"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("EMG SYSTEM PERFORMANCE MONITOR")
        print("=" * 60)
        print(f"Monitoring Time: {time.time() - self.start_time:.1f}s")
        print(f"Update Interval: {self.monitor_interval:.1f}s")
        print()
        
        print("SYSTEM METRICS:")
        print(f"  CPU Usage:     {self.system_metrics['cpu_percent']:6.1f}%")
        print(f"  Memory Usage:  {self.system_metrics['memory_percent']:6.1f}%")
        print(f"  Available RAM: {self.system_metrics['memory_available']/1024/1024/1024:6.1f} GB")
        print(f"  Processes:     {self.system_metrics['processes']:6d}")
        print()
        
        print("EMG SYSTEM METRICS:")
        print(f"  Feature Rate:        {self.emg_metrics['feature_rate']:8.1f} Hz")
        print(f"  Classification Rate: {self.emg_metrics['classification_rate']:8.1f} Hz")
        print(f"  Inference Time:      {self.emg_metrics['inference_time']:8.1f} μs")
        print(f"  Data Throughput:     {self.emg_metrics['data_throughput']:8.0f} features/s")
        print(f"  STM32 CPU Usage:     {self.emg_metrics['stm32_cpu_usage']:8.1f}%")
        print(f"  MAX78000 CPU Usage:  {self.emg_metrics['max78000_cpu_usage']:8.1f}%")
        print()
        
        print("RECENT ALERTS:")
        recent_alerts = list(self.alert_history)[-5:]  # Last 5 alerts
        if recent_alerts:
            for alert in recent_alerts:
                print(f"  [{alert['level']}] {alert['category']}: {alert['message']}")
        else:
            print("  No recent alerts")
        
        print()
        print("Press Ctrl+C to stop monitoring")


def main():
    """Main function for performance monitoring"""
    parser = argparse.ArgumentParser(description='EMG System Performance Monitor')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Monitoring interval in seconds')
    parser.add_argument('--history-size', type=int, default=3600,
                       help='Size of performance history buffer')
    parser.add_argument('--log-dir', type=str, default='performance_logs',
                       help='Log directory')
    parser.add_argument('--duration', type=int, default=0,
                       help='Monitoring duration in seconds (0 = infinite)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots after monitoring')
    parser.add_argument('--live-display', action='store_true',
                       help='Show live performance display')
    
    args = parser.parse_args()
    
    # Create performance monitor
    monitor = SystemPerformanceMonitor(
        monitor_interval=args.interval,
        history_size=args.history_size,
        log_dir=args.log_dir
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        start_time = time.time()
        
        while True:
            if args.live_display:
                monitor.print_live_status()
            
            # Check duration
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Save report
        report_file = monitor.save_performance_report()
        print(f"Performance report saved to: {report_file}")
        
        # Generate plots if requested
        if args.plot:
            monitor.plot_performance_graphs(save_plots=True)
        
        # Print final summary
        summary = monitor.get_performance_summary()
        print("\nFINAL PERFORMANCE SUMMARY:")
        print(f"  Monitoring Duration: {summary.get('monitoring_duration', 0):.1f}s")
        print(f"  Data Points: {summary.get('data_points', 0)}")
        print(f"  Average CPU: {summary.get('cpu_stats', {}).get('average', 0):.1f}%")
        print(f"  Average Memory: {summary.get('memory_stats', {}).get('average', 0):.1f}%")
        print(f"  Total Alerts: {summary.get('alert_summary', {}).get('total_alerts', 0)}")


if __name__ == "__main__":
    main()