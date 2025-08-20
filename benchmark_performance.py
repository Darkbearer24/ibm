#!/usr/bin/env python3
"""
Sprint 7: Performance Benchmarking and Optimization

Comprehensive performance analysis and optimization for the speech translation pipeline.
Includes memory profiling, execution time analysis, and system resource monitoring.
"""

import sys
import os
import time
import psutil
import gc
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile as sf
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class PerformanceProfiler:
    """Performance profiling utilities for the speech translation pipeline."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()
        self.performance_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': {},
            'optimizations': [],
            'recommendations': []
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def profile_function(self, func, *args, **kwargs) -> Dict:
        """Profile a function's performance."""
        # Force garbage collection before profiling
        gc.collect()
        
        # Get initial metrics
        start_memory = self.get_memory_usage()
        start_time = time.perf_counter()
        start_cpu = self.get_cpu_usage()
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Get final metrics
        end_time = time.perf_counter()
        end_memory = self.get_memory_usage()
        end_cpu = self.get_cpu_usage()
        
        # Force garbage collection after profiling
        gc.collect()
        final_memory = self.get_memory_usage()
        
        return {
            'success': success,
            'error': error,
            'result': result,
            'execution_time': end_time - start_time,
            'memory_start': start_memory,
            'memory_peak': end_memory,
            'memory_final': final_memory,
            'memory_delta': end_memory - start_memory,
            'memory_leaked': final_memory - start_memory,
            'cpu_usage': (start_cpu + end_cpu) / 2
        }


class PipelineBenchmark:
    """Comprehensive benchmarking for the speech translation pipeline."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.benchmark_results = {}
        self._create_test_audio_files()
    
    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_audio_files(self):
        """Create test audio files of various sizes."""
        sr = 44100
        
        # Test files with different durations
        test_configs = [
            ('short_1s', 1.0),
            ('medium_5s', 5.0),
            ('long_10s', 10.0),
            ('very_long_30s', 30.0)
        ]
        
        self.test_files = {}
        
        for name, duration in test_configs:
            # Create sine wave with some noise
            t = np.linspace(0, duration, int(sr * duration))
            signal = np.sin(2 * np.pi * 440 * t) * 0.7  # 440 Hz tone
            noise = np.random.normal(0, 0.05, len(signal))
            audio = signal + noise
            
            # Save to file
            file_path = self.temp_dir / f"{name}.wav"
            sf.write(file_path, audio, sr)
            
            self.test_files[name] = {
                'path': str(file_path),
                'duration': duration,
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            }
    
    def benchmark_pipeline_orchestrator(self) -> Dict:
        """Benchmark the Pipeline Orchestrator performance."""
        print("\n🚀 Benchmarking Pipeline Orchestrator")
        print("=" * 50)
        
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            
            # Initialize orchestrator
            def init_orchestrator():
                return PipelineOrchestrator(
                    enable_logging=False  # Disable logging for cleaner benchmarks
                )
            
            init_profile = self.profiler.profile_function(init_orchestrator)
            
            if not init_profile['success']:
                print(f"❌ Failed to initialize orchestrator: {init_profile['error']}")
                return {'status': 'failed', 'error': init_profile['error']}
            
            orchestrator = init_profile['result']
            
            # Benchmark different file sizes
            file_benchmarks = {}
            
            for file_name, file_info in self.test_files.items():
                print(f"\n📊 Testing {file_name} ({file_info['duration']}s, {file_info['size_mb']:.2f}MB)")
                
                def process_file():
                    return orchestrator.process_audio_complete(
                        audio_input=file_info['path'],
                        session_id=f"benchmark_{file_name}"
                    )
                
                # Run multiple iterations for statistical significance
                iterations = 3 if file_info['duration'] <= 10 else 1
                iteration_results = []
                
                for i in range(iterations):
                    profile = self.profiler.profile_function(process_file)
                    iteration_results.append(profile)
                    
                    if profile['success']:
                        print(f"  Iteration {i+1}: {profile['execution_time']:.2f}s, "
                              f"Memory: {profile['memory_delta']:.1f}MB")
                    else:
                        print(f"  Iteration {i+1}: FAILED - {profile['error']}")
                
                # Calculate statistics
                successful_runs = [r for r in iteration_results if r['success']]
                
                if successful_runs:
                    exec_times = [r['execution_time'] for r in successful_runs]
                    memory_deltas = [r['memory_delta'] for r in successful_runs]
                    
                    file_benchmarks[file_name] = {
                        'file_info': file_info,
                        'iterations': len(successful_runs),
                        'success_rate': len(successful_runs) / len(iteration_results),
                        'avg_execution_time': np.mean(exec_times),
                        'min_execution_time': np.min(exec_times),
                        'max_execution_time': np.max(exec_times),
                        'std_execution_time': np.std(exec_times),
                        'avg_memory_delta': np.mean(memory_deltas),
                        'processing_speed': file_info['duration'] / np.mean(exec_times),  # Real-time factor
                        'memory_efficiency': file_info['size_mb'] / np.mean(memory_deltas) if np.mean(memory_deltas) > 0 else float('inf')
                    }
                else:
                    file_benchmarks[file_name] = {
                        'file_info': file_info,
                        'iterations': 0,
                        'success_rate': 0,
                        'error': iteration_results[0]['error'] if iteration_results else 'Unknown error'
                    }
            
            return {
                'status': 'completed',
                'initialization': {
                    'execution_time': init_profile['execution_time'],
                    'memory_usage': init_profile['memory_delta']
                },
                'file_benchmarks': file_benchmarks
            }
            
        except ImportError as e:
            return {'status': 'failed', 'error': f"Pipeline orchestrator not available: {e}"}
        except Exception as e:
            return {'status': 'failed', 'error': f"Benchmark failed: {e}"}
    
    def benchmark_individual_components(self) -> Dict:
        """Benchmark individual pipeline components."""
        print("\n🔧 Benchmarking Individual Components")
        print("=" * 50)
        
        component_benchmarks = {}
        
        # Test audio preprocessing
        try:
            from utils.denoise import preprocess_audio_complete
            
            print("\n📊 Testing Audio Preprocessing")
            
            # Load test audio
            test_file = self.test_files['medium_5s']
            audio, sr = librosa.load(test_file['path'], sr=None)
            
            def preprocess_test():
                return preprocess_audio_complete(audio, sr)
            
            profile = self.profiler.profile_function(preprocess_test)
            
            component_benchmarks['preprocessing'] = {
                'success': profile['success'],
                'execution_time': profile['execution_time'],
                'memory_delta': profile['memory_delta'],
                'throughput': len(audio) / profile['execution_time'] if profile['success'] else 0,
                'error': profile['error']
            }
            
            print(f"  Preprocessing: {profile['execution_time']:.3f}s, Memory: {profile['memory_delta']:.1f}MB")
            
        except ImportError:
            component_benchmarks['preprocessing'] = {'status': 'not_available'}
        
        # Test feature extraction
        try:
            from utils.framing import create_feature_matrix_advanced
            
            print("\n📊 Testing Feature Extraction")
            
            test_file = self.test_files['medium_5s']
            audio, sr = librosa.load(test_file['path'], sr=None)
            
            def feature_extraction_test():
                return create_feature_matrix_advanced(
                    audio, sr,
                    frame_length_ms=20,
                    hop_length_ms=10,
                    n_features=441
                )
            
            profile = self.profiler.profile_function(feature_extraction_test)
            
            component_benchmarks['feature_extraction'] = {
                'success': profile['success'],
                'execution_time': profile['execution_time'],
                'memory_delta': profile['memory_delta'],
                'throughput': len(audio) / profile['execution_time'] if profile['success'] else 0,
                'error': profile['error']
            }
            
            print(f"  Feature Extraction: {profile['execution_time']:.3f}s, Memory: {profile['memory_delta']:.1f}MB")
            
        except ImportError:
            component_benchmarks['feature_extraction'] = {'status': 'not_available'}
        
        # Test model inference
        try:
            from models.encoder_decoder import create_model
            
            print("\n📊 Testing Model Inference")
            
            def model_inference_test():
                model, config = create_model()
                model.eval()
                
                # Create dummy input
                batch_size, seq_len, n_features = 1, 100, 441
                dummy_input = torch.randn(batch_size, seq_len, n_features)
                
                with torch.no_grad():
                    output, latent = model(dummy_input)
                
                return output, latent
            
            profile = self.profiler.profile_function(model_inference_test)
            
            component_benchmarks['model_inference'] = {
                'success': profile['success'],
                'execution_time': profile['execution_time'],
                'memory_delta': profile['memory_delta'],
                'error': profile['error']
            }
            
            print(f"  Model Inference: {profile['execution_time']:.3f}s, Memory: {profile['memory_delta']:.1f}MB")
            
        except ImportError:
            component_benchmarks['model_inference'] = {'status': 'not_available'}
        
        return component_benchmarks
    
    def benchmark_concurrent_processing(self) -> Dict:
        """Benchmark concurrent processing capabilities."""
        print("\n🔄 Benchmarking Concurrent Processing")
        print("=" * 50)
        
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            
            orchestrator = PipelineOrchestrator(
                enable_logging=False
            )
            
            # Test different concurrency levels
            concurrency_results = {}
            test_file = self.test_files['short_1s']  # Use short file for faster testing
            
            for num_threads in [1, 2, 4]:
                print(f"\n📊 Testing {num_threads} concurrent requests")
                
                def process_concurrent():
                    def single_request(request_id):
                        return orchestrator.process_audio_complete(
                            audio_input=test_file['path'],
                            session_id=f"concurrent_{num_threads}_{request_id}"
                        )
                    
                    start_time = time.perf_counter()
                    
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        futures = [executor.submit(single_request, i) for i in range(num_threads)]
                        results = [future.result() for future in as_completed(futures)]
                    
                    end_time = time.perf_counter()
                    
                    return {
                        'results': results,
                        'total_time': end_time - start_time,
                        'successful_requests': sum(1 for r in results if r and r.get('success', False))
                    }
                
                profile = self.profiler.profile_function(process_concurrent)
                
                if profile['success']:
                    concurrent_data = profile['result']
                    
                    concurrency_results[f"{num_threads}_threads"] = {
                        'num_threads': num_threads,
                        'total_execution_time': profile['execution_time'],
                        'successful_requests': concurrent_data['successful_requests'],
                        'success_rate': concurrent_data['successful_requests'] / num_threads,
                        'throughput': concurrent_data['successful_requests'] / profile['execution_time'],
                        'memory_delta': profile['memory_delta'],
                        'efficiency': concurrent_data['successful_requests'] / (profile['execution_time'] * num_threads)
                    }
                    
                    print(f"  {num_threads} threads: {profile['execution_time']:.2f}s, "
                          f"Success: {concurrent_data['successful_requests']}/{num_threads}, "
                          f"Memory: {profile['memory_delta']:.1f}MB")
                else:
                    concurrency_results[f"{num_threads}_threads"] = {
                        'num_threads': num_threads,
                        'error': profile['error']
                    }
            
            return {
                'status': 'completed',
                'results': concurrency_results
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage patterns."""
        print("\n💾 Benchmarking Memory Usage")
        print("=" * 50)
        
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            
            orchestrator = PipelineOrchestrator(
                enable_logging=False
            )
            
            memory_results = {}
            
            # Test memory usage with different file sizes
            for file_name, file_info in self.test_files.items():
                if file_info['duration'] > 10:  # Skip very long files for memory test
                    continue
                
                print(f"\n📊 Memory test: {file_name}")
                
                # Monitor memory during processing
                memory_samples = []
                
                def memory_monitor():
                    while not stop_monitoring:
                        memory_samples.append(self.profiler.get_memory_usage())
                        time.sleep(0.1)  # Sample every 100ms
                
                stop_monitoring = False
                monitor_thread = threading.Thread(target=memory_monitor)
                monitor_thread.start()
                
                try:
                    # Process file
                    result = orchestrator.process_audio_complete(
                        audio_input=file_info['path'],
                        session_id=f"memory_test_{file_name}"
                    )
                    
                    # Stop monitoring
                    stop_monitoring = True
                    monitor_thread.join(timeout=1)
                    
                    if memory_samples:
                        memory_results[file_name] = {
                            'file_info': file_info,
                            'success': result and result.get('success', False),
                            'baseline_memory': memory_samples[0],
                            'peak_memory': max(memory_samples),
                            'final_memory': memory_samples[-1],
                            'memory_increase': max(memory_samples) - memory_samples[0],
                            'memory_leaked': memory_samples[-1] - memory_samples[0],
                            'memory_efficiency': file_info['size_mb'] / (max(memory_samples) - memory_samples[0]) if max(memory_samples) - memory_samples[0] > 0 else float('inf')
                        }
                        
                        print(f"  Peak memory: {max(memory_samples):.1f}MB, "
                              f"Increase: {max(memory_samples) - memory_samples[0]:.1f}MB")
                    
                except Exception as e:
                    stop_monitoring = True
                    monitor_thread.join(timeout=1)
                    memory_results[file_name] = {
                        'file_info': file_info,
                        'error': str(e)
                    }
            
            return {
                'status': 'completed',
                'results': memory_results
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        print("\n📈 Generating Performance Report")
        print("=" * 50)
        
        # Run all benchmarks
        self.benchmark_results['pipeline_orchestrator'] = self.benchmark_pipeline_orchestrator()
        self.benchmark_results['individual_components'] = self.benchmark_individual_components()
        self.benchmark_results['concurrent_processing'] = self.benchmark_concurrent_processing()
        self.benchmark_results['memory_usage'] = self.benchmark_memory_usage()
        
        # Analyze results and generate recommendations
        recommendations = self._analyze_performance_results()
        
        # Create final report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.profiler.performance_data['system_info'],
            'benchmark_results': self.benchmark_results,
            'performance_analysis': recommendations,
            'deployment_readiness': self._assess_deployment_readiness()
        }
        
        return report
    
    def _analyze_performance_results(self) -> Dict:
        """Analyze benchmark results and generate recommendations."""
        analysis = {
            'performance_metrics': {},
            'bottlenecks': [],
            'optimizations': [],
            'recommendations': []
        }
        
        # Analyze pipeline orchestrator results
        pipeline_results = self.benchmark_results.get('pipeline_orchestrator', {})
        if pipeline_results.get('status') == 'completed':
            file_benchmarks = pipeline_results.get('file_benchmarks', {})
            
            # Calculate average processing speed
            speeds = []
            for file_name, benchmark in file_benchmarks.items():
                if 'processing_speed' in benchmark:
                    speeds.append(benchmark['processing_speed'])
            
            if speeds:
                avg_speed = np.mean(speeds)
                analysis['performance_metrics']['avg_processing_speed'] = avg_speed
                
                if avg_speed < 1.0:
                    analysis['bottlenecks'].append("Processing speed is slower than real-time")
                    analysis['optimizations'].append("Consider model optimization or hardware acceleration")
                elif avg_speed > 5.0:
                    analysis['performance_metrics']['real_time_capable'] = True
                    analysis['recommendations'].append("System can handle real-time processing with good margin")
        
        # Analyze memory usage
        memory_results = self.benchmark_results.get('memory_usage', {})
        if memory_results.get('status') == 'completed':
            memory_data = memory_results.get('results', {})
            
            max_memory_increase = 0
            memory_leaks = []
            
            for file_name, data in memory_data.items():
                if 'memory_increase' in data:
                    max_memory_increase = max(max_memory_increase, data['memory_increase'])
                
                if 'memory_leaked' in data and data['memory_leaked'] > 10:  # > 10MB leak
                    memory_leaks.append(file_name)
            
            analysis['performance_metrics']['max_memory_increase'] = max_memory_increase
            
            if max_memory_increase > 500:  # > 500MB
                analysis['bottlenecks'].append("High memory usage detected")
                analysis['optimizations'].append("Implement memory optimization strategies")
            
            if memory_leaks:
                analysis['bottlenecks'].append(f"Memory leaks detected in: {', '.join(memory_leaks)}")
                analysis['optimizations'].append("Fix memory leaks in pipeline components")
        
        # Analyze concurrent processing
        concurrent_results = self.benchmark_results.get('concurrent_processing', {})
        if concurrent_results.get('status') == 'completed':
            results = concurrent_results.get('results', {})
            
            # Check scalability
            thread_performances = []
            for thread_config, data in results.items():
                if 'efficiency' in data:
                    thread_performances.append((data['num_threads'], data['efficiency']))
            
            if len(thread_performances) >= 2:
                # Check if efficiency decreases with more threads
                thread_performances.sort()
                if thread_performances[-1][1] < thread_performances[0][1] * 0.8:
                    analysis['bottlenecks'].append("Poor scalability with concurrent requests")
                    analysis['optimizations'].append("Optimize for concurrent processing")
                else:
                    analysis['recommendations'].append("System scales well with concurrent requests")
        
        return analysis
    
    def _assess_deployment_readiness(self) -> Dict:
        """Assess deployment readiness based on performance benchmarks."""
        readiness = {
            'overall_status': 'UNKNOWN',
            'criteria': {},
            'blockers': [],
            'warnings': [],
            'ready_for_deployment': False
        }
        
        # Check performance criteria
        pipeline_results = self.benchmark_results.get('pipeline_orchestrator', {})
        
        if pipeline_results.get('status') == 'completed':
            file_benchmarks = pipeline_results.get('file_benchmarks', {})
            
            # Check if all file sizes processed successfully
            success_rates = []
            for file_name, benchmark in file_benchmarks.items():
                if 'success_rate' in benchmark:
                    success_rates.append(benchmark['success_rate'])
            
            if success_rates:
                avg_success_rate = np.mean(success_rates)
                readiness['criteria']['success_rate'] = avg_success_rate
                
                if avg_success_rate < 0.8:
                    readiness['blockers'].append(f"Low success rate: {avg_success_rate:.1%}")
                elif avg_success_rate < 0.95:
                    readiness['warnings'].append(f"Moderate success rate: {avg_success_rate:.1%}")
        
        # Check memory usage
        memory_results = self.benchmark_results.get('memory_usage', {})
        if memory_results.get('status') == 'completed':
            memory_data = memory_results.get('results', {})
            
            max_memory = 0
            for file_name, data in memory_data.items():
                if 'peak_memory' in data:
                    max_memory = max(max_memory, data['peak_memory'])
            
            readiness['criteria']['max_memory_usage'] = max_memory
            
            if max_memory > 2000:  # > 2GB
                readiness['blockers'].append(f"Excessive memory usage: {max_memory:.0f}MB")
            elif max_memory > 1000:  # > 1GB
                readiness['warnings'].append(f"High memory usage: {max_memory:.0f}MB")
        
        # Determine overall status
        if readiness['blockers']:
            readiness['overall_status'] = 'NOT_READY'
            readiness['ready_for_deployment'] = False
        elif readiness['warnings']:
            readiness['overall_status'] = 'READY_WITH_WARNINGS'
            readiness['ready_for_deployment'] = True
        else:
            readiness['overall_status'] = 'READY'
            readiness['ready_for_deployment'] = True
        
        return readiness
    
    def save_report(self, report: Dict, filename: str = "performance_report.json"):
        """Save performance report to file."""
        report_path = Path(filename)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Performance report saved to: {report_path.absolute()}")
        return report_path
    
    def print_summary(self, report: Dict):
        """Print performance summary."""
        print("\n" + "=" * 80)
        print("[*] PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        
        # System info
        system_info = report.get('system_info', {})
        print(f"\n[*] System Information:")
        print(f"   CPU Cores: {system_info.get('cpu_count', 'Unknown')}")
        print(f"   Memory: {system_info.get('memory_total', 0):.1f}GB total, {system_info.get('memory_available', 0):.1f}GB available")
        print(f"   CUDA Available: {system_info.get('cuda_available', False)}")
        
        # Performance metrics
        analysis = report.get('performance_analysis', {})
        metrics = analysis.get('performance_metrics', {})
        
        print(f"\n[*] Performance Metrics:")
        if 'avg_processing_speed' in metrics:
            speed = metrics['avg_processing_speed']
            print(f"   Average Processing Speed: {speed:.2f}x real-time")
            if speed >= 1.0:
                print(f"   [OK] Real-time processing capable")
            else:
                print(f"   [WARNING] Slower than real-time")
        
        if 'max_memory_increase' in metrics:
            memory = metrics['max_memory_increase']
            print(f"   Maximum Memory Increase: {memory:.1f}MB")
        
        # Deployment readiness
        readiness = report.get('deployment_readiness', {})
        status = readiness.get('overall_status', 'UNKNOWN')
        
        print(f"\n[*] Deployment Readiness: {status}")
        
        blockers = readiness.get('blockers', [])
        if blockers:
            print(f"\n[ERROR] Blockers:")
            for blocker in blockers:
                print(f"   • {blocker}")
        
        warnings = readiness.get('warnings', [])
        if warnings:
            print(f"\n[WARNING] Warnings:")
            for warning in warnings:
                print(f"   • {warning}")
        
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        optimizations = analysis.get('optimizations', [])
        
        if recommendations or optimizations:
            print(f"\n[*] Recommendations:")
            for rec in recommendations:
                print(f"   [OK] {rec}")
            for opt in optimizations:
                print(f"   [*] {opt}")


def main():
    """Main function to run performance benchmarks."""
    print("[*] Starting Performance Benchmarking Suite")
    print("=" * 80)
    
    benchmark = PipelineBenchmark()
    
    try:
        # Generate comprehensive performance report
        report = benchmark.generate_performance_report()
        
        # Save report
        report_path = benchmark.save_report(report)
        
        # Print summary
        benchmark.print_summary(report)
        
        # Return deployment readiness status
        readiness = report.get('deployment_readiness', {})
        if readiness.get('ready_for_deployment', False):
            print("\n🎉 System is ready for deployment!")
            return 0
        else:
            print("\n⚠️  System needs optimization before deployment.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmarking interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n💥 Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        del benchmark


if __name__ == "__main__":
    sys.exit(main())