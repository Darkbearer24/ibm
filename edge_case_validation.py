"""Edge Case Validation for Reconstruction Pipeline

This script performs comprehensive robustness testing of the reconstruction pipeline
with various edge cases, boundary conditions, and stress scenarios.

Author: IBM Internship Project
Date: Sprint 6 - Signal Reconstruction & Evaluation
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from typing import List, Dict, Tuple, Optional
import json
import time
import gc
import psutil
import os

# Import our utilities
from utils.reconstruction_pipeline import ReconstructionPipeline
from utils.reconstruction import reconstruct_audio_overlap_add
from utils.evaluation import AudioEvaluator


class EdgeCaseValidator:
    """Comprehensive edge case validation for reconstruction pipeline."""
    
    def __init__(self, pipeline: ReconstructionPipeline):
        self.pipeline = pipeline
        self.results = []
        self.performance_metrics = []
        
    def validate_data_corruption(self) -> List[Dict]:
        """Test reconstruction with various types of data corruption."""
        print("\n=== Data Corruption Validation ===")
        corruption_tests = []
        
        # Base feature matrix for corruption tests
        base_features = np.random.randn(100, 128) * 0.5
        
        # Test 1: Random bit flips
        print("\nTest 1: Random bit corruption")
        corrupted_features = base_features.copy()
        corruption_mask = np.random.random(corrupted_features.shape) < 0.01  # 1% corruption
        corrupted_features[corruption_mask] = np.random.randn(np.sum(corruption_mask)) * 10
        
        result = self._run_test(
            "bit_corruption", 
            corrupted_features, 
            "Testing resilience to random bit corruption"
        )
        corruption_tests.append(result)
        
        # Test 2: Systematic zeros
        print("\nTest 2: Systematic zero corruption")
        zero_corrupted = base_features.copy()
        zero_corrupted[::10, :] = 0  # Every 10th frame is zero
        
        result = self._run_test(
            "zero_corruption", 
            zero_corrupted, 
            "Testing resilience to systematic zero frames"
        )
        corruption_tests.append(result)
        
        # Test 3: Gaussian noise injection
        print("\nTest 3: Gaussian noise injection")
        noisy_features = base_features + np.random.normal(0, 0.1, base_features.shape)
        
        result = self._run_test(
            "gaussian_noise", 
            noisy_features, 
            "Testing resilience to Gaussian noise"
        )
        corruption_tests.append(result)
        
        # Test 4: Impulse noise
        print("\nTest 4: Impulse noise corruption")
        impulse_corrupted = base_features.copy()
        impulse_positions = np.random.random(base_features.shape) < 0.005  # 0.5% impulses
        impulse_corrupted[impulse_positions] = np.random.choice([-10, 10], np.sum(impulse_positions))
        
        result = self._run_test(
            "impulse_noise", 
            impulse_corrupted, 
            "Testing resilience to impulse noise"
        )
        corruption_tests.append(result)
        
        # Test 5: Missing data blocks
        print("\nTest 5: Missing data blocks")
        block_missing = base_features.copy()
        block_missing[40:60, :] = np.nan  # Missing block
        
        result = self._run_test(
            "missing_blocks", 
            block_missing, 
            "Testing resilience to missing data blocks"
        )
        corruption_tests.append(result)
        
        return corruption_tests
    
    def validate_boundary_conditions(self) -> List[Dict]:
        """Test reconstruction with boundary conditions."""
        print("\n=== Boundary Conditions Validation ===")
        boundary_tests = []
        
        # Test 1: Minimum size features
        print("\nTest 1: Minimum size features")
        min_features = np.random.randn(1, 1) * 0.5
        
        result = self._run_test(
            "minimum_size", 
            min_features, 
            "Testing minimum possible feature size"
        )
        boundary_tests.append(result)
        
        # Test 2: Maximum practical size
        print("\nTest 2: Large feature matrix (memory test)")
        try:
            # Check available memory
            available_memory = psutil.virtual_memory().available
            max_elements = min(10000 * 1000, available_memory // (8 * 4))  # Conservative estimate
            
            if max_elements > 1000000:  # Only test if we have enough memory
                rows = int(np.sqrt(max_elements // 100))
                cols = 100
                large_features = np.random.randn(rows, cols) * 0.1
                
                result = self._run_test(
                    "large_matrix", 
                    large_features, 
                    f"Testing large matrix ({rows}x{cols})"
                )
            else:
                result = {
                    'test_name': 'large_matrix',
                    'status': 'SKIPPED',
                    'reason': 'Insufficient memory for large matrix test',
                    'description': 'Testing large matrix processing'
                }
                print("  ⚠ Skipped: Insufficient memory")
        except Exception as e:
            result = {
                'test_name': 'large_matrix',
                'status': 'ERROR',
                'error': str(e),
                'description': 'Testing large matrix processing'
            }
            print(f"  ✗ Error: {e}")
        
        boundary_tests.append(result)
        
        # Test 3: Extreme aspect ratios
        print("\nTest 3: Extreme aspect ratios")
        
        # Very wide matrix
        wide_features = np.random.randn(5, 1000) * 0.5
        result = self._run_test(
            "wide_matrix", 
            wide_features, 
            "Testing very wide feature matrix"
        )
        boundary_tests.append(result)
        
        # Very tall matrix
        tall_features = np.random.randn(1000, 5) * 0.5
        result = self._run_test(
            "tall_matrix", 
            tall_features, 
            "Testing very tall feature matrix"
        )
        boundary_tests.append(result)
        
        # Test 4: Numerical precision limits
        print("\nTest 4: Numerical precision limits")
        
        # Very small values
        tiny_features = np.random.randn(50, 50) * 1e-10
        result = self._run_test(
            "tiny_values", 
            tiny_features, 
            "Testing very small numerical values"
        )
        boundary_tests.append(result)
        
        # Very large values (but not infinite)
        huge_features = np.random.randn(50, 50) * 1e6
        result = self._run_test(
            "huge_values", 
            huge_features, 
            "Testing very large numerical values"
        )
        boundary_tests.append(result)
        
        return boundary_tests
    
    def validate_data_types_and_formats(self) -> List[Dict]:
        """Test reconstruction with different data types and formats."""
        print("\n=== Data Types and Formats Validation ===")
        format_tests = []
        
        base_data = np.random.randn(50, 100) * 0.5
        
        # Test different numpy data types
        data_types = [
            (np.float32, 'float32'),
            (np.float64, 'float64'),
            (np.int16, 'int16'),
            (np.int32, 'int32')
        ]
        
        for dtype, dtype_name in data_types:
            print(f"\nTest: {dtype_name} data type")
            try:
                if dtype in [np.int16, np.int32]:
                    # Scale and convert for integer types
                    scaled_data = (base_data * 1000).astype(dtype)
                else:
                    scaled_data = base_data.astype(dtype)
                
                result = self._run_test(
                    f"dtype_{dtype_name}", 
                    scaled_data, 
                    f"Testing {dtype_name} data type"
                )
            except Exception as e:
                result = {
                    'test_name': f'dtype_{dtype_name}',
                    'status': 'ERROR',
                    'error': str(e),
                    'description': f'Testing {dtype_name} data type'
                }
                print(f"  ✗ Error: {e}")
            
            format_tests.append(result)
        
        # Test different array layouts
        print("\nTest: Non-contiguous array")
        non_contiguous = base_data[::2, ::2]  # Non-contiguous view
        result = self._run_test(
            "non_contiguous", 
            non_contiguous, 
            "Testing non-contiguous array layout"
        )
        format_tests.append(result)
        
        # Test Fortran-order array
        print("\nTest: Fortran-order array")
        fortran_order = np.asfortranarray(base_data)
        result = self._run_test(
            "fortran_order", 
            fortran_order, 
            "Testing Fortran-order array"
        )
        format_tests.append(result)
        
        return format_tests
    
    def validate_performance_stress(self) -> List[Dict]:
        """Test reconstruction under performance stress conditions."""
        print("\n=== Performance Stress Validation ===")
        stress_tests = []
        
        # Test 1: Rapid successive calls
        print("\nTest 1: Rapid successive processing")
        start_time = time.time()
        rapid_results = []
        
        try:
            for i in range(10):
                features = np.random.randn(20, 50) * 0.5
                result = self.pipeline.process_single(
                    features, 
                    output_name=f"rapid_test_{i}"
                )
                rapid_results.append(result['quality_score'])
            
            end_time = time.time()
            avg_quality = np.mean(rapid_results)
            total_time = end_time - start_time
            
            stress_test = {
                'test_name': 'rapid_processing',
                'status': 'SUCCESS',
                'description': 'Testing rapid successive processing',
                'metrics': {
                    'total_time': total_time,
                    'avg_time_per_call': total_time / 10,
                    'avg_quality_score': avg_quality,
                    'quality_std': np.std(rapid_results)
                }
            }
            print(f"  ✓ Success: {total_time:.2f}s total, avg quality {avg_quality:.3f}")
            
        except Exception as e:
            stress_test = {
                'test_name': 'rapid_processing',
                'status': 'ERROR',
                'error': str(e),
                'description': 'Testing rapid successive processing'
            }
            print(f"  ✗ Error: {e}")
        
        stress_tests.append(stress_test)
        
        # Test 2: Memory pressure
        print("\nTest 2: Memory pressure test")
        try:
            # Create multiple large feature matrices
            large_matrices = []
            for i in range(5):
                matrix = np.random.randn(200, 200) * 0.5
                large_matrices.append(matrix)
            
            # Process them while keeping references (memory pressure)
            memory_results = []
            for i, matrix in enumerate(large_matrices):
                result = self.pipeline.process_single(
                    matrix, 
                    output_name=f"memory_test_{i}"
                )
                memory_results.append(result['quality_score'])
            
            stress_test = {
                'test_name': 'memory_pressure',
                'status': 'SUCCESS',
                'description': 'Testing under memory pressure',
                'metrics': {
                    'matrices_processed': len(memory_results),
                    'avg_quality_score': np.mean(memory_results)
                }
            }
            print(f"  ✓ Success: Processed {len(memory_results)} large matrices")
            
            # Clean up
            del large_matrices
            gc.collect()
            
        except Exception as e:
            stress_test = {
                'test_name': 'memory_pressure',
                'status': 'ERROR',
                'error': str(e),
                'description': 'Testing under memory pressure'
            }
            print(f"  ✗ Error: {e}")
        
        stress_tests.append(stress_test)
        
        return stress_tests
    
    def validate_feature_type_robustness(self) -> List[Dict]:
        """Test reconstruction with different feature types and edge cases."""
        print("\n=== Feature Type Robustness Validation ===")
        feature_tests = []
        
        # Generate base audio for feature extraction
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        feature_types = ['raw', 'spectral', 'mfcc']
        
        for feature_type in feature_types:
            print(f"\nTesting {feature_type} features with edge cases")
            
            try:
                # Extract features
                if feature_type == 'raw':
                    frames = librosa.util.frame(test_audio, frame_length=512, hop_length=256, axis=0)
                    features = frames.T
                elif feature_type == 'spectral':
                    stft = librosa.stft(test_audio, n_fft=512, hop_length=256)
                    features = np.abs(stft).T
                elif feature_type == 'mfcc':
                    mfcc = librosa.feature.mfcc(y=test_audio, sr=sr, n_mfcc=13, hop_length=256)
                    features = mfcc.T
                
                # Test 1: Normal features
                result = self._run_test(
                    f"{feature_type}_normal", 
                    features, 
                    f"Testing normal {feature_type} features",
                    feature_type=feature_type
                )
                feature_tests.append(result)
                
                # Test 2: Scaled features
                scaled_features = features * 10
                result = self._run_test(
                    f"{feature_type}_scaled", 
                    scaled_features, 
                    f"Testing scaled {feature_type} features",
                    feature_type=feature_type
                )
                feature_tests.append(result)
                
                # Test 3: Normalized features
                normalized_features = (features - np.mean(features)) / (np.std(features) + 1e-8)
                result = self._run_test(
                    f"{feature_type}_normalized", 
                    normalized_features, 
                    f"Testing normalized {feature_type} features",
                    feature_type=feature_type
                )
                feature_tests.append(result)
                
            except Exception as e:
                error_result = {
                    'test_name': f'{feature_type}_extraction_error',
                    'status': 'ERROR',
                    'error': str(e),
                    'description': f'Error extracting {feature_type} features'
                }
                feature_tests.append(error_result)
                print(f"  ✗ Feature extraction error: {e}")
        
        return feature_tests
    
    def _run_test(self, test_name: str, features: np.ndarray, description: str, 
                  feature_type: str = 'raw', original_audio: Optional[np.ndarray] = None) -> Dict:
        """Run a single test and return results."""
        try:
            start_time = time.time()
            
            result = self.pipeline.process_single(
                features,
                original_audio=original_audio,
                output_name=test_name,
                feature_type=feature_type
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            test_result = {
                'test_name': test_name,
                'status': 'SUCCESS',
                'description': description,
                'feature_shape': list(features.shape),
                'feature_type': feature_type,
                'processing_time': processing_time,
                'quality_score': result.get('quality_score', 0),
                'output_duration': result.get('output_duration', 0),
                'metrics': result.get('metrics', {})
            }
            
            print(f"  ✓ Success: Quality {test_result['quality_score']:.3f}, Time {processing_time:.3f}s")
            
        except Exception as e:
            test_result = {
                'test_name': test_name,
                'status': 'ERROR',
                'description': description,
                'feature_shape': list(features.shape),
                'feature_type': feature_type,
                'error': str(e)
            }
            print(f"  ✗ Error: {e}")
        
        return test_result
    
    def run_comprehensive_validation(self) -> Dict:
        """Run all edge case validations and return comprehensive results."""
        print("=== Comprehensive Edge Case Validation ===")
        print("Testing reconstruction pipeline robustness...\n")
        
        all_results = {
            'validation_timestamp': np.datetime64('now').astype(str),
            'test_categories': {}
        }
        
        # Run all validation categories
        validation_categories = [
            ('data_corruption', self.validate_data_corruption),
            ('boundary_conditions', self.validate_boundary_conditions),
            ('data_types_formats', self.validate_data_types_and_formats),
            ('performance_stress', self.validate_performance_stress),
            ('feature_type_robustness', self.validate_feature_type_robustness)
        ]
        
        total_tests = 0
        total_passed = 0
        total_errors = 0
        
        for category_name, validation_func in validation_categories:
            try:
                category_results = validation_func()
                all_results['test_categories'][category_name] = category_results
                
                # Count results
                category_total = len(category_results)
                category_passed = len([r for r in category_results if r['status'] == 'SUCCESS'])
                category_errors = len([r for r in category_results if r['status'] == 'ERROR'])
                
                total_tests += category_total
                total_passed += category_passed
                total_errors += category_errors
                
                print(f"\n{category_name.replace('_', ' ').title()}: {category_passed}/{category_total} passed")
                
            except Exception as e:
                print(f"\n✗ Category {category_name} failed: {e}")
                all_results['test_categories'][category_name] = {
                    'category_error': str(e)
                }
        
        # Generate summary
        all_results['summary'] = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_errors': total_errors,
            'success_rate': total_passed / total_tests if total_tests > 0 else 0,
            'error_rate': total_errors / total_tests if total_tests > 0 else 0
        }
        
        return all_results


def save_validation_report(results: Dict, output_path: str = "test_results/edge_case_validation_report.json"):
    """Save comprehensive validation report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\n✓ Validation report saved to: {output_path}")


def main():
    """Main validation function."""
    print("=== Edge Case Validation for Reconstruction Pipeline ===")
    
    # Initialize pipeline with conservative settings for testing
    pipeline = ReconstructionPipeline(
        output_dir="test_outputs/edge_case_validation",
        quality_threshold=0.0  # Accept all outputs for testing
    )
    
    # Create validator
    validator = EdgeCaseValidator(pipeline)
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Save results
    save_validation_report(results)
    
    # Print final summary
    summary = results['summary']
    print("\n=== Final Validation Summary ===")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['total_passed']}")
    print(f"Errors: {summary['total_errors']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Error Rate: {summary['error_rate']:.1%}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if summary['error_rate'] > 0.2:
        print("- High error rate detected. Review error handling and input validation.")
    if summary['success_rate'] < 0.8:
        print("- Low success rate. Consider improving robustness of reconstruction algorithms.")
    if summary['success_rate'] >= 0.9:
        print("- Excellent robustness! Pipeline handles edge cases well.")
    
    print("\n✓ Edge case validation completed!")
    return results


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    try:
        results = main()
    except Exception as e:
        print(f"\n✗ Validation execution failed: {e}")
        raise