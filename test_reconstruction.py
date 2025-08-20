"""Test Script for Reconstruction Pipeline

This script validates the reconstruction pipeline with sample model outputs
and tests audio quality across different scenarios.

Author: IBM Internship Project
Date: Sprint 6 - Signal Reconstruction & Evaluation
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from typing import List, Dict, Tuple
import json

# Import our utilities
from utils.reconstruction_pipeline import ReconstructionPipeline
from utils.reconstruction import reconstruct_audio_overlap_add
from utils.evaluation import AudioEvaluator


def generate_test_audio_samples(sr: int = 44100, duration: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Generate various test audio samples for validation.
    
    Parameters:
    -----------
    sr : int
        Sample rate
    duration : float
        Duration in seconds
    
    Returns:
    --------
    Dict of test audio samples
    """
    t = np.linspace(0, duration, int(sr * duration))
    samples = {}
    
    # Pure sine wave
    samples['sine_440hz'] = np.sin(2 * np.pi * 440 * t) * 0.7
    
    # Chirp signal (frequency sweep)
    samples['chirp'] = librosa.chirp(fmin=200, fmax=2000, sr=sr, duration=duration) * 0.5
    
    # Multi-tone signal
    freqs = [220, 440, 880, 1760]  # A notes across octaves
    multi_tone = np.zeros_like(t)
    for freq in freqs:
        multi_tone += np.sin(2 * np.pi * freq * t) * 0.2
    samples['multi_tone'] = multi_tone
    
    # Noisy signal
    clean_signal = np.sin(2 * np.pi * 440 * t) * 0.5
    noise = np.random.normal(0, 0.1, len(t))
    samples['noisy_sine'] = clean_signal + noise
    
    # Speech-like formant structure
    formants = [800, 1200, 2400]  # Typical vowel formants
    speech_like = np.zeros_like(t)
    for formant in formants:
        speech_like += np.sin(2 * np.pi * formant * t) * np.exp(-t * 0.5) * 0.3
    samples['speech_like'] = speech_like
    
    # Impulse train (for testing transient response)
    impulse_train = np.zeros_like(t)
    impulse_positions = np.arange(0, len(t), sr // 10)  # Every 0.1 seconds
    impulse_train[impulse_positions] = 1.0
    samples['impulse_train'] = impulse_train * 0.8
    
    return samples


def generate_test_features(audio_samples: Dict[str, np.ndarray], 
                          sr: int = 44100) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate different types of features from audio samples.
    
    Parameters:
    -----------
    audio_samples : Dict[str, np.ndarray]
        Dictionary of audio samples
    sr : int
        Sample rate
    
    Returns:
    --------
    Dict of feature matrices for each sample and feature type
    """
    features = {}
    
    for sample_name, audio in audio_samples.items():
        features[sample_name] = {}
        
        # Raw features (windowed frames)
        frame_length = 1024
        hop_length = 512
        frames = librosa.util.frame(audio, frame_length=frame_length, 
                                   hop_length=hop_length, axis=0)
        features[sample_name]['raw'] = frames.T  # (n_frames, frame_length)
        
        # Spectral features (STFT magnitude)
        stft = librosa.stft(audio, n_fft=1024, hop_length=hop_length)
        magnitude = np.abs(stft)
        features[sample_name]['spectral'] = magnitude.T  # (n_frames, n_freq_bins)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, 
                                   hop_length=hop_length)
        features[sample_name]['mfcc'] = mfcc.T  # (n_frames, n_mfcc)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, 
                                                 hop_length=hop_length)
        features[sample_name]['mel'] = mel_spec.T  # (n_frames, n_mel_bins)
    
    return features


def test_reconstruction_quality(pipeline: ReconstructionPipeline,
                              audio_samples: Dict[str, np.ndarray],
                              feature_sets: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """
    Test reconstruction quality across different audio types and features.
    
    Parameters:
    -----------
    pipeline : ReconstructionPipeline
        Reconstruction pipeline instance
    audio_samples : Dict[str, np.ndarray]
        Original audio samples
    feature_sets : Dict[str, Dict[str, np.ndarray]]
        Feature matrices for each sample
    
    Returns:
    --------
    Dict containing test results
    """
    print("\n=== Testing Reconstruction Quality ===")
    
    results = {
        'test_results': [],
        'summary_stats': {},
        'quality_analysis': {}
    }
    
    # Test each combination of sample and feature type
    for sample_name, original_audio in audio_samples.items():
        print(f"\nTesting sample: {sample_name}")
        
        for feature_type, features in feature_sets[sample_name].items():
            print(f"  Feature type: {feature_type} - Shape: {features.shape}")
            
            try:
                # Process with pipeline
                result = pipeline.process_single(
                    features,
                    original_audio=original_audio,
                    output_name=f"{sample_name}_{feature_type}",
                    feature_type=feature_type
                )
                
                # Extract key metrics
                test_result = {
                    'sample_name': sample_name,
                    'feature_type': feature_type,
                    'feature_shape': features.shape,
                    'quality_score': result.get('quality_score', 0),
                    'snr_db': result['metrics'].get('snr_db', float('nan')),
                    'correlation': result['metrics'].get('correlation', float('nan')),
                    'mse': result['metrics'].get('mse', float('nan')),
                    'output_duration': result.get('output_duration', 0),
                    'status': 'SUCCESS'
                }
                
                results['test_results'].append(test_result)
                
                print(f"    ✓ Quality Score: {test_result['quality_score']:.3f}")
                print(f"    ✓ SNR: {test_result['snr_db']:.2f} dB")
                print(f"    ✓ Correlation: {test_result['correlation']:.3f}")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                test_result = {
                    'sample_name': sample_name,
                    'feature_type': feature_type,
                    'feature_shape': features.shape,
                    'error': str(e),
                    'status': 'FAILED'
                }
                results['test_results'].append(test_result)
    
    # Analyze results
    successful_tests = [r for r in results['test_results'] if r['status'] == 'SUCCESS']
    failed_tests = [r for r in results['test_results'] if r['status'] == 'FAILED']
    
    if successful_tests:
        quality_scores = [r['quality_score'] for r in successful_tests]
        snr_values = [r['snr_db'] for r in successful_tests if not np.isnan(r['snr_db'])]
        correlations = [r['correlation'] for r in successful_tests if not np.isnan(r['correlation'])]
        
        results['summary_stats'] = {
            'total_tests': len(results['test_results']),
            'successful': len(successful_tests),
            'failed': len(failed_tests),
            'success_rate': len(successful_tests) / len(results['test_results']),
            'avg_quality_score': np.mean(quality_scores),
            'std_quality_score': np.std(quality_scores),
            'avg_snr_db': np.mean(snr_values) if snr_values else float('nan'),
            'avg_correlation': np.mean(correlations) if correlations else float('nan')
        }
        
        # Quality analysis by feature type
        feature_types = list(set(r['feature_type'] for r in successful_tests))
        results['quality_analysis'] = {}
        
        for ft in feature_types:
            ft_results = [r for r in successful_tests if r['feature_type'] == ft]
            if ft_results:
                ft_quality = [r['quality_score'] for r in ft_results]
                results['quality_analysis'][ft] = {
                    'count': len(ft_results),
                    'avg_quality': np.mean(ft_quality),
                    'std_quality': np.std(ft_quality),
                    'best_sample': max(ft_results, key=lambda x: x['quality_score'])['sample_name']
                }
    
    return results


def test_edge_cases(pipeline: ReconstructionPipeline) -> Dict:
    """
    Test reconstruction pipeline with edge cases.
    
    Parameters:
    -----------
    pipeline : ReconstructionPipeline
        Reconstruction pipeline instance
    
    Returns:
    --------
    Dict containing edge case test results
    """
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = []
    
    # Test 1: Empty feature matrix
    print("\nTest 1: Empty feature matrix")
    try:
        empty_features = np.array([]).reshape(0, 10)
        result = pipeline.process_single(empty_features, output_name="empty_test")
        edge_cases.append({'test': 'empty_matrix', 'status': 'UNEXPECTED_SUCCESS', 'result': result})
        print("  ✗ Unexpected success with empty matrix")
    except Exception as e:
        edge_cases.append({'test': 'empty_matrix', 'status': 'EXPECTED_FAILURE', 'error': str(e)})
        print(f"  ✓ Expected failure: {e}")
    
    # Test 2: Single frame
    print("\nTest 2: Single frame feature matrix")
    try:
        single_frame = np.random.randn(1, 100)
        result = pipeline.process_single(single_frame, output_name="single_frame_test")
        edge_cases.append({'test': 'single_frame', 'status': 'SUCCESS', 'result': result})
        print(f"  ✓ Success: Quality score {result['quality_score']:.3f}")
    except Exception as e:
        edge_cases.append({'test': 'single_frame', 'status': 'FAILURE', 'error': str(e)})
        print(f"  ✗ Failed: {e}")
    
    # Test 3: Very large feature matrix
    print("\nTest 3: Large feature matrix")
    try:
        large_features = np.random.randn(1000, 512) * 0.1
        result = pipeline.process_single(large_features, output_name="large_test")
        edge_cases.append({'test': 'large_matrix', 'status': 'SUCCESS', 'result': result})
        print(f"  ✓ Success: Duration {result['output_duration']:.2f}s")
    except Exception as e:
        edge_cases.append({'test': 'large_matrix', 'status': 'FAILURE', 'error': str(e)})
        print(f"  ✗ Failed: {e}")
    
    # Test 4: Extreme values
    print("\nTest 4: Extreme feature values")
    try:
        extreme_features = np.random.randn(50, 100) * 100  # Very large values
        result = pipeline.process_single(extreme_features, output_name="extreme_test")
        edge_cases.append({'test': 'extreme_values', 'status': 'SUCCESS', 'result': result})
        print(f"  ✓ Success: Quality score {result['quality_score']:.3f}")
    except Exception as e:
        edge_cases.append({'test': 'extreme_values', 'status': 'FAILURE', 'error': str(e)})
        print(f"  ✗ Failed: {e}")
    
    # Test 5: NaN/Inf values
    print("\nTest 5: NaN/Inf feature values")
    try:
        nan_features = np.random.randn(50, 100)
        nan_features[10:15, :] = np.nan
        nan_features[20:25, :] = np.inf
        result = pipeline.process_single(nan_features, output_name="nan_test")
        edge_cases.append({'test': 'nan_inf_values', 'status': 'UNEXPECTED_SUCCESS', 'result': result})
        print("  ✗ Unexpected success with NaN/Inf values")
    except Exception as e:
        edge_cases.append({'test': 'nan_inf_values', 'status': 'EXPECTED_FAILURE', 'error': str(e)})
        print(f"  ✓ Expected failure: {e}")
    
    # Test 6: Mismatched dimensions
    print("\nTest 6: Unusual feature dimensions")
    try:
        unusual_features = np.random.randn(10, 1)  # Very narrow features
        result = pipeline.process_single(unusual_features, output_name="narrow_test")
        edge_cases.append({'test': 'narrow_features', 'status': 'SUCCESS', 'result': result})
        print(f"  ✓ Success: Quality score {result['quality_score']:.3f}")
    except Exception as e:
        edge_cases.append({'test': 'narrow_features', 'status': 'FAILURE', 'error': str(e)})
        print(f"  ✗ Failed: {e}")
    
    return {'edge_case_results': edge_cases}


def generate_test_report(quality_results: Dict, edge_case_results: Dict, 
                        output_path: str = "test_results/reconstruction_test_report.json"):
    """
    Generate comprehensive test report.
    
    Parameters:
    -----------
    quality_results : Dict
        Results from quality testing
    edge_case_results : Dict
        Results from edge case testing
    output_path : str
        Path to save the report
    """
    report = {
        'test_timestamp': np.datetime64('now').astype(str),
        'test_summary': {
            'total_quality_tests': len(quality_results.get('test_results', [])),
            'quality_success_rate': quality_results.get('summary_stats', {}).get('success_rate', 0),
            'avg_quality_score': quality_results.get('summary_stats', {}).get('avg_quality_score', 0),
            'edge_cases_tested': len(edge_case_results.get('edge_case_results', [])),
            'edge_cases_passed': len([r for r in edge_case_results.get('edge_case_results', []) 
                                    if r['status'] in ['SUCCESS', 'EXPECTED_FAILURE']])
        },
        'detailed_results': {
            'quality_testing': quality_results,
            'edge_case_testing': edge_case_results
        },
        'recommendations': []
    }
    
    # Add recommendations based on results
    if report['test_summary']['quality_success_rate'] < 0.8:
        report['recommendations'].append(
            "Low success rate detected. Review reconstruction parameters and feature preprocessing."
        )
    
    if report['test_summary']['avg_quality_score'] < 0.5:
        report['recommendations'].append(
            "Low average quality score. Consider improving reconstruction algorithm or evaluation metrics."
        )
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
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
        
        json.dump(convert_numpy(report), f, indent=2)
    
    print(f"\n✓ Test report saved to: {output_path}")
    return report


def main():
    """Main test function."""
    print("=== Reconstruction Pipeline Testing ===")
    print("Testing reconstruction quality and robustness...\n")
    
    # Initialize pipeline
    pipeline = ReconstructionPipeline(
        output_dir="test_outputs/reconstruction_validation",
        quality_threshold=0.3  # Lower threshold for testing
    )
    
    # Generate test data
    print("Generating test audio samples...")
    audio_samples = generate_test_audio_samples()
    print(f"Generated {len(audio_samples)} test audio samples")
    
    print("\nGenerating feature matrices...")
    feature_sets = generate_test_features(audio_samples)
    total_features = sum(len(features) for features in feature_sets.values())
    print(f"Generated {total_features} feature matrices")
    
    # Run quality tests
    quality_results = test_reconstruction_quality(pipeline, audio_samples, feature_sets)
    
    # Run edge case tests
    edge_case_results = test_edge_cases(pipeline)
    
    # Generate report
    report = generate_test_report(quality_results, edge_case_results)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Quality Tests: {report['test_summary']['total_quality_tests']}")
    print(f"Success Rate: {report['test_summary']['quality_success_rate']:.1%}")
    print(f"Average Quality Score: {report['test_summary']['avg_quality_score']:.3f}")
    print(f"Edge Cases Tested: {report['test_summary']['edge_cases_tested']}")
    print(f"Edge Cases Passed: {report['test_summary']['edge_cases_passed']}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
    
    print("\n✓ All tests completed!")
    return report


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    try:
        report = main()
    except Exception as e:
        print(f"\n✗ Test execution failed: {e}")
        raise