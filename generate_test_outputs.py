"""Generate Test Outputs and Audio Samples

This script generates comprehensive test outputs including reconstructed audio samples
with various configurations, quality assessments, and comparative analysis.

Author: IBM Internship Project
Date: Sprint 6 - Signal Reconstruction & Evaluation
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple, Optional
import warnings
from datetime import datetime

# Import our utilities
from utils.reconstruction_pipeline import ReconstructionPipeline
from utils.reconstruction import reconstruct_audio_overlap_add
from utils.evaluation import AudioEvaluator


class TestOutputGenerator:
    """Generate comprehensive test outputs and audio samples."""
    
    def __init__(self, output_base_dir: str = "test_outputs/generated_samples"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline and evaluator
        self.pipeline = ReconstructionPipeline(
            output_dir=str(self.output_base_dir / "reconstructions"),
            quality_threshold=0.0  # Save all outputs for analysis
        )
        self.evaluator = AudioEvaluator()
        
        # Results storage
        self.generation_results = []
        self.audio_samples = []
        
    def generate_synthetic_audio_library(self) -> List[Tuple[np.ndarray, str, Dict]]:
        """Generate a comprehensive library of synthetic audio samples."""
        print("=== Generating Synthetic Audio Library ===")
        
        audio_library = []
        sr = 22050
        duration = 2.0  # 2 seconds per sample
        t = np.linspace(0, duration, int(sr * duration))
        
        # 1. Pure tones at different frequencies
        frequencies = [220, 440, 880, 1760]  # A notes across octaves
        for freq in frequencies:
            audio = np.sin(2 * np.pi * freq * t) * 0.7
            metadata = {
                'type': 'pure_tone',
                'frequency': freq,
                'duration': duration,
                'sample_rate': sr
            }
            audio_library.append((audio, f"pure_tone_{freq}hz", metadata))
            print(f"  ✓ Generated pure tone: {freq} Hz")
        
        # 2. Complex harmonic tones
        for freq in [220, 440]:
            # Fundamental + harmonics
            audio = (np.sin(2 * np.pi * freq * t) * 0.5 +
                    np.sin(2 * np.pi * freq * 2 * t) * 0.3 +
                    np.sin(2 * np.pi * freq * 3 * t) * 0.2)
            audio *= 0.7
            metadata = {
                'type': 'harmonic_tone',
                'fundamental': freq,
                'harmonics': [2, 3],
                'duration': duration,
                'sample_rate': sr
            }
            audio_library.append((audio, f"harmonic_tone_{freq}hz", metadata))
            print(f"  ✓ Generated harmonic tone: {freq} Hz")
        
        # 3. Chirp signals (frequency sweeps)
        chirp_configs = [
            (100, 1000, 'linear'),
            (1000, 100, 'linear'),
            (200, 2000, 'exponential')
        ]
        
        for f0, f1, method in chirp_configs:
            if method == 'linear':
                freq_t = f0 + (f1 - f0) * t / duration
                audio = np.sin(2 * np.pi * np.cumsum(freq_t) / sr) * 0.7
            else:  # exponential
                freq_t = f0 * (f1/f0) ** (t/duration)
                audio = np.sin(2 * np.pi * np.cumsum(freq_t) / sr) * 0.7
            
            metadata = {
                'type': 'chirp',
                'f0': f0,
                'f1': f1,
                'method': method,
                'duration': duration,
                'sample_rate': sr
            }
            audio_library.append((audio, f"chirp_{method}_{f0}to{f1}hz", metadata))
            print(f"  ✓ Generated {method} chirp: {f0}-{f1} Hz")
        
        # 4. Noise signals
        noise_types = [
            ('white', np.random.normal(0, 0.3, len(t))),
            ('pink', self._generate_pink_noise(len(t)) * 0.3),
            ('brown', self._generate_brown_noise(len(t)) * 0.3)
        ]
        
        for noise_name, noise_audio in noise_types:
            metadata = {
                'type': 'noise',
                'noise_type': noise_name,
                'duration': duration,
                'sample_rate': sr
            }
            audio_library.append((noise_audio, f"noise_{noise_name}", metadata))
            print(f"  ✓ Generated {noise_name} noise")
        
        # 5. Amplitude modulated signals
        carrier_freq = 440
        mod_freq = 5
        audio = np.sin(2 * np.pi * carrier_freq * t) * (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)) * 0.5
        metadata = {
            'type': 'amplitude_modulated',
            'carrier_freq': carrier_freq,
            'modulation_freq': mod_freq,
            'duration': duration,
            'sample_rate': sr
        }
        audio_library.append((audio, "am_signal_440hz_5hz_mod", metadata))
        print(f"  ✓ Generated AM signal: {carrier_freq} Hz carrier, {mod_freq} Hz modulation")
        
        # 6. Frequency modulated signals
        carrier_freq = 440
        mod_freq = 10
        mod_depth = 50
        audio = np.sin(2 * np.pi * carrier_freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t)) * 0.7
        metadata = {
            'type': 'frequency_modulated',
            'carrier_freq': carrier_freq,
            'modulation_freq': mod_freq,
            'modulation_depth': mod_depth,
            'duration': duration,
            'sample_rate': sr
        }
        audio_library.append((audio, "fm_signal_440hz_10hz_mod", metadata))
        print(f"  ✓ Generated FM signal: {carrier_freq} Hz carrier, {mod_freq} Hz modulation")
        
        # 7. Impulse and step responses
        # Impulse
        impulse = np.zeros(len(t))
        impulse[len(t)//4] = 1.0  # Impulse at 1/4 duration
        metadata = {
            'type': 'impulse',
            'position': 0.25,
            'duration': duration,
            'sample_rate': sr
        }
        audio_library.append((impulse, "impulse_response", metadata))
        print("  ✓ Generated impulse response")
        
        # Step
        step = np.zeros(len(t))
        step[len(t)//2:] = 0.5  # Step at half duration
        metadata = {
            'type': 'step',
            'position': 0.5,
            'amplitude': 0.5,
            'duration': duration,
            'sample_rate': sr
        }
        audio_library.append((step, "step_response", metadata))
        print("  ✓ Generated step response")
        
        print(f"\n✓ Generated {len(audio_library)} synthetic audio samples")
        return audio_library
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f noise)."""
        # Generate white noise
        white = np.random.normal(0, 1, length)
        
        # Apply 1/f filter in frequency domain
        fft_white = np.fft.fft(white)
        freqs = np.fft.fftfreq(length)
        
        # Avoid division by zero
        freqs[0] = 1e-10
        
        # Apply 1/sqrt(f) filter for pink noise
        pink_fft = fft_white / np.sqrt(np.abs(freqs))
        pink_fft[0] = 0  # Remove DC component
        
        pink = np.real(np.fft.ifft(pink_fft))
        return pink / np.std(pink)  # Normalize
    
    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown noise (1/f^2 noise)."""
        # Generate white noise
        white = np.random.normal(0, 1, length)
        
        # Apply 1/f^2 filter in frequency domain
        fft_white = np.fft.fft(white)
        freqs = np.fft.fftfreq(length)
        
        # Avoid division by zero
        freqs[0] = 1e-10
        
        # Apply 1/f filter for brown noise
        brown_fft = fft_white / np.abs(freqs)
        brown_fft[0] = 0  # Remove DC component
        
        brown = np.real(np.fft.ifft(brown_fft))
        return brown / np.std(brown)  # Normalize
    
    def extract_features_from_audio(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, np.ndarray]:
        """Extract different types of features from audio."""
        features = {}
        
        # Raw features (framed audio)
        frame_length = 512
        hop_length = 256
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length, axis=0)
        features['raw'] = frames.T
        
        # Spectral features (STFT magnitude)
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        features['spectral'] = np.abs(stft).T
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
        features['mfcc'] = mfcc.T
        
        # Mel-spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length)
        features['mel'] = mel_spec.T
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
        features['chroma'] = chroma.T
        
        return features
    
    def generate_reconstruction_samples(self, audio_library: List[Tuple[np.ndarray, str, Dict]]) -> List[Dict]:
        """Generate reconstruction samples for all audio in library."""
        print("\n=== Generating Reconstruction Samples ===")
        
        reconstruction_results = []
        feature_types = ['raw', 'spectral', 'mfcc']  # Focus on main types
        
        for audio, name, metadata in audio_library:
            print(f"\nProcessing: {name}")
            
            # Extract features
            try:
                features_dict = self.extract_features_from_audio(audio, metadata['sample_rate'])
                
                # Save original audio
                original_path = self.output_base_dir / "originals" / f"{name}_original.wav"
                original_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(original_path), audio, metadata['sample_rate'])
                
                sample_results = {
                    'sample_name': name,
                    'original_metadata': metadata,
                    'original_path': str(original_path),
                    'reconstructions': {}
                }
                
                # Test reconstruction with different feature types
                for feature_type in feature_types:
                    if feature_type in features_dict:
                        features = features_dict[feature_type]
                        
                        print(f"  Testing {feature_type} features ({features.shape})")
                        
                        try:
                            # Reconstruct using pipeline
                            result = self.pipeline.process_single(
                                features,
                                original_audio=audio,
                                output_name=f"{name}_{feature_type}",
                                feature_type=feature_type
                            )
                            
                            # Load reconstructed audio for additional analysis
                            reconstructed_audio = result.get('reconstructed_audio')
                            if reconstructed_audio is not None:
                                # Save reconstructed audio
                                recon_path = self.output_base_dir / "reconstructions" / f"{name}_{feature_type}_reconstructed.wav"
                                recon_path.parent.mkdir(parents=True, exist_ok=True)
                                sf.write(str(recon_path), reconstructed_audio, metadata['sample_rate'])
                                
                                # Additional evaluation
                                detailed_metrics = self.evaluator.evaluate_reconstruction(
                                    audio, reconstructed_audio, metadata['sample_rate']
                                )
                                
                                sample_results['reconstructions'][feature_type] = {
                                    'reconstruction_path': str(recon_path),
                                    'feature_shape': list(features.shape),
                                    'pipeline_result': result,
                                    'detailed_metrics': detailed_metrics
                                }
                                
                                print(f"    ✓ {feature_type}: Quality {result.get('quality_score', 0):.3f}, SNR {detailed_metrics.get('snr_db', 0):.1f} dB")
                            else:
                                print(f"    ✗ {feature_type}: No reconstructed audio returned")
                                
                        except Exception as e:
                            print(f"    ✗ {feature_type}: Error - {e}")
                            sample_results['reconstructions'][feature_type] = {
                                'error': str(e),
                                'feature_shape': list(features.shape)
                            }
                
                reconstruction_results.append(sample_results)
                
            except Exception as e:
                print(f"  ✗ Feature extraction failed: {e}")
                reconstruction_results.append({
                    'sample_name': name,
                    'original_metadata': metadata,
                    'error': f"Feature extraction failed: {e}"
                })
        
        return reconstruction_results
    
    def create_comparison_plots(self, reconstruction_results: List[Dict]):
        """Create comparison plots for reconstruction quality."""
        print("\n=== Creating Comparison Plots ===")
        
        plots_dir = self.output_base_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data for plotting
        feature_types = ['raw', 'spectral', 'mfcc']
        quality_data = {ft: [] for ft in feature_types}
        snr_data = {ft: [] for ft in feature_types}
        sample_names = []
        
        for result in reconstruction_results:
            if 'reconstructions' in result:
                sample_names.append(result['sample_name'])
                
                for ft in feature_types:
                    if ft in result['reconstructions'] and 'pipeline_result' in result['reconstructions'][ft]:
                        quality = result['reconstructions'][ft]['pipeline_result'].get('quality_score', 0)
                        snr = result['reconstructions'][ft]['detailed_metrics'].get('snr_db', -100)
                        quality_data[ft].append(quality)
                        snr_data[ft].append(snr)
                    else:
                        quality_data[ft].append(0)
                        snr_data[ft].append(-100)
        
        if not sample_names:
            print("  ⚠ No data available for plotting")
            return
        
        # Plot 1: Quality Score Comparison
        plt.figure(figsize=(15, 8))
        x = np.arange(len(sample_names))
        width = 0.25
        
        for i, ft in enumerate(feature_types):
            plt.bar(x + i * width, quality_data[ft], width, label=f'{ft.title()} Features', alpha=0.8)
        
        plt.xlabel('Audio Samples')
        plt.ylabel('Quality Score')
        plt.title('Reconstruction Quality Score by Feature Type')
        plt.xticks(x + width, sample_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Quality comparison plot saved")
        
        # Plot 2: SNR Comparison
        plt.figure(figsize=(15, 8))
        
        for i, ft in enumerate(feature_types):
            plt.bar(x + i * width, snr_data[ft], width, label=f'{ft.title()} Features', alpha=0.8)
        
        plt.xlabel('Audio Samples')
        plt.ylabel('SNR (dB)')
        plt.title('Signal-to-Noise Ratio by Feature Type')
        plt.xticks(x + width, sample_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'snr_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ SNR comparison plot saved")
        
        # Plot 3: Feature Type Performance Summary
        plt.figure(figsize=(12, 6))
        
        avg_quality = [np.mean(quality_data[ft]) for ft in feature_types]
        avg_snr = [np.mean(snr_data[ft]) for ft in feature_types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Average quality
        bars1 = ax1.bar(feature_types, avg_quality, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon'])
        ax1.set_ylabel('Average Quality Score')
        ax1.set_title('Average Quality by Feature Type')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, avg_quality):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Average SNR
        bars2 = ax2.bar(feature_types, avg_snr, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon'])
        ax2.set_ylabel('Average SNR (dB)')
        ax2.set_title('Average SNR by Feature Type')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars2, avg_snr):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_type_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Feature type summary plot saved")
    
    def generate_comprehensive_report(self, reconstruction_results: List[Dict]) -> Dict:
        """Generate comprehensive analysis report."""
        print("\n=== Generating Comprehensive Report ===")
        
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_samples': len(reconstruction_results),
                'successful_samples': 0,
                'failed_samples': 0,
                'feature_types_tested': ['raw', 'spectral', 'mfcc']
            },
            'performance_analysis': {},
            'detailed_results': reconstruction_results
        }
        
        # Analyze performance by feature type
        feature_types = ['raw', 'spectral', 'mfcc']
        performance = {}
        
        for ft in feature_types:
            quality_scores = []
            snr_values = []
            success_count = 0
            
            for result in reconstruction_results:
                if 'reconstructions' in result and ft in result['reconstructions']:
                    recon = result['reconstructions'][ft]
                    if 'pipeline_result' in recon:
                        quality_scores.append(recon['pipeline_result'].get('quality_score', 0))
                        snr_values.append(recon['detailed_metrics'].get('snr_db', -100))
                        success_count += 1
            
            if quality_scores:
                performance[ft] = {
                    'success_count': success_count,
                    'success_rate': success_count / len(reconstruction_results),
                    'avg_quality_score': np.mean(quality_scores),
                    'std_quality_score': np.std(quality_scores),
                    'avg_snr_db': np.mean(snr_values),
                    'std_snr_db': np.std(snr_values),
                    'min_quality': np.min(quality_scores),
                    'max_quality': np.max(quality_scores),
                    'min_snr': np.min(snr_values),
                    'max_snr': np.max(snr_values)
                }
            else:
                performance[ft] = {
                    'success_count': 0,
                    'success_rate': 0,
                    'error': 'No successful reconstructions'
                }
        
        report['performance_analysis'] = performance
        
        # Count successful samples
        for result in reconstruction_results:
            if 'reconstructions' in result and any('pipeline_result' in result['reconstructions'].get(ft, {}) 
                                                   for ft in feature_types):
                report['summary']['successful_samples'] += 1
            else:
                report['summary']['failed_samples'] += 1
        
        # Best and worst performing samples
        best_samples = {}
        worst_samples = {}
        
        for ft in feature_types:
            if ft in performance and 'avg_quality_score' in performance[ft]:
                ft_results = [(result['sample_name'], 
                              result['reconstructions'][ft]['pipeline_result'].get('quality_score', 0))
                             for result in reconstruction_results 
                             if 'reconstructions' in result and ft in result['reconstructions'] 
                             and 'pipeline_result' in result['reconstructions'][ft]]
                
                if ft_results:
                    ft_results.sort(key=lambda x: x[1], reverse=True)
                    best_samples[ft] = ft_results[0]
                    worst_samples[ft] = ft_results[-1]
        
        report['best_samples'] = best_samples
        report['worst_samples'] = worst_samples
        
        return report
    
    def save_comprehensive_report(self, report: Dict):
        """Save comprehensive report to JSON file."""
        report_path = self.output_base_dir / "comprehensive_test_report.json"
        
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
        
        with open(report_path, 'w') as f:
            json.dump(convert_numpy(report), f, indent=2)
        
        print(f"✓ Comprehensive report saved to: {report_path}")
        return report_path
    
    def run_complete_generation(self) -> Dict:
        """Run complete test output generation process."""
        print("=== Complete Test Output Generation ===")
        print("Generating comprehensive test outputs and audio samples...\n")
        
        # Step 1: Generate synthetic audio library
        audio_library = self.generate_synthetic_audio_library()
        
        # Step 2: Generate reconstruction samples
        reconstruction_results = self.generate_reconstruction_samples(audio_library)
        
        # Step 3: Create comparison plots
        self.create_comparison_plots(reconstruction_results)
        
        # Step 4: Generate comprehensive report
        report = self.generate_comprehensive_report(reconstruction_results)
        
        # Step 5: Save report
        report_path = self.save_comprehensive_report(report)
        
        # Print summary
        print("\n=== Generation Summary ===")
        print(f"Total samples generated: {report['summary']['total_samples']}")
        print(f"Successful reconstructions: {report['summary']['successful_samples']}")
        print(f"Failed reconstructions: {report['summary']['failed_samples']}")
        print(f"Success rate: {report['summary']['successful_samples']/report['summary']['total_samples']:.1%}")
        
        print("\n=== Performance by Feature Type ===")
        for ft, perf in report['performance_analysis'].items():
            if 'avg_quality_score' in perf:
                print(f"{ft.title()}: Avg Quality {perf['avg_quality_score']:.3f}, Avg SNR {perf['avg_snr_db']:.1f} dB")
            else:
                print(f"{ft.title()}: {perf.get('error', 'No data')}")
        
        print(f"\n[OK] All outputs saved to: {self.output_base_dir}")
        print(f"[OK] Report saved to: {report_path}")
        
        return report


def main():
    """Main generation function."""
    print("=== Test Output and Audio Sample Generation ===")
    
    # Create generator
    generator = TestOutputGenerator()
    
    # Run complete generation
    report = generator.run_complete_generation()
    
    print("\n✓ Test output generation completed successfully!")
    return report


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    try:
        results = main()
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        raise