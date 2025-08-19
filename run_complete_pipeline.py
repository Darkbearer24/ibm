#!/usr/bin/env python3
"""
Complete Pipeline Runner for Multilingual Speech Translation System

This script demonstrates how to run all notebooks in sequence to execute
the complete project pipeline from data preprocessing to model evaluation.

Usage:
    python run_complete_pipeline.py
    
Or run individual stages:
    python run_complete_pipeline.py --stage 1  # Audio cleaning only
    python run_complete_pipeline.py --stage 2  # Feature extraction only
    etc.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

# Pipeline configuration
PIPELINE_STAGES = {
    1: {
        'name': 'Audio Cleaning & Preprocessing',
        'notebook': '01_audio_cleaning.ipynb',
        'description': 'Load raw audio data, apply denoising, normalization, and save processed audio',
        'inputs': ['data/raw/'],
        'outputs': ['data/processed/']
    },
    2: {
        'name': 'Feature Matrix Generation',
        'notebook': '02_feature_matrix_builder.ipynb', 
        'description': 'Frame audio into 20ms windows and extract 441-dimensional feature vectors',
        'inputs': ['data/processed/'],
        'outputs': ['data/features/']
    },
    3: {
        'name': 'Model Architecture Design',
        'notebook': '03_model_architecture.ipynb',
        'description': 'Build and test encoder-decoder model architecture',
        'inputs': ['models/encoder_decoder.py'],
        'outputs': ['test_checkpoints/test_checkpoint.pt']
    },
    4: {
        'name': 'Training Preparation & Export',
        'notebook': '04_training_preparation_and_gpu_export.ipynb',
        'description': 'Prepare training scripts and export for GPU training',
        'inputs': ['data/features/', 'models/'],
        'outputs': ['outputs/sprint4_training_config.json']
    },
    5: {
        'name': 'CPU Training & Validation',
        'notebook': '05_cpu_training_validation.ipynb',
        'description': 'Train model on CPU and validate performance',
        'inputs': ['data/features/', 'models/train.py'],
        'outputs': ['test_checkpoints/cpu_validation/']
    },
    6: {
        'name': 'Audio Reconstruction & Evaluation',
        'notebook': '06_reconstruction_and_evaluation.ipynb',
        'description': 'Reconstruct audio from model output and evaluate quality',
        'inputs': ['test_checkpoints/cpu_validation/best_model.pt'],
        'outputs': ['outputs/reconstructed_audio/']
    }
}

class PipelineRunner:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.notebooks_dir = self.project_root / 'notebooks'
        self.outputs_dir = self.project_root / 'outputs'
        self.log_file = self.outputs_dir / f'pipeline_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # Ensure output directory exists
        self.outputs_dir.mkdir(exist_ok=True)
        
    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def check_prerequisites(self):
        """Check if all required files and directories exist"""
        self.log("Checking prerequisites...")
        
        # Check if notebooks directory exists
        if not self.notebooks_dir.exists():
            raise FileNotFoundError(f"Notebooks directory not found: {self.notebooks_dir}")
        
        # Check if all notebooks exist
        missing_notebooks = []
        for stage_num, stage_info in PIPELINE_STAGES.items():
            notebook_path = self.notebooks_dir / stage_info['notebook']
            if not notebook_path.exists():
                missing_notebooks.append(stage_info['notebook'])
        
        if missing_notebooks:
            raise FileNotFoundError(f"Missing notebooks: {missing_notebooks}")
        
        # Check if jupyter is available
        try:
            subprocess.run(['jupyter', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Jupyter is not installed or not in PATH")
        
        self.log("✅ All prerequisites satisfied")
    
    def run_notebook(self, notebook_name, stage_num):
        """Execute a single notebook using nbconvert"""
        notebook_path = self.notebooks_dir / notebook_name
        output_notebook = self.outputs_dir / f"executed_{notebook_name}"
        
        self.log(f"🚀 Running Stage {stage_num}: {PIPELINE_STAGES[stage_num]['name']}")
        self.log(f"   Notebook: {notebook_name}")
        self.log(f"   Description: {PIPELINE_STAGES[stage_num]['description']}")
        
        try:
            # Execute notebook and save output
            cmd = [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--output', str(output_notebook),
                str(notebook_path)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                self.log(f"✅ Stage {stage_num} completed successfully")
                return True
            else:
                self.log(f"❌ Stage {stage_num} failed with return code {result.returncode}")
                self.log(f"   Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"⏰ Stage {stage_num} timed out after 30 minutes")
            return False
        except Exception as e:
            self.log(f"❌ Stage {stage_num} failed with exception: {str(e)}")
            return False
    
    def run_pipeline(self, start_stage=1, end_stage=6):
        """Run the complete pipeline or a subset of stages"""
        self.log("="*60)
        self.log("🎯 MULTILINGUAL SPEECH TRANSLATION PIPELINE")
        self.log("="*60)
        
        try:
            self.check_prerequisites()
            
            successful_stages = []
            failed_stages = []
            
            for stage_num in range(start_stage, end_stage + 1):
                if stage_num not in PIPELINE_STAGES:
                    self.log(f"⚠️  Stage {stage_num} not defined, skipping")
                    continue
                
                stage_info = PIPELINE_STAGES[stage_num]
                success = self.run_notebook(stage_info['notebook'], stage_num)
                
                if success:
                    successful_stages.append(stage_num)
                else:
                    failed_stages.append(stage_num)
                    # Ask user if they want to continue
                    response = input(f"\nStage {stage_num} failed. Continue with next stage? (y/n): ")
                    if response.lower() != 'y':
                        break
            
            # Summary
            self.log("\n" + "="*60)
            self.log("📊 PIPELINE EXECUTION SUMMARY")
            self.log("="*60)
            self.log(f"✅ Successful stages: {successful_stages}")
            if failed_stages:
                self.log(f"❌ Failed stages: {failed_stages}")
            
            # Save summary to JSON
            summary = {
                'timestamp': datetime.now().isoformat(),
                'successful_stages': successful_stages,
                'failed_stages': failed_stages,
                'total_stages': len(successful_stages) + len(failed_stages)
            }
            
            summary_file = self.outputs_dir / 'pipeline_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.log(f"📄 Summary saved to: {summary_file}")
            self.log(f"📄 Detailed log saved to: {self.log_file}")
            
            return len(failed_stages) == 0
            
        except Exception as e:
            self.log(f"💥 Pipeline execution failed: {str(e)}")
            return False
    
    def show_pipeline_info(self):
        """Display information about all pipeline stages"""
        print("\n🔍 PIPELINE STAGES OVERVIEW")
        print("="*80)
        
        for stage_num, stage_info in PIPELINE_STAGES.items():
            print(f"\n📋 Stage {stage_num}: {stage_info['name']}")
            print(f"   📓 Notebook: {stage_info['notebook']}")
            print(f"   📝 Description: {stage_info['description']}")
            print(f"   📥 Inputs: {', '.join(stage_info['inputs'])}")
            print(f"   📤 Outputs: {', '.join(stage_info['outputs'])}")

def main():
    parser = argparse.ArgumentParser(
        description='Run the complete multilingual speech translation pipeline'
    )
    parser.add_argument(
        '--stage', type=int, choices=range(1, 7),
        help='Run a specific stage only (1-6)'
    )
    parser.add_argument(
        '--start', type=int, default=1, choices=range(1, 7),
        help='Start from this stage (default: 1)'
    )
    parser.add_argument(
        '--end', type=int, default=6, choices=range(1, 7),
        help='End at this stage (default: 6)'
    )
    parser.add_argument(
        '--info', action='store_true',
        help='Show pipeline information and exit'
    )
    parser.add_argument(
        '--project-root', type=str,
        help='Project root directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    runner = PipelineRunner(args.project_root)
    
    if args.info:
        runner.show_pipeline_info()
        return
    
    if args.stage:
        # Run single stage
        success = runner.run_pipeline(args.stage, args.stage)
    else:
        # Run range of stages
        success = runner.run_pipeline(args.start, args.end)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()