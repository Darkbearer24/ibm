#!/usr/bin/env python3
"""
Sprint 4 Training Pipeline Test

This script tests the complete training pipeline to ensure all components
are working correctly for GPU training export.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.train import AudioDataset, Trainer, collate_fn
from models.encoder_decoder import create_model, count_parameters
from utils.framing import create_feature_matrix_advanced
from utils.denoise import preprocess_audio_complete


def test_system_requirements():
    """Test system requirements and GPU availability."""
    print("🔍 Testing System Requirements")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        device = torch.device('cuda')
    else:
        print("CUDA not available - will use CPU")
        device = torch.device('cpu')
    
    print(f"Selected device: {device}")
    print("✅ System requirements check completed\n")
    
    return device


def test_model_creation(device):
    """Test model creation and GPU compatibility."""
    print("🏗️ Testing Model Creation")
    print("=" * 50)
    
    try:
        model, loss_fn = create_model()
        param_count = count_parameters(model)
        print(f"✅ Model created with {param_count:,} parameters")
        
        # Move to device
        model = model.to(device)
        print(f"✅ Model moved to device: {device}")
        
        # Test forward pass with dummy data
        batch_size, seq_len, n_features = 2, 100, 441
        dummy_input = torch.randn(batch_size, seq_len, n_features).to(device)
        
        with torch.no_grad():
            output, latent = model(dummy_input)
            print(f"✅ Forward pass successful")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Latent shape: {latent.shape}")
        
        # Test loss calculation
        loss_dict = loss_fn(output, dummy_input, latent)
        print(f"✅ Loss calculation successful")
        print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"   Reconstruction loss: {loss_dict['reconstruction_loss'].item():.4f}")
        if 'latent_regularization' in loss_dict:
            print(f"   Latent regularization: {loss_dict['latent_regularization'].item():.4f}")
        else:
            print(f"   Latent regularization: Not applied")
        
        print("✅ Model creation test completed\n")
        return model, loss_fn
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None, None


def test_dataset_loading():
    """Test dataset loading with sample data."""
    print("📁 Testing Dataset Loading")
    print("=" * 50)
    
    # Check for Bengali data
    data_dir = project_root / 'data' / 'raw' / 'Bengali'
    
    if not data_dir.exists():
        print(f"⚠️ Data directory not found: {data_dir}")
        print("   Creating dummy dataset for testing...")
        return create_dummy_dataset()
    
    try:
        # Create dataset with limited files for testing
        dataset = AudioDataset(
            data_dir=str(data_dir),
            max_files=3  # Small sample for testing
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded successfully")
            print(f"   Sample shape: {sample.shape}")
            print(f"   Sample dtype: {sample.dtype}")
            print(f"   Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        print("✅ Dataset loading test completed\n")
        return dataset
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        print("   Creating dummy dataset for testing...")
        return create_dummy_dataset()


def create_dummy_dataset():
    """Create a dummy dataset for testing when real data is not available."""
    class DummyDataset:
        def __init__(self, num_samples=5):
            self.num_samples = num_samples
            # Generate random feature matrices
            self.data = []
            for _ in range(num_samples):
                seq_len = np.random.randint(50, 150)  # Variable length
                features = torch.randn(seq_len, 441)
                self.data.append(features)
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = DummyDataset()
    print(f"✅ Dummy dataset created with {len(dataset)} samples")
    return dataset


def test_dataloader(dataset, device):
    """Test DataLoader with collate function."""
    print("🔄 Testing DataLoader")
    print("=" * 50)
    
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print(f"✅ DataLoader created successfully")
        
        # Test batch loading
        for i, batch in enumerate(dataloader):
            print(f"✅ Batch {i+1} loaded")
            print(f"   Batch shape: {batch.shape}")
            print(f"   Batch dtype: {batch.dtype}")
            
            # Test moving to device
            batch = batch.to(device)
            print(f"   Batch moved to device: {device}")
            
            if i == 0:  # Only test first batch
                break
        
        print("✅ DataLoader test completed\n")
        return dataloader
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return None


def test_training_step(model, loss_fn, dataloader, device):
    """Test a single training step."""
    print("🏃 Testing Training Step")
    print("=" * 50)
    
    try:
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output, latent = model(batch)
            
            # Calculate loss
            loss_dict = loss_fn(output, batch, latent)
            loss = loss_dict['total_loss']
            
            print(f"✅ Forward pass completed")
            print(f"   Loss: {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            print(f"✅ Backward pass completed")
            print(f"✅ Training step successful")
            
            break  # Only test one step
        
        print("✅ Training step test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False


def test_checkpoint_system(model, device):
    """Test checkpoint saving and loading."""
    print("💾 Testing Checkpoint System")
    print("=" * 50)
    
    try:
        checkpoint_dir = project_root / 'test_checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create dummy trainer for checkpoint testing
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        # Create checkpoint
        checkpoint = {
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': [0.5],
            'val_losses': [0.4],
            'val_loss': 0.4,
            'best_val_loss': 0.4
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / 'test_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Checkpoint keys: {list(loaded_checkpoint.keys())}")
        print(f"   Epoch: {loaded_checkpoint['epoch']}")
        print(f"   Val loss: {loaded_checkpoint['val_loss']}")
        
        # Test model state loading
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        print(f"✅ Model state loaded successfully")
        
        print("✅ Checkpoint system test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint system test failed: {e}")
        return False


def generate_training_commands():
    """Generate example training commands."""
    print("📋 Training Commands")
    print("=" * 50)
    
    commands = {
        'gpu_training': {
            'command': 'python models/train.py --data_dir data/raw/Bengali --epochs 50 --batch_size 16 --learning_rate 0.001 --device cuda',
            'description': 'GPU training with recommended settings'
        },
        'cpu_training': {
            'command': 'python models/train.py --data_dir data/raw/Bengali --epochs 50 --batch_size 8 --learning_rate 0.001 --device cpu',
            'description': 'CPU training with reduced batch size'
        },
        'quick_test': {
            'command': 'python models/train.py --data_dir data/raw/Bengali --epochs 5 --batch_size 4 --max_files 10 --device auto',
            'description': 'Quick test run with limited data'
        }
    }
    
    for name, info in commands.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Command: {info['command']}")
    
    print("\n✅ Training commands generated\n")
    return commands


def main():
    """Main test function."""
    print("🎯 SPRINT 4 TRAINING PIPELINE TEST")
    print("=" * 60)
    print()
    
    # Test results
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # 1. Test system requirements
    device = test_system_requirements()
    results['device'] = str(device)
    
    # 2. Test model creation
    model, loss_fn = test_model_creation(device)
    results['tests']['model_creation'] = model is not None
    
    if model is None:
        print("❌ Cannot continue without working model")
        return
    
    # 3. Test dataset loading
    dataset = test_dataset_loading()
    results['tests']['dataset_loading'] = dataset is not None
    
    if dataset is None:
        print("❌ Cannot continue without dataset")
        return
    
    # 4. Test dataloader
    dataloader = test_dataloader(dataset, device)
    results['tests']['dataloader'] = dataloader is not None
    
    if dataloader is None:
        print("❌ Cannot continue without dataloader")
        return
    
    # 5. Test training step
    training_success = test_training_step(model, loss_fn, dataloader, device)
    results['tests']['training_step'] = training_success
    
    # 6. Test checkpoint system
    checkpoint_success = test_checkpoint_system(model, device)
    results['tests']['checkpoint_system'] = checkpoint_success
    
    # 7. Generate training commands
    commands = generate_training_commands()
    results['training_commands'] = commands
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = all(results['tests'].values())
    
    for test_name, passed in results['tests'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"Device: {device}")
    print(f"Model Parameters: {count_parameters(model):,}")
    
    # Save results
    results_path = project_root / 'outputs' / 'sprint4_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {results_path}")
    
    if all_passed:
        print("\n🚀 Sprint 4 training pipeline is ready for GPU export!")
    else:
        print("\n⚠️ Some issues need to be resolved before GPU training.")


if __name__ == "__main__":
    main()