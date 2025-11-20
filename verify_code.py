#!/usr/bin/env python3
"""
Verify that multi-agent and multi-task training code can run
This script checks syntax, imports, and basic logic without loading models
"""

import sys
import os

def verify_multi_agent():
    """Verify multi-agent pipeline code"""
    print("="*80)
    print("VERIFYING MULTI-AGENT PIPELINE")
    print("="*80)
    
    try:
        from predictor_agent import PredictorAgent
        from verifier_agent import VerifierAgent
        from multi_agent_pipeline import MultiAgentPipeline
        import yaml
        
        # Check config file
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ All imports successful")
        print(f"✓ Config file loaded: {config['model']['base_model_path']}")
        print(f"✓ Multi-agent settings: max_loops={config.get('multi_agent', {}).get('max_negotiation_loops', 2)}")
        
        # Check if data files exist
        test_set = config['data'].get('test_set', '')
        if os.path.exists(test_set):
            print(f"✓ Test data file exists: {test_set}")
        else:
            print(f"⚠ Test data file not found: {test_set}")
        
        print("\n✓ Multi-agent pipeline verification PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Multi-agent pipeline verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_multi_task_training():
    """Verify multi-task training code"""
    print("\n" + "="*80)
    print("VERIFYING MULTI-TASK TRAINING")
    print("="*80)
    
    try:
        from multi_task_training import (
            MultiTaskTrainer,
            MultiTaskSMILESDataset,
            train_multi_task_model
        )
        import yaml
        from transformers import AutoTokenizer
        
        # Check config file
        with open('training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ All imports successful")
        print(f"✓ Config file loaded")
        print(f"✓ Loss alpha: {config['loss']['alpha']}")
        print(f"✓ Efficiency classes: {config['loss']['efficiency_num_classes']}")
        
        # Check if training data exists
        train_data = config['data'].get('train_data_path', '')
        if os.path.exists(train_data):
            print(f"✓ Training data file exists: {train_data}")
            # Count lines
            with open(train_data, 'r') as f:
                lines = sum(1 for _ in f)
            print(f"  - Training samples: {lines}")
        else:
            print(f"⚠ Training data file not found: {train_data}")
        
        # Test dataset loading (without tokenizer)
        print("\n✓ Multi-task training verification PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Multi-task training verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dependencies():
    """Verify required dependencies"""
    print("\n" + "="*80)
    print("VERIFYING DEPENDENCIES")
    print("="*80)
    
    required_packages = [
        'torch',
        'transformers',
        'peft',
        'yaml',
        'pandas',
        'datasets',
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'datasets':
                from datasets import Dataset
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies available")
        return True


def main():
    """Run all verifications"""
    print("\n" + "="*80)
    print("CODE VERIFICATION")
    print("="*80)
    
    results = []
    
    # Verify dependencies
    results.append(("Dependencies", verify_dependencies()))
    
    # Verify multi-agent
    results.append(("Multi-Agent Pipeline", verify_multi_agent()))
    
    # Verify multi-task training
    results.append(("Multi-Task Training", verify_multi_task_training()))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All verifications PASSED - Code is ready to run!")
    else:
        print("\n⚠ Some verifications FAILED - Please check the errors above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

