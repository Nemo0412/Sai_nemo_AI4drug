#!/usr/bin/env python3
"""
Demo script showing Multi-Agent Pipeline with Accuracy Tracking
This demonstrates how accuracy changes across negotiation loops
"""

import os
import sys
import yaml
import json
from multi_agent_pipeline import MultiAgentPipeline

def create_demo_config():
    """Create a demo configuration"""
    config = {
        'model': {
            'base_model_path': 'models/qwen_7b',
            'lora_path': 'qwen_lora_finetuned_multi_task'
        },
        'data': {
            'test_set': 'dataset/1600set.xlsx',
            'output_dir': 'demo_accuracy_results',
            'feedback_history_file': 'demo_accuracy_results/feedback_history.json'
        },
        'prediction': {
            'num_samples': 3,
            'temperature': 0.7,
            'top_p': 0.8,
            'max_length': 512
        },
        'multi_agent': {
            'max_negotiation_loops': 3,  # More loops to see accuracy trend
            'consensus_threshold': 0.8,
            'low_confidence_threshold': 6.0
        }
    }
    
    # Save demo config
    with open('demo_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return 'demo_config.yaml'

def simulate_multi_agent_with_accuracy():
    """Simulate multi-agent pipeline with accuracy tracking"""
    print("="*80)
    print("DEMO: MULTI-AGENT PIPELINE WITH ACCURACY TRACKING")
    print("="*80)
    print("This demo shows how accuracy is tracked across negotiation loops")
    print("="*80)
    
    # Create demo config
    config_path = create_demo_config()
    
    try:
        # Initialize pipeline
        pipeline = MultiAgentPipeline(config_path)
        
        # Load test labels
        pipeline.load_test_labels()
        print(f"\nLoaded test labels for {len(pipeline.test_labels)} molecules")
        print("Note: Toxicity accuracy includes 2000 default correct predictions for efficiency molecules")
        
        if not pipeline.test_labels:
            print("No test labels found. Cannot demonstrate accuracy tracking.")
            return
        
        # Simulate predictions for a few molecules across multiple loops
        sample_molecules = list(pipeline.test_labels.keys())[:5]  # Take 5 molecules
        
        print(f"\nSimulating predictions for {len(sample_molecules)} molecules...")
        print("This will show how accuracy changes as more molecules are processed.")
        
        for loop_num in range(1, 4):  # Simulate 3 loops
            print(f"\n{'='*60}")
            print(f"SIMULATION LOOP {loop_num}")
            print(f"{'='*60}")
            
            # Simulate predictions for molecules in this loop
            molecules_in_loop = sample_molecules[:loop_num + 1]  # Add one more molecule each loop
            
            for i, smiles in enumerate(molecules_in_loop):
                true_labels = pipeline.test_labels[smiles]
                
                # Simulate improving predictions over loops
                # Loop 1: Random predictions (low accuracy)
                # Loop 2: Better predictions (medium accuracy) 
                # Loop 3: Good predictions (high accuracy)
                
                if loop_num == 1:
                    # Poor predictions
                    pred_tox = 1 - true_labels.get('toxicity', 0) if 'toxicity' in true_labels else 0
                    pred_eff = max(1, min(10, true_labels.get('efficiency', 5) + 3)) if 'efficiency' in true_labels else 5
                elif loop_num == 2:
                    # Better predictions
                    pred_tox = true_labels.get('toxicity', 0) if i % 2 == 0 else 1 - true_labels.get('toxicity', 0)
                    pred_eff = true_labels.get('efficiency', 5) if i % 2 == 0 else max(1, min(10, true_labels.get('efficiency', 5) + 1))
                else:
                    # Good predictions
                    pred_tox = true_labels.get('toxicity', 0)
                    pred_eff = true_labels.get('efficiency', 5)
                
                # Update predictions
                pipeline.current_predictions[smiles] = {
                    'toxicity': pred_tox,
                    'efficiency': pred_eff
                }
                
                print(f"  Molecule {i+1}: {smiles[:30]}...")
                print(f"    True:  tox={true_labels.get('toxicity', 'N/A')}, eff={true_labels.get('efficiency', 'N/A')}")
                print(f"    Pred:  tox={pred_tox}, eff={pred_eff}")
            
            # Calculate and display accuracy for this loop
            accuracy = pipeline.calculate_current_accuracy()
            print(f"\n  [Loop {loop_num}] Accuracy Summary:")
            print(f"    Toxicity:  {accuracy['toxicity_accuracy']:.4f} ({accuracy['toxicity_samples']} samples)")
            print(f"    Efficiency: {accuracy['efficiency_accuracy']:.4f} ({accuracy['efficiency_samples']} samples)")
            
            # Store in history
            pipeline.loop_accuracy_history.append({
                'loop': loop_num,
                'molecule': f"Batch of {len(molecules_in_loop)} molecules",
                'toxicity_accuracy': accuracy['toxicity_accuracy'],
                'efficiency_accuracy': accuracy['efficiency_accuracy'],
                'toxicity_samples': accuracy['toxicity_samples'],
                'efficiency_samples': accuracy['efficiency_samples']
            })
        
        # Display final accuracy trend
        print(f"\n{'='*80}")
        print("ACCURACY TREND ANALYSIS")
        print(f"{'='*80}")
        
        pipeline.display_accuracy_trend()
        
        print(f"\n{'='*80}")
        print("DEMO COMPLETED")
        print(f"{'='*80}")
        print("Key Features Demonstrated:")
        print("  ✓ Real-time accuracy calculation during loops")
        print("  ✓ Accuracy tracking across multiple negotiation rounds")
        print("  ✓ Separate tracking for toxicity and efficiency")
        print("  ✓ Accuracy trend visualization")
        print("  ✓ Automatic saving of accuracy history")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    simulate_multi_agent_with_accuracy()
