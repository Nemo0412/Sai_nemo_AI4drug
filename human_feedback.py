"""
Human Feedback Interface
Collect pairwise comparisons from human experts
"""

import json
import os
from datetime import datetime

class HumanFeedbackInterface:
    """Interface for collecting human feedback on molecular pairs"""
    
    def __init__(self, history_file="output/feedback_history.json"):
        """
        Initialize feedback interface
        
        Args:
            history_file: Path to save feedback history
        """
        self.history_file = history_file
        self.feedback_history = []
        self.load_history()
    
    def load_history(self):
        """Load feedback history from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.feedback_history = json.load(f)
            print(f"Loaded {len(self.feedback_history)} feedback records from history")
    
    def save_history(self):
        """Save feedback history to file"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_history, f, indent=2, ensure_ascii=False)
        print(f"Saved feedback history to {self.history_file}")
    
    def select_pairs(self, unsure_molecules, num_pairs=5, strategy='entropy'):
        """
        Select molecule pairs for human feedback
        
        Args:
            unsure_molecules: List of molecules with low confidence
            num_pairs: Number of pairs to select
            strategy: Selection strategy ('entropy', 'random', 'margin')
            
        Returns:
            list: List of molecule pairs
        """
        if len(unsure_molecules) < 2:
            return []
        
        pairs = []
        
        if strategy == 'entropy':
            # Sort by entropy (highest uncertainty first)
            sorted_mols = sorted(unsure_molecules, 
                               key=lambda x: x.get('entropy', 0), 
                               reverse=True)
        elif strategy == 'margin':
            # Sort by score margin (molecules close to threshold)
            sorted_mols = sorted(unsure_molecules,
                               key=lambda x: abs(x.get('predicted_score', 5) - 7))
        else:  # random
            import random
            sorted_mols = unsure_molecules.copy()
            random.shuffle(sorted_mols)
        
        # Create pairs from most uncertain molecules
        for i in range(min(num_pairs, len(sorted_mols) // 2)):
            mol1 = sorted_mols[i * 2]
            mol2 = sorted_mols[i * 2 + 1]
            pairs.append((mol1, mol2))
        
        return pairs
    
    def collect_feedback(self, pairs, round_num=1):
        """
        Collect feedback from human expert (pairwise comparison - legacy method)
        
        Args:
            pairs: List of molecule pairs
            round_num: Current round number
            
        Returns:
            list: List of feedback records
        """
        print(f"\n{'='*80}")
        print(f"HUMAN FEEDBACK ROUND {round_num}")
        print(f"{'='*80}")
        print(f"\nPlease compare the following {len(pairs)} pairs of molecules.")
        print("For each pair, indicate which molecule should have a HIGHER score.\n")
        
        feedback_records = []
        
        for i, (mol1, mol2) in enumerate(pairs):
            print(f"\n{'-'*80}")
            print(f"PAIR {i+1}/{len(pairs)}")
            print(f"{'-'*80}")
            
            print(f"\nMolecule A:")
            print(f"  SMILES: {mol1['smiles']}")
            print(f"  Predicted Score: {mol1.get('predicted_score', 'N/A'):.2f}" if mol1.get('predicted_score') else "  Predicted Score: N/A")
            print(f"  Confidence: {mol1.get('confidence', 'N/A'):.2f}" if mol1.get('confidence') else "  Confidence: N/A")
            
            print(f"\nMolecule B:")
            print(f"  SMILES: {mol2['smiles']}")
            print(f"  Predicted Score: {mol2.get('predicted_score', 'N/A'):.2f}" if mol2.get('predicted_score') else "  Predicted Score: N/A")
            print(f"  Confidence: {mol2.get('confidence', 'N/A'):.2f}" if mol2.get('confidence') else "  Confidence: N/A")
            
            # Get human input
            while True:
                print(f"\nWhich molecule should have a HIGHER score?")
                print("  Enter 'A' for Molecule A")
                print("  Enter 'B' for Molecule B")
                print("  Enter 'E' for Equal/Similar")
                print("  Enter 'S' to Skip this pair")
                
                choice = input("Your choice: ").strip().upper()
                
                if choice in ['A', 'B', 'E', 'S']:
                    break
                else:
                    print("Invalid input. Please enter A, B, E, or S.")
            
            if choice == 'S':
                print("Skipped this pair.")
                continue
            
            # Record feedback
            feedback = {
                'round': round_num,
                'pair_id': i + 1,
                'timestamp': datetime.now().isoformat(),
                'molecule_a': {
                    'smiles': mol1['smiles'],
                    'predicted_score': mol1.get('predicted_score'),
                    'confidence': mol1.get('confidence')
                },
                'molecule_b': {
                    'smiles': mol2['smiles'],
                    'predicted_score': mol2.get('predicted_score'),
                    'confidence': mol2.get('confidence')
                },
                'human_choice': choice,
                'interpretation': {
                    'A': 'A > B',
                    'B': 'B > A',
                    'E': 'A ≈ B'
                }.get(choice, 'Unknown')
            }
            
            feedback_records.append(feedback)
            self.feedback_history.append(feedback)
            
            print(f"✓ Recorded: {feedback['interpretation']}")
        
        # Save after each round
        self.save_history()
        
        print(f"\n{'='*80}")
        print(f"Collected {len(feedback_records)} feedback records in this round.")
        print(f"{'='*80}\n")
        
        return feedback_records
    
    def collect_efficiency_feedback(self, molecule, round_num=1):
        """
        Collect true efficiency score from human expert for a molecule
        
        Args:
            molecule: Molecule dictionary with predictions
            round_num: Current round number
            
        Returns:
            dict: Feedback record with true efficiency score
        """
        print(f"\n{'='*80}")
        print(f"HUMAN FEEDBACK - EFFICIENCY SCORE")
        print(f"{'='*80}")
        
        print(f"\nMolecule Information:")
        print(f"  SMILES: {molecule.get('smiles', 'N/A')}")
        print(f"  Predicted Toxicity: {molecule.get('toxicity', {}).get('score', 'N/A')}")
        print(f"  Predicted Efficiency: {molecule.get('efficiency', {}).get('score', 'N/A')}")
        print(f"  Reasoning: {molecule.get('reasoning', 'N/A')[:200]}...")
        print(f"  Confidence: {molecule.get('overall_confidence', 'N/A'):.2f}")
        
        print(f"\n{'='*80}")
        print("The Predictor and Verifier agents could not reach agreement.")
        print("Please provide the TRUE efficiency score for this molecule.")
        print(f"{'='*80}\n")
        
        # Get human input
        while True:
            try:
                true_efficiency = input("Enter the true efficiency score (1-10): ").strip()
                true_efficiency = float(true_efficiency)
                
                if 1 <= true_efficiency <= 10:
                    break
                else:
                    print("Invalid input. Please enter a score between 1 and 10.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Record feedback
        feedback = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'molecule': {
                'smiles': molecule.get('smiles'),
                'predicted_toxicity': molecule.get('toxicity', {}).get('score'),
                'predicted_efficiency': molecule.get('efficiency', {}).get('score'),
                'reasoning': molecule.get('reasoning'),
                'confidence': molecule.get('overall_confidence')
            },
            'true_efficiency': true_efficiency,
            'feedback_type': 'efficiency_score'
        }
        
        self.feedback_history.append(feedback)
        self.save_history()
        
        print(f"\n✓ Recorded: True efficiency score = {true_efficiency}")
        print(f"{'='*80}\n")
        
        return feedback
    
    def get_feedback_summary(self):
        """Get summary of all feedback collected"""
        if not self.feedback_history:
            return "No feedback collected yet."
        
        total = len(self.feedback_history)
        choices = {}
        for record in self.feedback_history:
            choice = record.get('human_choice', 'Unknown')
            choices[choice] = choices.get(choice, 0) + 1
        
        summary = f"Total Feedback Records: {total}\n"
        summary += f"Choices Distribution:\n"
        for choice, count in sorted(choices.items()):
            summary += f"  {choice}: {count} ({count/total*100:.1f}%)\n"
        
        return summary

if __name__ == "__main__":
    # Test the interface
    print("Testing Human Feedback Interface...")
    
    # Create dummy molecules
    mol1 = {
        'smiles': 'CCO',
        'predicted_score': 6.5,
        'confidence': 7.2,
        'entropy': 0.8
    }
    
    mol2 = {
        'smiles': 'CCCO',
        'predicted_score': 7.1,
        'confidence': 6.8,
        'entropy': 0.9
    }
    
    interface = HumanFeedbackInterface("test_feedback_history.json")
    pairs = [(mol1, mol2)]
    
    # Simulate feedback collection
    # feedback = interface.collect_feedback(pairs, round_num=1)
    # print("\nFeedback Summary:")
    # print(interface.get_feedback_summary())
    
    print("Interface module loaded successfully.")

