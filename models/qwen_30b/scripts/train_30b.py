"""
LoRA Finetuning script for Qwen3-30B model with SMILES drug discovery data
Optimized for 30B parameter models with memory efficiency
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

class SMILESDataset:
    """Dataset class for SMILES molecular data"""
    
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(jsonl_file)
    
    def load_data(self, jsonl_file):
        """Load data from JSONL file"""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def format_conversation(self, messages):
        """Format messages into a single string for training"""
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return formatted
    
    def preprocess_data(self):
        """Preprocess data for training"""
        processed = []
        for item in self.data:
            text = self.format_conversation(item['messages'])
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            encoded['labels'] = encoded['input_ids'].copy()
            processed.append(encoded)
        return Dataset.from_list(processed)

def setup_lora_config():
    """Configure LoRA parameters for 30B model"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Higher rank for larger model
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return lora_config

def train_model(
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
    train_data_path="../../../data/train_data.jsonl",
    output_dir="../checkpoints/qwen_30b_lora_finetuned",
    num_epochs=3,
    batch_size=1,
    learning_rate=1e-4,
    max_length=512
):
    """
    Finetune Qwen3-30B model with LoRA on SMILES data
    
    Args:
        model_name: HuggingFace model name or local path
        train_data_path: Path to training data JSONL file
        output_dir: Directory to save the finetuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        max_length: Maximum sequence length
    """
    
    print(f"Loading tokenizer and model: {model_name}")
    print("Model size: 30B+ parameters - Using 4-bit quantization for memory efficiency")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right',
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization for 30B model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with 4-bit quantization
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA configuration
    print("Applying LoRA configuration...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print(f"Loading training data from: {train_data_path}")
    dataset_loader = SMILESDataset(train_data_path, tokenizer, max_length)
    train_dataset = dataset_loader.preprocess_data()
    print(f"Training samples: {len(train_dataset)}")
    
    # Training arguments - optimized for 30B model (40 steps, save every 5 steps)
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=40,  # Train for exactly 40 steps
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=5,  # 40 steps with 200 samples
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        save_strategy="steps",  # Save by steps
        save_steps=5,  # Save checkpoint every 5 steps
        logging_steps=5,
        warmup_steps=20,
        optim="paged_adamw_8bit",
        save_total_limit=10,  # Keep all checkpoints (8 checkpoints: steps 5,10,15,20,25,30,35,40)
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        logging_dir=f"{output_dir}/logs",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Start training
    print("Starting training for Qwen2-32B...")
    print("Note: 30B model training will take longer than 7B")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    # Configuration for Qwen3-30B
    config = {
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "train_data_path": "../../../data/train_data.jsonl",
        "output_dir": "../checkpoints/qwen_30b_lora_40steps",  # New directory for 40-step training
        "num_epochs": 3,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "max_length": 512
    }
    
    # Train the model
    train_model(**config)

