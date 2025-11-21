"""
LoRA Finetuning script for Qwen 7B model with SMILES drug discovery data
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

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
    """Configure LoRA parameters"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # LoRA rank
        lora_alpha=32,  # LoRA alpha parameter
        lora_dropout=0.1,  # Dropout probability
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen2 modules
    )
    return lora_config

def train_model(
    model_name="Qwen/Qwen-7B-Chat",
    train_data_path="dataset/train_data.jsonl",
    output_dir="./qwen_lora_output",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_length=512
):
    """
    Finetune Qwen 7B model with LoRA on SMILES data
    
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
    
    # Load tokenizer with revision to avoid stream_generator issue
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right',
        revision="main"
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=True,  # Use 8-bit quantization to reduce memory
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Prepare model for kbit training
    from peft import prepare_model_for_kbit_training
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
    
    # Training arguments - optimized for memory
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=40,  # Train for exactly 40 steps
        per_device_train_batch_size=1,  # Reduced to 1 to save memory
        gradient_accumulation_steps=5,  # 40 steps with 200 samples
        learning_rate=learning_rate,
        fp16=False,  # Disabled due to 8-bit quantization
        bf16=True,  # Use bfloat16 instead
        save_strategy="steps",  # Save by steps
        save_steps=5,  # Save checkpoint every 5 steps
        logging_steps=5,
        warmup_steps=20,
        optim="paged_adamw_8bit",  # Use 8-bit optimizer to save memory
        save_total_limit=10,  # Keep all checkpoints (8 checkpoints: steps 5,10,15,20,25,30,35,40)
        report_to="none",
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=0.3,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    # Configuration for Qwen2-7B
    config = {
        "model_name": "Qwen/Qwen2-7B-Instruct",
        "train_data_path": "../../../data/train_data.jsonl",
        "output_dir": "../checkpoints/qwen_7b_lora_40steps",  # New directory for 40-step training
        "num_epochs": 3,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "max_length": 512
    }
    
    # Train the model
    train_model(**config)

