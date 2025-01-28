from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import AdamW
import json
import os
import torch.distributed as dist

# Step 1: Load and Tokenize the Dataset
def load_and_tokenize_dataset():
    # Load Wikipedia with trust_remote_code=True
    wikipedia = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)["train"]  # Replace with the latest version

    # Load BookCorpus (e.g., Project Gutenberg)
    bookcorpus = load_dataset("bookcorpus", split="train")

    # Combine datasets
    combined_dataset = concatenate_datasets([wikipedia, bookcorpus])

    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Tokenize the dataset and prepare labels for MLM
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)  # Shorter sequence length
        tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels for MLM
        return tokenized

    tokenized_datasets = combined_dataset.map(tokenize_function, batched=True)

    # Split into train and validation sets
    split_datasets = tokenized_datasets.train_test_split(test_size=0.1)  # 90% train, 10% validation
    return split_datasets

# Step 2: Set Up the 110M Parameter RoBERTa Model
def setup_roberta_model():
    # Define the standard RoBERTa-base configuration (110M parameters)
    config = RobertaConfig(
        vocab_size=50265,  # Standard RoBERTa vocabulary size
        hidden_size=768,   # Standard hidden size for RoBERTa-base
        num_hidden_layers=12,  # Standard number of layers for RoBERTa-base
        num_attention_heads=12,  # Standard number of attention heads for RoBERTa-base
        intermediate_size=3072,  # Standard intermediate size for RoBERTa-base
        max_position_embeddings=514,  # Maximum sequence length
        type_vocab_size=1,  # Typically 1 for RoBERTa
        dropout=0.1,  # Dropout rate
        attention_dropout=0.1,  # Attention dropout rate
    )

    # Create the RoBERTa model for masked language modeling
    model = RobertaForMaskedLM(config=config)
    return model

# Step 3: Define a Function to Compute Perplexity
def compute_perplexity(eval_pred):
    logits, labels = eval_pred
    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(torch.tensor(logits).view(-1, logits.shape[-1]), torch.tensor(labels).view(-1))
    perplexity = torch.exp(loss).item()  # Perplexity = exp(loss)
    return {"perplexity": perplexity}

# Step 4: Define a Custom Callback to Save Perplexity Values
class PerplexityCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.perplexity_values = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Save perplexity after each evaluation
        self.perplexity_values.append(metrics["eval_perplexity"])
        with open("perplexity_values.json", "w") as f:
            json.dump(self.perplexity_values, f)  # Save to a JSON file
        return control

# Step 5: Set Up Training Arguments and Trainer
def setup_trainer(model, tokenized_datasets):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        max_steps=80,  # Run for 80 steps only to approximate time
        per_device_train_batch_size=8,  # Effective batch size = 8 * 8 GPUs = 64
        per_device_eval_batch_size=8,
        save_steps=40,  # Save model every 40 steps
        save_total_limit=1,  # Keep only the last checkpoint
        logging_dir="./logs",
        logging_steps=10,  # Log every 10 steps
        evaluation_strategy="steps",  # Evaluate every `eval_steps`
        eval_steps=10,  # Evaluate every 10 steps
        warmup_steps=10,  # Warmup steps
        learning_rate=1e-3,  # Scaled learning rate for 8 GPUs
        weight_decay=0.01,  # Weight decay
        fp16=True,  # Enable mixed precision training
        dataloader_num_workers=4,  # Number of CPU workers for data loading
        gradient_accumulation_steps=1,  # No gradient accumulation needed
        load_best_model_at_end=False,  # No need to load best model for short run
        metric_for_best_model="perplexity",
        greater_is_better=False,  # Lower perplexity is better
        local_rank=int(os.getenv("LOCAL_RANK", 0)),  # For distributed training
    )

    # Use AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # Use the full training dataset
        eval_dataset=tokenized_datasets["test"],  # Use the full validation dataset
        optimizers=(optimizer, None),  # Use AdamW optimizer
        compute_metrics=compute_perplexity,  # Compute perplexity during evaluation
    )
    return trainer

# Step 6: Main Function to Run the Training
def main():
    # Initialize distributed backend (SLURM-aware)
    if "SLURM_PROCID" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")

    # Load and tokenize the dataset
    tokenized_datasets = load_and_tokenize_dataset()

    # Set up the 110M parameter RoBERTa model
    model = setup_roberta_model()

    # Set up the Trainer
    trainer = setup_trainer(model, tokenized_datasets)

    # Add the custom callback to the trainer
    perplexity_callback = PerplexityCallback()
    trainer.add_callback(perplexity_callback)

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training complete!")

# Run the script
if __name__ == "__main__":
    main()
