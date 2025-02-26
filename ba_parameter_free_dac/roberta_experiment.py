from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, Trainer, TrainingArguments, TrainerCallback, set_seed
from datasets import load_dataset, concatenate_datasets, DatasetDict
import time
import torch
from transformers import AdamW
import json
import os
import sys
import torch.nn.functional as F
from parameterfree.cocob_optimizer import COCOB
from parameterfree.cocob_trackable_optimizer import COCOBTrackable
from parameterfree.STORMplus import STORMplus
from parameterfree.DoWG import DoWG, CDoWG
from parameterfree.dadaptation import DAdaptAdam
from parameterfree.prodigy import Prodigy
from torch.optim import AdamW
# from codecarbon import track_emissions
import hydra

# Step 1: Load and Tokenize the Dataset
# def load_and_tokenize_dataset():
#     print("Load Wikipedia")
#     wikipedia = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)["train"]  # Maybe replace with the latest version

#     print("Load BookCorpus")
#     bookcorpus = load_dataset("bookcorpus", split="train")

#     print("Concat")
#     # Combine datasets
#     combined_dataset = concatenate_datasets([wikipedia, bookcorpus])

#     print("Load Tokenizer")
#     # Load the tokenizer
#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#     # I think we can't leave out this pretrained tokenization because embeddings are very very hard to find

#     # Tokenize the dataset and prepare labels for MLM
#     def tokenize_function(examples):
#         # If I understand "Max tokens per sample" from D_Adapt paper correctly, we might have to set max_length=512 
#         tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
#         tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels for MLM
#         return tokenized

#     # Assuming 'combined_dataset' is your concatenated dataset:
#     subset_size = int(0.001 * len(combined_dataset))
#     # Optionally, shuffle the dataset to get a random 2%
#     combined_dataset = combined_dataset.shuffle(seed=42).select(range(subset_size))


#     print("Tokenize...")
#     tokenized_datasets = combined_dataset.map(tokenize_function, batched=True)

#     print("Train-Val-Split...")
#     # Split into train and validation sets
#     split_datasets = tokenized_datasets.train_test_split(test_size=0.1)  # 90% train, 10% validation
#     return split_datasets

def load_and_tokenize_dataset(save_path='tokenized_dataset', subset_ratio=0.001, batch_size=8):
    if os.path.exists(save_path):
        print(f"Loading tokenized dataset from {save_path}...")
        return DatasetDict.load_from_disk(save_path)
    start_time = time.time()  # Start timing

    # Load full datasets
    wikipedia = load_dataset("wikipedia", "20220301.en", split="train")
    bookcorpus = load_dataset("bookcorpus", split="train", trust_remote_code=True)

    # Combine datasets

    full_dataset = concatenate_datasets([wikipedia, bookcorpus])

    # Shuffle dataset before selecting a subset
    full_dataset = full_dataset.shuffle(seed=42)

    # Get dataset sizes
    full_size = len(full_dataset)
    subset_size = max(1, int(full_size * subset_ratio))

    # Take a shuffled small subset
    small_dataset = full_dataset.select(range(subset_size))

    print(f"📊 Full Dataset Size: {full_size:,} samples")
    print(f"📉 Subset Size ({subset_ratio * 100:.2f}%): {subset_size:,} samples (shuffled)")

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Tokenize datasets
    tokenized_subset = small_dataset.map(tokenize_function, batched=True)

    split_datasets = tokenized_subset.train_test_split(test_size=0.05)

    # Save tokenized dataset
    print(f"Saving tokenized dataset to {save_path}...")
    split_datasets.save_to_disk(save_path)

    # Compute batch-adjusted sizes
    subset_batches = len(tokenized_subset) // batch_size

    tokenization_time = time.time() - start_time  # End timing
    print(f"⏳ Tokenization Time: {tokenization_time:.2f} seconds")
    print(f"🟡 Subset Dataset: {subset_batches} batches (batch size={batch_size})")

    return split_datasets


# Step 2: Set Up the 110M Parameter RoBERTa Model
def setup_roberta_model():
    # This should be the standard RoBERTa-base configuration (110M parameters)
    config = RobertaConfig(
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=514,
        type_vocab_size=1,
        dropout=0.1,
        attention_dropout=0.1
    )

    # Create the RoBERTa model for masked language modeling
    model = RobertaForMaskedLM(config=config)
    return model

def preprocess_logits_for_metrics(logits, labels):
    """
    Instead of returning the full logits, compute the average cross-entropy loss for the batch.
    This avoids having to store all logits.
    """
    # Compute loss per token; ensure labels with -100 are ignored.
    loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        ignore_index=-100,
        reduction="none"
    )
    # Reshape loss to [batch_size, sequence_length] and compute average loss per sample
    loss = loss.view(labels.shape)
    batch_loss = loss.mean(dim=1)  # Average loss per sample in the batch
    # Return the average loss across the batch (scalar) along with labels (if needed)
    return batch_loss.mean().unsqueeze(0), labels

def compute_perplexity(eval_pred):
    avg_loss, labels = eval_pred
    # If avg_loss is a tuple, extract the first element
    if isinstance(avg_loss, (tuple, list)):
        avg_loss = avg_loss[0]
    perplexity = torch.exp(torch.tensor(avg_loss).mean()).item()
    return {"perplexity": perplexity}

class TrainPerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            # Compute perplexity from loss
            logs["train_perplexity"] = torch.exp(torch.tensor(logs["loss"])).item()
        return control

class EffectiveLrCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # print("EffectiveLRCallback triggered")  # Debug print
        print(logs)
        if logs is not None:
            optimizer = kwargs.get("optimizer")
            if hasattr(optimizer, 'avg_effective_lr'):
                logs["avg_effective_lr"] = optimizer.avg_effective_lr
                print(f"Effective LR: {optimizer.avg_effective_lr}")  # Debug print
        return control

# def compute_perplexity(eval_pred):
#     logits, labels = eval_pred
#     # Calculate cross-entropy loss
#     loss_fct = torch.nn.CrossEntropyLoss()
#     loss = loss_fct(torch.tensor(logits).view(-1, logits.shape[-1]), torch.tensor(labels).view(-1))
#     perplexity = torch.exp(loss).item()  # Perplexity = exp(loss)
#     return {"perplexity": perplexity}

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
def setup_trainer(model, tokenized_datasets, optimizer_cfg):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        max_steps=100,
        per_device_train_batch_size=16,  # Effective batch size = 64 * 4 GPUs = 256
        per_device_eval_batch_size=256,
        eval_accumulation_steps=64,
        save_steps=1000,
        save_total_limit=1,  # Keep only the last checkpoint
        logging_dir="./logs",
        logging_steps=1,  # Log every step
        evaluation_strategy="steps",  # Evaluate every `eval_steps`
        eval_steps=50,  # Evaluate every 10 steps. Maybe we should even evaluate every step but this would make it much more expensive
        warmup_steps=1000,  # Warmup steps from D-Adaptation
        learning_rate=1e-3,  # Scaled learning rate for 8 GPUs
        weight_decay=0.0,  # Weight decay
        fp16=True,  # Enable mixed precision training
        dataloader_num_workers=2,  # Number of CPU workers for data loading
        gradient_accumulation_steps=16,
        load_best_model_at_end=False, 
        metric_for_best_model="perplexity",
        greater_is_better=False,
        local_rank=int(os.getenv("LOCAL_RANK", 0)),
    )

    # Use AdamW optimizer
    optimizer = get_optimizer_type(optimizer_cfg.type)(model.parameters(),
                                                       lr=optimizer_cfg.lr,
                                                       weight_decay=optimizer_cfg.weight_decay)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # Use the full training dataset
        eval_dataset=tokenized_datasets["test"],  # Use the full validation dataset
        optimizers=(optimizer, None),  # Use AdamW optimizer
        compute_metrics=compute_perplexity,  # Compute perplexity during evaluation
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    trainer.add_callback(TrainPerplexityCallback())
    trainer.add_callback(EffectiveLrCallback())


    return trainer

# get optimizer type from string
def get_optimizer_type(optimizer_type_name):
    match optimizer_type_name:
        case "ProdigyAdam":
            return Prodigy
        case "DAdaptAdam":
            return DAdaptAdam
        case "COCOB":
            return COCOB
        case "COCOB_trackable":
            return COCOBTrackable
        case "stormplus":
            return STORMplus
        case "dowg":
            return DoWG
        case "cdowg":
            return CDoWG
        case "adam":
            return AdamW
        
    return AdamW

@hydra.main(version_base=None, config_path="configs", config_name="adamfixed_bookwiki_roberta")
#@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg):
    set_seed(cfg.seed)
    print(cfg)
    print("Load and Tokenize dataset")
    # Load and tokenize the dataset
    tokenized_datasets = load_and_tokenize_dataset(save_path='../../tokenized_dataset', subset_ratio=0.02, batch_size=48)

    print("Setup Model")
    # Set up the 110M parameter RoBERTa model
    model = setup_roberta_model()

    print("Setup Trainer")
    # Set up the Trainer
    trainer = setup_trainer(model, tokenized_datasets, cfg.optimizer)

    # Add the custom callback to the trainer
    perplexity_callback = PerplexityCallback()
    trainer.add_callback(perplexity_callback)

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training complete!")

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover