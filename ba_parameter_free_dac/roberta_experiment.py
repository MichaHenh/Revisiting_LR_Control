from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, Trainer, TrainingArguments, TrainerCallback, set_seed
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import AdamW
import json
import os
from parameterfree.cocob_optimizer import COCOB
from parameterfree.cocob_trackable_optimizer import COCOBTrackable
from parameterfree.STORMplus import STORMplus
from parameterfree.DoWG import DoWG, CDoWG
from parameterfree.dadaptation import DAdaptAdam
from parameterfree.prodigy import Prodigy
from torch.optim import AdamW
from codecarbon import track_emissions
import hydra

# Step 1: Load and Tokenize the Dataset
def load_and_tokenize_dataset():
    print("Load Wikipedia")
    wikipedia = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)["train"]  # Maybe replace with the latest version

    print("Load BookCorpus")
    bookcorpus = load_dataset("bookcorpus", split="train")

    print("Concat")
    # Combine datasets
    combined_dataset = concatenate_datasets([wikipedia, bookcorpus])

    print("Load Tokenizer")
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # I think we can't leave out this pretrained tokenization because embeddings are very very hard to find

    # Tokenize the dataset and prepare labels for MLM
    def tokenize_function(examples):
        # If I understand "Max tokens per sample" from D_Adapt paper correctly, we might have to set max_length=512 
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels for MLM
        return tokenized

    # Assuming 'combined_dataset' is your concatenated dataset:
    subset_size = int(0.001 * len(combined_dataset))
    # Optionally, shuffle the dataset to get a random 2%
    combined_dataset = combined_dataset.shuffle(seed=42).select(range(subset_size))


    print("Tokenize...")
    tokenized_datasets = combined_dataset.map(tokenize_function, batched=True)

    print("Train-Val-Split...")
    # Split into train and validation sets
    split_datasets = tokenized_datasets.train_test_split(test_size=0.1)  # 90% train, 10% validation
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
def setup_trainer(model, tokenized_datasets, optimizer_cfg):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        max_steps=23000, 
        per_device_train_batch_size=8,  # Effective batch size = 8 * 8 GPUs = 64
        per_device_eval_batch_size=8,
        save_steps=1000,
        save_total_limit=1,  # Keep only the last checkpoint
        logging_dir="./logs",
        logging_steps=1,  # Log every step
        evaluation_strategy="steps",  # Evaluate every `eval_steps`
        eval_steps=10,  # Evaluate every 10 steps. Maybe we should even evaluate every step but this would make it much more expensive
        warmup_steps=10000,  # Warmup steps from D-Adaptation
        learning_rate=1e-3,  # Scaled learning rate for 8 GPUs
        weight_decay=0.0,  # Weight decay
        fp16=True,  # Enable mixed precision training
        dataloader_num_workers=4,  # Number of CPU workers for data loading
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
    )
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
@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg):
    set_seed(cfg.seed)
    print("Load and Tokenize dataset")
    # Load and tokenize the dataset
    tokenized_datasets = load_and_tokenize_dataset()

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