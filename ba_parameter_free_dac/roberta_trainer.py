from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, Trainer, TrainingArguments, TrainerCallback, set_seed, get_cosine_with_hard_restarts_schedule_with_warmup
from datasets import load_dataset, concatenate_datasets, DatasetDict
import time
import torch
from transformers import AdamW
from transformers import DataCollatorForLanguageModeling
import json
import os
import subprocess
import sys
from transformers import DataCollatorForLanguageModeling
from torch.distributed.run import run as torchrun_run, get_args_parser
import torch.nn.functional as F
from parameterfree.cocob_optimizer import COCOB
from parameterfree.cocob_trackable_optimizer import COCOBTrackable
from parameterfree.STORMplus import STORMplus
from parameterfree.DoWG import DoWG, CDoWG
from parameterfree.dadaptation import DAdaptAdam
from parameterfree.prodigy import Prodigy
from torch.optim import AdamW, Adam
import hydra
from accelerate import notebook_launcher
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR


os.environ["WANDB_DISABLED"] = "true"


def load_and_tokenize_dataset(save_path='tokenized_dataset', subset_ratio=0.001, batch_size=8):
    if os.path.exists(save_path):
        print(f"Loading tokenized dataset from {save_path}...")
        return DatasetDict.load_from_disk(save_path), RobertaTokenizer.from_pretrained("roberta-base")
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
        tokenized = tokenizer(examples["text"], padding=True, truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Tokenize datasets
    tokenized_subset = small_dataset.map(tokenize_function, batched=True)

    split_datasets = tokenized_subset.train_test_split(test_size=0.005)

    # Save tokenized dataset
    print(f"Saving tokenized dataset to {save_path}...")
    split_datasets.save_to_disk(save_path)

    # Compute batch-adjusted sizes
    subset_batches = len(tokenized_subset) // batch_size

    tokenization_time = time.time() - start_time  # End timing
    print(f"⏳ Tokenization Time: {tokenization_time:.2f} seconds")
    print(f"🟡 Subset Dataset: {subset_batches} batches (batch size={batch_size})")

    return split_datasets, tokenizer

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
            perplexity = torch.exp(torch.tensor(logs["loss"])).item()
            logs["train_perplexity"] = perplexity

            if state.log_history:
                # This ensures the custom metric is saved in the final log history.
                state.log_history[-1]["train_perplexity"] = perplexity
        return control

class EffectiveLrCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            optimizer = kwargs.get("optimizer")
            if hasattr(optimizer, 'avg_effective_lr') and optimizer.avg_effective_lr:
                logs["avg_effective_lr"] = optimizer.avg_effective_lr.item()

                if state.log_history:
                    # This ensures the custom metric is saved in the final log history.
                    state.log_history[-1]["effective_lr"] = optimizer.avg_effective_lr.item()
        return control
    
class DLRCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            optimizer = kwargs.get("optimizer")
            if hasattr(optimizer, 'dlr') and optimizer.dlr:
                logs["dlr"] = optimizer.dlr

                if state.log_history:
                    # This ensures the custom metric is saved in the final log history.
                    state.log_history[-1]["dlr"] = optimizer.dlr
        return control

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
    
class LearningRateTrackerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Extract the current learning rate from the optimizer's first parameter group.
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            logs["tracked_learning_rate"] = current_lr
            state.log_history[-1]["tracked_learning_rate"] = current_lr
        return control

# Step 5: Set Up Training Arguments and Trainer
def setup_trainer(model, tokenized_datasets, tokenizer, optimizer_cfg, use_evaluation=True, steps=23000, warmup=10000):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        max_steps=steps,
        # RESET to 128, just for testing
        per_device_train_batch_size=64,  # Effective batch size = 64 * 4 GPUs = 256
        # RESET to 256
        per_device_eval_batch_size=128,
        # deepspeed="../deepspeed_config.json",
        # eval_accumulation_steps=64,
        #gradient_accumulation_steps=8,
        save_steps=1000,
        save_total_limit=1,  # Keep only the last checkpoint
        logging_dir="./logs",
        logging_steps=1,  # Log every step
        evaluation_strategy="steps" if use_evaluation else "no",  # Evaluate every `eval_steps`
        eval_steps=50,  # Evaluate every 10 steps. Maybe we should even evaluate every step but this would make it much more expensive
        warmup_steps=warmup,  # Warmup steps from D-Adaptation
        lr_scheduler_type="linear",  # Disables lr decay
        learning_rate=0.001,  # Scaled learning rate for 8 GPUs
        weight_decay=0.0,  # Weight decay
        bf16=True,  # Enable mixed precision training
        dataloader_num_workers=2,  # Number of CPU workers for data loading
        ddp_find_unused_parameters=False,
        # gradient_accumulation_steps=16,
        # load_best_model_at_end=False, 
        # metric_for_best_model="perplexity",
        # greater_is_better=False,
        # local_rank=int(os.getenv("LOCAL_RANK", 0)),
    )

    # Use AdamW optimizer
    kwargs = {}
    if 'lr' in optimizer_cfg:
        kwargs['lr'] = optimizer_cfg.lr
    if 'weight_decay' in optimizer_cfg:
        kwargs['weight_decay'] = optimizer_cfg.weight_decay
    if 'decouple' in optimizer_cfg:
        kwargs['decouple'] = optimizer_cfg.decouple
    if 'betas' in optimizer_cfg:
        kwargs['betas'] = optimizer_cfg.betas

    optimizer = (get_optimizer_type(optimizer_cfg.type)(**kwargs, params=model.parameters()) if kwargs is not None else
                get_optimizer_type(optimizer_cfg.type)(params=model.parameters()))
    print(optimizer)

    scheduler = None
    if 'cawr' in optimizer_cfg:
        # Assuming training_args is defined before this block
        cawr = CosineAnnealingWarmRestarts(optimizer, optimizer_cfg.cawr.T_0,
                                                optimizer_cfg.cawr.t_mult, optimizer_cfg.cawr.eta_min)

        warmup_schedule = LinearLR(optimizer, start_factor=1/warmup, end_factor=1.0, total_iters=warmup)

        scheduler = SequentialLR(optimizer,
                            schedulers=[warmup_schedule, cawr],
                            milestones=[warmup])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # Use the full training dataset
        eval_dataset=tokenized_datasets["test"],  # Use the full validation dataset
        optimizers=(optimizer, scheduler),  # Use AdamW optimizer
        compute_metrics=compute_perplexity,  # Compute perplexity during evaluation
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[TrainPerplexityCallback, EffectiveLrCallback, DLRCallback]
    )

    if 'track' in optimizer_cfg and optimizer_cfg.track:
        trainer.add_callback(LearningRateTrackerCallback())

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
        case "pureadam":
            return Adam
        
    return AdamW

def main(cfg):
    set_seed(cfg.seed)
    print(cfg)
    print("Load and Tokenize dataset")
    # Load and tokenize the dataset
    # TODO RESET subset_ratio to 0.16
    tokenized_datasets, tokenizer = load_and_tokenize_dataset(save_path='../../tokenized_dataset', subset_ratio=0.01, batch_size=512)

    print("Setup Model")
    # Set up the 110M parameter RoBERTa model
    model = setup_roberta_model()
    device = torch.device("cuda:0")
    model.to(device)


    print("Setup Trainer")
    # Set up the Trainer
    trainer = setup_trainer(model, tokenized_datasets, tokenizer, cfg.optimizer, cfg.use_evaluation, int(cfg.steps), cfg.get("warmup"))

    # Add the custom callback to the trainer
    perplexity_callback = PerplexityCallback()
    trainer.add_callback(perplexity_callback)

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training complete!") 
    return trainer.state.log_history[-1]["train_loss"]

def main_worker(rank: int, cfg, return_dict):
    # Set the CUDA device for this process.
    torch.cuda.set_device(rank)
    
    # Set DDP environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(cfg.nproc)
    
    # os.environ.setdefault("MASTER_ADDR", "localhost")
    # os.environ.setdefault("MASTER_PORT", str(get_free_port()))

    # Initialize the process group (using NCCL for GPU training)
    dist.init_process_group(backend="nccl", init_method="env://")
    
    # Now run your main training function
    result = main(cfg)
    
    # Clean up the process group after training
    dist.destroy_process_group()
    return_dict[rank] = result
