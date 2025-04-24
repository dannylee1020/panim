import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


def format_instruction(sample):
    instruction = sample.get("instruction", "")
    answer = sample.get("answer", "")
    if instruction and answer:
        return f"### Instruction:\n{instruction}\n\n### Answer:\n{answer}"
    else:
        return None


def main(args):
    model_name = args.model_name

    # QLoRA configuration
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # --- Dataset Loading and Preparation ---
    train_dataset_path = args.dataset_path
    if not os.path.isfile(train_dataset_path):
        raise ValueError(f"Training dataset file not found: {train_dataset_path}")
    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")
    print(f"Loaded {len(train_dataset)} training samples.")

    # Load evaluation data if strategy is not 'no'
    eval_dataset = None
    if args.evaluation_strategy != "no":
        eval_dataset_path = args.eval_dataset_path
        if not eval_dataset_path or not os.path.isfile(eval_dataset_path):
            raise ValueError(
                f"Evaluation dataset file not found or not specified: {eval_dataset_path}"
            )
        print(f"Loading evaluation dataset from: {eval_dataset_path}")
        eval_dataset = load_dataset("json", data_files=eval_dataset_path, split="train")
        print(f"Loaded {len(eval_dataset)} evaluation samples.")

    # --- PEFT Configuration ---
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # --- Training Arguments ---
    training_args = SFTConfig(
        # --- Core Training Arguments ---
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="wandb",
        gradient_checkpointing=True,
        # --- Evaluation Arguments ---
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if args.evaluation_strategy != "no" else None,
        greater_is_better=False,  # Lower loss is better
        # --- SFT Specific Arguments ---
        max_seq_length=args.max_seq_length,  # Note: SFTConfig uses max_seq_length, not max_length
        packing=False,
    )

    # Add early stopping callback explicitly if needed (though load_best_model_at_end usually suffices)
    callbacks = []
    if args.evaluation_strategy != "no" and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    # --- Initialize SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=format_instruction,
        peft_config=peft_config,
        callbacks=callbacks,
        args=training_args,
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- Save Model ---
    final_output_dir = os.path.join(args.output_dir, "final_checkpoint")
    print(f"Saving final adapter model to {final_output_dir}")
    trainer.model.save_pretrained(final_output_dir)  # Saves only the adapter weights
    tokenizer.save_pretrained(final_output_dir)
    print("Model and tokenizer saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM using SFTTrainer and QLoRA"
    )

    # Model/Tokenizer Args
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-12b-it",
        help="Hugging Face model identifier",
    )

    # Dataset Args
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/train/train.json",
        help="Path to the training dataset JSON file",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="./data/test/test.json",
        help="Path to the evaluation dataset JSON file",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for training",
    )

    # QLoRA Args
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bfloat16",
        help="Compute dtype for 4-bit base models (e.g., float16, bfloat16)",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        help="Quantization type (fp4 or nf4)",
    )
    parser.add_argument(
        "--use_nested_quant",
        action="store_true",
        help="Activate nested quantization for 4-bit base models",
    )

    # PEFT Args
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="LoRA dropout probability"
    )
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")

    # Training Args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=1.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of steps to accumulate gradients",
    )  # Effective batch size = batch_size * accumulation_steps
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Optimizer (e.g., paged_adamw_8bit, adamw_hf)",
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log metrics every X steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Enable bf16 training (recommended if supported)",
    )  # Defaulting to True as it's often better for LLMs if hardware supports it
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.3,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Total number of training steps (overrides epochs)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type (e.g., linear, cosine)",
    )

    # Evaluation & Early Stopping Args
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy during training.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Run evaluation every N steps (if evaluation_strategy='steps').",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        default=True,
        help="Load the best model checkpoint found during training at the end.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluation steps with no improvement to wait before stopping (requires load_best_model_at_end=True).",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum improvement threshold for early stopping.",
    )

    args = parser.parse_args()

    # Basic validation
    if args.fp16 and args.bf16:
        raise ValueError("Cannot enable both fp16 and bf16 training. Choose one.")
    if args.load_best_model_at_end and args.evaluation_strategy == "no":
        raise ValueError(
            "load_best_model_at_end requires an evaluation_strategy ('steps' or 'epoch')."
        )
    if args.evaluation_strategy == "steps" and args.eval_steps <= 0:
        raise ValueError(
            "eval_steps must be positive if evaluation_strategy is 'steps'."
        )

    main(args)
