#!/bin/bash

# --- Script to run the SFTTrainer fine-tuning ---

# --- Environment Variables (Optional - for W&B) ---
# Uncomment and set these if using W&B and not logged in via CLI
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"
# export WANDB_PROJECT="panim"
# export WANDB_NAME="experiment-$(date +%s)" # Example run name

# --- Training Arguments ---
# Adjust these parameters as needed for your training run

# Model/Data Args
MODEL_NAME="google/gemma-3-12b-it"
DATASET_PATH="./inst/"
MAX_SEQ_LENGTH=2048

# QLoRA Args
BNB_COMPUTE_DTYPE="bfloat16" # Or "float16" for T4 GPUs
BNB_QUANT_TYPE="nf4"
# USE_NESTED_QUANT="--use_nested_quant" # Uncomment to enable

# PEFT Args
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_R=64

# Training Args
OUTPUT_DIR="./results/$(date +%Y%m%d_%H%M%S)" # Unique output dir per run
NUM_TRAIN_EPOCHS=1.0
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
OPTIM="paged_adamw_8bit"
SAVE_STEPS=500 # Save checkpoints every N steps
LOGGING_STEPS=10 # Log metrics every N steps
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.001
# FP16="--fp16" # Uncomment if using fp16 (e.g., on T4 GPU)
BF16="--bf16" # Default in script is True, uncomment this line or the one above if you want to be explicit or change it
MAX_GRAD_NORM=0.3
MAX_STEPS=-1 # Set to positive integer to limit steps, overrides epochs
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"

# --- Execute Training Script ---
echo "Starting training run..."
python src/train/train.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --bnb_4bit_compute_dtype "$BNB_COMPUTE_DTYPE" \
    --bnb_4bit_quant_type "$BNB_QUANT_TYPE" \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_r $LORA_R \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --optim "$OPTIM" \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    # $FP16 \
    $BF16 \
    --max_grad_norm $MAX_GRAD_NORM \
    --max_steps $MAX_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"

echo "Training run finished. Results in $OUTPUT_DIR"
