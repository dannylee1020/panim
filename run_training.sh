#!/bin/bash

# Model/Data Args
MODEL_NAME="google/gemma-3-12b-it"
DATASET_PATH="./data/train/train.json" # Default path to the training JSON file
EVAL_DATASET_PATH="./data/test/test.json" # Default path to the evaluation JSON file
MAX_SEQ_LENGTH=1024

# QLoRA Args
BNB_COMPUTE_DTYPE="bfloat16"
BNB_QUANT_TYPE="nf4"

# PEFT Args
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_R=32

# Training Args
OUTPUT_DIR="./results/$(date +%Y%m%d_%H%M%S)" # Unique output dir per run
NUM_TRAIN_EPOCHS=20.0
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
OPTIM="paged_adamw_8bit"
SAVE_STEPS=50 # Save checkpoints every N steps
LOGGING_STEPS=10 # Log metrics every N steps
LEARNING_RATE=3e-5
WEIGHT_DECAY=0.001
FP16_FLAG="" # Set to "--fp16" to enable, leave empty to disable
BF16_FLAG="--bf16" # Set to "--bf16" to enable, leave empty to disable (Script default is True for bf16)
MAX_GRAD_NORM=0.5
MAX_STEPS=-1 # Set to positive integer to limit steps, overrides epochs
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"

# Evaluation & Early Stopping Args
EVALUATION_STRATEGY="steps"
EVAL_STEPS=50
LOAD_BEST_MODEL_AT_END_FLAG="--load_best_model_at_end"
EARLY_STOPPING_PATIENCE=3 #
EARLY_STOPPING_THRESHOLD=0.0

# --- Execute Training Script ---
# Note: Use 'accelerate launch' instead of 'python' for multi-GPU training
echo "Starting training run..."
python src/train/train.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --eval_dataset_path "$EVAL_DATASET_PATH" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --bnb_4bit_compute_dtype "$BNB_COMPUTE_DTYPE" \
    --bnb_4bit_quant_type "$BNB_QUANT_TYPE" \
    $USE_NESTED_QUANT \
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
    $FP16_FLAG \
    $BF16_FLAG \
    --max_grad_norm $MAX_GRAD_NORM \
    --max_steps $MAX_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --evaluation_strategy "$EVALUATION_STRATEGY" \
    --eval_steps $EVAL_STEPS \
    $LOAD_BEST_MODEL_AT_END_FLAG \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --early_stopping_threshold $EARLY_STOPPING_THRESHOLD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training run finished successfully. Results in $OUTPUT_DIR"
else
    echo "Training run failed with exit code $EXIT_CODE."
fi
