#!/bin/bash

# Initialize variables
ADAPTER_PATH="../results/final_checkpoint"
BASE_MODEL_NAME="google/gemma-3-12b-it"
HOST="0.0.0.0"
PORT="8000"
MAX_NEW_TOKENS="1024"
TEMPERATURE="0.7"
DO_SAMPLE=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --adapter_path) ADAPTER_PATH="$2"; shift ;;
        --base_model_name) BASE_MODEL_NAME="$2"; shift ;;
        --host) HOST="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --do_sample) DO_SAMPLE="--do_sample" ;; # Flag argument
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


COMMAND="python inference/api.py"

# Add arguments if provided
if [ -n "$ADAPTER_PATH" ]; then
  COMMAND="$COMMAND --adapter_path $ADAPTER_PATH"
fi
if [ -n "$BASE_MODEL_NAME" ]; then
  COMMAND="$COMMAND --base_model_name $BASE_MODEL_NAME"
fi
if [ -n "$HOST" ]; then
  COMMAND="$COMMAND --host $HOST"
fi
if [ -n "$PORT" ]; then
  COMMAND="$COMMAND --port $PORT"
fi
if [ -n "$MAX_NEW_TOKENS" ]; then
  COMMAND="$COMMAND --max_new_tokens $MAX_NEW_TOKENS"
fi
if [ -n "$TEMPERATURE" ]; then
  COMMAND="$COMMAND --temperature $TEMPERATURE"
fi
if [ -n "$DO_SAMPLE" ]; then
  COMMAND="$COMMAND $DO_SAMPLE"
fi


echo "Starting inference server with command: $COMMAND"
exec $COMMAND
