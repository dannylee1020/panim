import argparse
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from model import GenerateRequest, GenerateResponse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_cli_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FastAPI Inference Server for Fine-tuned LLM"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="google/gemma-3-12b-it",
        help="Base Hugging Face model identifier",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="../results/final_checkpoint",
        help="Path to the trained PEFT adapter checkpoint",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to run the server on",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    # Add generation parameters if needed
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Generation temperature"
    )
    parser.add_argument(
        "--do_sample", action="store_true", default=True, help="Enable sampling"
    )
    return parser.parse_args()


# --- Configuration ---
cli_args = parse_cli_args()
model = None
tokenizer = None


def load_model_and_tokenizer():
    global model, tokenizer
    print(f"Loading base model: {cli_args.base_model_name}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            cli_args.base_model_name,
            torch_dtype=torch.bfloat16,  # Or float16, adjust based on hardware
            device_map="auto",
            trust_remote_code=True,
        )
        print("Base model loaded.")

        print(f"Loading PEFT adapter from: {cli_args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, cli_args.adapter_path)
        model = (
            model.merge_and_unload()
        )  # Merge adapter into the base model for faster inference
        model.eval()
        print("PEFT adapter loaded and merged.")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cli_args.base_model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        print("Tokenizer loaded.")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise RuntimeError(f"Failed to load model/tokenizer: {e}")


# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model_and_tokenizer()
        print("Model and tokenizer loaded successfully.")
        yield
    except Exception as e:
        print(f"FATAL: Could not load model on startup: {e}")


app = FastAPI(lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server might be starting up or encountered an error.",
        )

    if not request.instruction:
        raise HTTPException(status_code=400, detail="Instruction cannot be empty.")

    # Format the prompt exactly like in training
    prompt = f"### Instruction:\n{request.instruction}\n\n### Answer:\n"

    try:
        print(
            f"Received instruction: {request.instruction[:100]}..."
        )  # Log truncated instruction
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cli_args.max_new_tokens,
                temperature=cli_args.temperature,
                do_sample=cli_args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Generated answer: {answer[:100]}...")  # Log truncated answer
        return GenerateResponse(answer=answer.strip())

    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error during generation: {e}"
        )


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok" if model and tokenizer else "error",
        "model_loaded": bool(model and tokenizer),
    }


if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
