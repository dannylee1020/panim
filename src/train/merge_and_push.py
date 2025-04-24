import os

import fire
import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_push_to_hub(
    base_model_name: str = "google/gemma-3-12b-it",
    adapter_path: str = "./results/final_checkpoint",
    hub_model_id: str = None,
    hub_private_repo: bool = False,
    torch_dtype_str: str = "bfloat16",
):
    """
    Loads a base model and a PEFT adapter, merges them, and pushes the merged
    model and tokenizer to the Hugging Face Hub.

    Args:
        base_model_name (str): Identifier of the base model on Hugging Face Hub.
        adapter_path (str): Path to the directory containing the saved PEFT adapter.
        hub_model_id (str): Repository ID on the Hub to push to (e.g., 'your-username/your-model-name'). Required.
        hub_token (str, optional): Hugging Face API token. Uses cached login or HF_TOKEN env var if None.
        hub_private_repo (bool, optional): Whether to make the Hub repository private. Defaults to False.
        torch_dtype_str (str, optional): The torch dtype to load the model in ('bfloat16', 'float16', 'float32'). Defaults to 'bfloat16'.
    """
    if not hub_model_id:
        raise ValueError("`hub_model_id` is required to push the model.")
    if not os.path.isdir(adapter_path):
        raise ValueError(f"Adapter path not found or not a directory: {adapter_path}")

    # --- Login to Hugging Face Hub ---
    try:
        login()
        print("Logged into Hugging Face Hub (using cached credentials or HF_TOKEN).")
    except Exception as e:
        print(f"Error logging into Hugging Face Hub: {e}")
        print(
            "Please ensure you are logged in via `huggingface-cli login` or have HF_TOKEN set."
        )
        return

    # --- Determine torch dtype ---
    try:
        torch_dtype = getattr(torch, torch_dtype_str)
        print(f"Using torch dtype: {torch_dtype}")
    except AttributeError:
        print(
            f"Error: Invalid torch_dtype_str '{torch_dtype_str}'. Choose from 'bfloat16', 'float16', 'float32'."
        )
        return

    # --- Load Base Model ---
    print(f"Loading base model: {base_model_name} with dtype {torch_dtype}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Base model loaded.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # --- Load PEFT Adapter ---
    print(f"Loading PEFT adapter from: {adapter_path}")
    try:
        # Load the PEFT model without merging initially
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("PEFT adapter loaded.")
    except Exception as e:
        print(f"Error loading PEFT adapter: {e}")
        return

    # --- Merge Adapter and Base Model ---
    print("Merging adapter weights into the base model...")
    try:
        # Ensure the model is on CPU for merging if memory is tight, or keep on device_map='auto'
        # Merging can be memory intensive. If you encounter OOM, try device_map='cpu'
        merged_model = model.merge_and_unload()
        print("Model merged successfully.")
    except Exception as e:
        print(f"Error merging model: {e}")
        print(
            "Merging can be memory-intensive. Consider using a machine with more RAM or trying device_map='cpu' during base model loading."
        )
        return

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for base model: {base_model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        # Set padding token if needed (important for consistency)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Often recommended for generation
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # --- Push Merged Model and Tokenizer to Hub ---
    print(f"Pushing merged model and tokenizer to Hub repo: {hub_model_id}")
    try:
        merged_model.push_to_hub(hub_model_id, private=hub_private_repo)
        tokenizer.push_to_hub(hub_model_id, private=hub_private_repo)
        print(f"Successfully pushed merged model and tokenizer to {hub_model_id}")

        # Optional: Create a model card (basic example)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=f"Merged version of {base_model_name} fine-tuned with adapter from {adapter_path}.".encode(),
            path_in_repo="README.md",
            repo_id=hub_model_id,
            repo_type="model",
        )
        print("Uploaded a basic README.md to the repository.")

    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")


if __name__ == "__main__":
    fire.Fire(merge_and_push_to_hub)
