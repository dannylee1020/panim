import json
import random
import math
import os
import argparse
import numpy as np # Using numpy for average calculation
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer

# Define default paths relative to the project root
# Assuming script is run from project root or paths are adjusted accordingly
DEFAULT_INPUT_DIR = Path("./inst")
DEFAULT_OUTPUT_DIR = Path("./data")
DEFAULT_TRAIN_PATH = DEFAULT_OUTPUT_DIR / "train" / "train.json"
DEFAULT_TEST_PATH = DEFAULT_OUTPUT_DIR / "test" / "test.json"
DEFAULT_MODEL_NAME = "google/gemma-3-12b-it"

def split_inst_data(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> None:
    """
    Splits JSON data from subdirectories of input_dir into stratified
    train and test sets, saving them to output_dir.

    Assumes each .json file in input_dir contains a list of JSON objects.
    Performs a stratified split based on the immediate subdirectory
    of input_dir the JSON file resides in. Creates output files
    output_dir/train/train.json and output_dir/test/test.json.

    Args:
        input_dir: Path to the directory containing source data subdirectories
                   (e.g., 'inst/'). Defaults to ./inst.
        output_dir: Path to the directory where 'train/' and 'test/'
                    subdirectories will be created. Defaults to ./finished.
        test_ratio: The proportion of data to allocate to the test set (e.g., 0.1 for 10%).
        random_seed: Seed for the random number generator for reproducible splits.
    """
    random.seed(random_seed)
    print(f"Starting data split from '{input_dir}' to '{output_dir}'...")
    print(f"Test ratio: {test_ratio:.2f}, Random seed: {random_seed}")

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # --- 1. Group data by source subdirectory ---
    data_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_items = 0
    json_files = list(input_dir.rglob("*.json"))

    if not json_files:
        print(f"Error: No .json files found recursively in '{input_dir}'.")
        return

    print(f"Found {len(json_files)} JSON files. Reading and grouping data...")
    for json_file in json_files:
        # Determine the source directory relative to input_dir
        try:
            relative_path = json_file.relative_to(input_dir)
            # Use the first part of the relative path as the source key
            # If the file is directly in input_dir, use a default key like '__root__'
            source_key = relative_path.parts[0] if len(relative_path.parts) > 1 else "__root__"
        except ValueError:
            print(f"Warning: Could not determine relative path for {json_file}. Skipping.")
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    # Add source information to each item if needed later,
                    # but for splitting, just group them.
                    data_by_source[source_key].extend(content)
                    total_items += len(content)
                else:
                    print(f"Warning: Content of {json_file} is not a list. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {json_file}. Skipping.")
        except Exception as e:
            print(f"Warning: Error reading {json_file}: {e}. Skipping.")

    if not data_by_source or total_items == 0:
        print("Error: No valid data loaded from JSON files.")
        return

    print(f"Total items loaded: {total_items}")
    print(f"Sources found: {list(data_by_source.keys())}")

    # --- 2. Perform stratified split ---
    train_data: List[Dict[str, Any]] = []
    test_data: List[Dict[str, Any]] = []

    print("Performing stratified split...")
    for source, items in data_by_source.items():
        if not items:
            print(f"  Source '{source}' has no items. Skipping.")
            continue

        num_items = len(items)
        # Ensure at least one test sample if ratio > 0 and items exist
        num_test = math.ceil(num_items * test_ratio) if test_ratio > 0 else 0
        # Ensure num_test doesn't exceed total items (can happen with ceil and small lists)
        num_test = min(num_test, num_items)
        num_train = num_items - num_test

        print(f"  Source '{source}': Total={num_items}, Train={num_train}, Test={num_test}")

        # Shuffle items within the source group
        random.shuffle(items)

        # Split
        test_data.extend(items[:num_test])
        train_data.extend(items[num_test:])

    print(f"Split complete: Total Train={len(train_data)}, Total Test={len(test_data)}")

    # --- 3. Create output directories ---
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_output_dir = output_dir / "train"
    test_output_dir = output_dir / "test"

    try:
        train_output_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directories exist: '{train_output_dir}', '{test_output_dir}'")
    except OSError as e:
        print(f"Error: Could not create output directories: {e}")
        return

    # --- 4. Write output files ---
    train_output_path = train_output_dir / "train.json"
    test_output_path = test_output_dir / "test.json"

    try:
        print(f"Writing training data ({len(train_data)} items) to '{train_output_path}'...")
        with open(train_output_path, 'w', encoding='utf-8') as f_train:
            json.dump(train_data, f_train, indent=2, ensure_ascii=False)

        print(f"Writing test data ({len(test_data)} items) to '{test_output_path}'...")
        with open(test_output_path, 'w', encoding='utf-8') as f_test:
            json.dump(test_data, f_test, indent=2, ensure_ascii=False)

        print("Data successfully written.")
    except IOError as e:
        print(f"Error: Could not write output JSON files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")


# --- Token Statistics Calculation ---

def format_instruction_for_tokenization(sample: Dict[str, Any]) -> str | None:
    """Formats a sample for tokenization, matching the training script."""
    instruction = sample.get("instruction", "")
    answer = sample.get("answer", "")
    if instruction and answer:
        # Ensure consistent formatting with train.py
        return f"### Instruction:\n{instruction}\n\n### Answer:\n{answer}"
    else:
        print(f"Warning: Skipping sample due to missing instruction or answer: {sample}")
        return None

def calculate_token_stats(
    train_path: Path = DEFAULT_TRAIN_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """
    Calculates and prints the average and maximum token lengths for
    train and test datasets using the specified tokenizer.

    Args:
        train_path: Path to the training data JSON file.
        test_path: Path to the test data JSON file.
        model_name: Hugging Face model identifier for the tokenizer.
    """
    print(f"Calculating token stats using tokenizer: '{model_name}'")
    print(f"Train data: '{train_path}'")
    print(f"Test data: '{test_path}'")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Set padding token if needed (though not strictly necessary for length calculation)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer '{model_name}': {e}")
        return

    def get_stats_for_file(file_path: Path) -> Tuple[float, int] | None:
        """Helper function to calculate stats for a single JSON file."""
        if not file_path.is_file():
            print(f"Error: Data file not found: '{file_path}'")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Data in '{file_path}' is not a list.")
                return None
            if not data:
                print(f"Warning: Data file '{file_path}' is empty.")
                return 0.0, 0 # Return zero stats for empty file
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_path}'.")
            return None
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            return None

        token_lengths = []
        print(f"Processing {len(data)} samples from '{file_path}'...")
        processed_count = 0
        for i, sample in enumerate(data):
            formatted_text = format_instruction_for_tokenization(sample)
            if formatted_text:
                try:
                    tokens = tokenizer(formatted_text, add_special_tokens=True)['input_ids']
                    token_lengths.append(len(tokens))
                    processed_count += 1
                except Exception as e:
                    print(f"Warning: Error tokenizing sample {i} in {file_path}: {e}. Skipping.")
            # Optional: Add progress indicator for large files
            # if (i + 1) % 1000 == 0:
            #     print(f"  Processed {i + 1}/{len(data)} samples...")

        if not token_lengths:
            print(f"No valid samples found or tokenized in '{file_path}'.")
            return 0.0, 0

        avg_len = np.mean(token_lengths)
        max_len = np.max(token_lengths)
        print(f"Finished processing '{file_path}'. Found {processed_count} valid samples.")
        avg_len = np.mean(token_lengths)
        max_len = np.max(token_lengths)
        p90_len = np.percentile(token_lengths, 90)
        p95_len = np.percentile(token_lengths, 95)
        print(f"Finished processing '{file_path}'. Found {processed_count} valid samples.")
        return avg_len, max_len, p90_len, p95_len

    # Calculate stats for train data
    print("\n--- Training Data Stats ---")
    train_stats = get_stats_for_file(train_path)
    if train_stats:
        avg_train, max_train, p90_train, p95_train = train_stats
        print(f"Average Token Length (Train): {avg_train:.2f}")
        print(f"Maximum Token Length (Train): {max_train}")
        print(f"90th Percentile Length (Train): {p90_train:.2f}")
        print(f"95th Percentile Length (Train): {p95_train:.2f}")


    # Calculate stats for test data
    print("\n--- Test Data Stats ---")
    test_stats = get_stats_for_file(test_path)
    if test_stats:
        avg_test, max_test, p90_test, p95_test = test_stats
        print(f"Average Token Length (Test): {avg_test:.2f}")
        print(f"Maximum Token Length (Test): {max_test}")
        print(f"90th Percentile Length (Test): {p90_test:.2f}")
        print(f"95th Percentile Length (Test): {p95_test:.2f}")


    print("\nToken statistics calculation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing script for Panim project.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Subparser for 'split' command ---
    parser_split = subparsers.add_parser("split", help="Split instruction data into train/test sets.")
    parser_split.add_argument(
        "--input_dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing source data subdirectories (default: {DEFAULT_INPUT_DIR})"
    )
    parser_split.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for train/test splits (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser_split.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Proportion of data for the test set (default: 0.1)"
    )
    parser_split.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42)"
    )

    # --- Subparser for 'stats' command ---
    parser_stats = subparsers.add_parser("stats", help="Calculate token statistics for train/test data.")
    parser_stats.add_argument(
        "--train_path", type=Path, default=DEFAULT_TRAIN_PATH,
        help=f"Path to the training data JSON file (default: {DEFAULT_TRAIN_PATH})"
    )
    parser_stats.add_argument(
        "--test_path", type=Path, default=DEFAULT_TEST_PATH,
        help=f"Path to the test data JSON file (default: {DEFAULT_TEST_PATH})"
    )
    parser_stats.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model identifier for tokenizer (default: {DEFAULT_MODEL_NAME})"
    )

    args = parser.parse_args()

    if args.command == "split":
        print("Running data split...")
        split_inst_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
        )
        print("Data split finished.")
    elif args.command == "stats":
        print("Running token statistics calculation...")
        # Ensure paths are resolved correctly if script is run from project root
        calculate_token_stats(
            train_path=args.train_path.resolve(),
            test_path=args.test_path.resolve(),
            model_name=args.model_name
        )
        print("Token statistics calculation finished.")
    else:
        print("No command specified. Use 'split' or 'stats'. Try --help for options.")
        # Optionally run a default command or show help
        # parser.print_help()
