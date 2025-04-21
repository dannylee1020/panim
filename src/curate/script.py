import json
import random
import math
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Define default paths relative to the project root where this script might be run from
DEFAULT_INPUT_DIR = Path("../../inst")
DEFAULT_OUTPUT_DIR = Path("../../data")

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

if __name__ == "__main__":
    print("Running data split script directly...")
    split_inst_data()
    print("Script finished.")
