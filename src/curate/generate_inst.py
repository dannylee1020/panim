import os
import sys
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import fire
import re # Added for year extraction
import prompt
from provider import Gemini

# --- Configuration ---
DOC_INPUT_BASE_DIR = Path("../build/extract") # Renamed for clarity
CODE_INPUT_BASE_DIR = Path("../source_code")
OUTPUT_BASE_DIR = Path("../inst")
CODE_OUTPUT_BASE_DIR = OUTPUT_BASE_DIR / "source_code" # Specific output for code instructions
REFERENCE_DIR_NAME = "reference" # Define constant for the special directory
REFERENCE_OUTPUT_FILENAME = "reference.json" # Define constant for the consolidated output file
MIN_YEAR_CODE = 2020 # Requirement: Only process code from 2020 onwards

def _generate_instructions_for_file(input_txt_path: Path) -> list | None:
    """
    Reads a text file and generates instruction/answer pairs using the LLM provider.

    Args:
        input_txt_path: Path to the input .txt file.
    Returns:
        A list of instruction/answer pair dictionaries on success,
        None on failure or if the file was skipped.
    """
    try:
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            input_text = f.read()

        if not input_text.strip():
            print(f"    Warning: Input file {input_txt_path} is empty. Skipping.")
            return None

        # Call LLM
        llm = Gemini()
        response = llm.generate(message=prompt.generate_inst_from_doc.format(input_text=input_text))

        # Attempt to extract and parse the JSON part of the response
        try:
            # Adapt based on actual response structure from LLM provider
            response_text = response.text
            json_text = response_text.strip().strip('```json').strip('```').strip()
            instruction_pairs = json.loads(json_text)

            # Basic validation
            if not isinstance(instruction_pairs, list):
                raise ValueError("Generated output is not a JSON list.")
            for pair in instruction_pairs:
                if not isinstance(pair, dict) or "instruction" not in pair or "answer" not in pair:
                    raise ValueError("Generated list contains invalid pair objects.")

            return instruction_pairs

        except json.JSONDecodeError as e:
            print(f"    Error: Failed to decode JSON response for {input_txt_path.name}.", file=sys.stderr)
            return None
        except ValueError as e:
             print(f"    Error: Invalid JSON structure response for {input_txt_path.name}: {e}", file=sys.stderr)
             return None
        except KeyError as e:
             print(f"    Error: Unexpected response structure (missing key {e}) for {input_txt_path.name}.", file=sys.stderr)
             return None
        except Exception as e:
            # Catch other potential issues with response structure
            print(f"    Error processing response for {input_txt_path.name}: {e}", file=sys.stderr)
            return None

    except FileNotFoundError:
        print(f"  Error: Input file not found: {input_txt_path}", file=sys.stderr)
        return None
    except Exception as e:
        # Log exceptions during file processing
        print(f"  Error processing file {input_txt_path.name}: {e}", file=sys.stderr)
        return None

# --- Generate Instructions from Code File ---
def _generate_instructions_for_code_file(input_py_path: Path) -> list | None:
    """
    Reads a Python source code file and generates instruction/answer pairs using the LLM provider.

    Args:
        input_py_path: Path to the input .py file.
    Returns:
        A list of instruction/answer pair dictionaries on success,
        None on failure or if the file was skipped.
    """
    try:
        with open(input_py_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        if not source_code.strip():
            print(f"    Warning: Input file {input_py_path} is empty. Skipping.")
            return None

        # Call LLM
        llm = Gemini()
        # Use the specific prompt for code
        response = llm.generate(message=prompt.generate_inst_from_code.format(source_code=source_code))

        # Attempt to extract and parse the JSON part of the response
        try:
            # Adapt based on actual response structure from LLM provider
            response_text = response.text
            # Handle potential markdown code block fences
            json_text = response_text.strip().strip('```json').strip('```').strip()
            instruction_pairs = json.loads(json_text)

            # Basic validation
            if not isinstance(instruction_pairs, list):
                raise ValueError("Generated output is not a JSON list.")
            for pair in instruction_pairs:
                if not isinstance(pair, dict) or "instruction" not in pair or "answer" not in pair:
                    raise ValueError("Generated list contains invalid pair objects.")

            return instruction_pairs

        except json.JSONDecodeError as e:
            print(f"    Error: Failed to decode JSON response for {input_py_path.name}.", file=sys.stderr)
            return None
        except ValueError as e:
             print(f"    Error: Invalid JSON structure response for {input_py_path.name}: {e}", file=sys.stderr)
             return None
        except KeyError as e:
             print(f"    Error: Unexpected response structure (missing key {e}) for {input_py_path.name}.", file=sys.stderr)
             return None
        except Exception as e:
            # Catch other potential issues with response structure
            print(f"    Error processing response for {input_py_path.name}: {e}", file=sys.stderr)
            return None

    except FileNotFoundError:
        print(f"  Error: Input file not found: {input_py_path}", file=sys.stderr)
        return None
    except Exception as e:
        # Log exceptions during file processing
        print(f"  Error processing file {input_py_path.name}: {e}", file=sys.stderr)
        return None


# --- Worker Function for Processing Doc Files Concurrently within a Directory ---
def _process_directory_concurrently(dir_name, summary_lock, summary_stats, reference_lock):
    """
    Processes a single directory, generating instruction pairs for its .txt files
    concurrently using a thread pool.

    Args:
        dir_name (str): The name of the directory under DOC_INPUT_BASE_DIR.
        summary_lock (threading.Lock): Lock for updating shared summary_stats.
        summary_stats (dict): Dictionary holding overall processing statistics.
        reference_lock (threading.Lock): Lock for accessing the consolidated reference file.
    """
    is_reference_dir = (dir_name == REFERENCE_DIR_NAME)
    reference_output_dir = OUTPUT_BASE_DIR / REFERENCE_DIR_NAME
    reference_output_path = reference_output_dir / REFERENCE_OUTPUT_FILENAME
    input_dir = DOC_INPUT_BASE_DIR / dir_name # Use renamed constant
    output_dir = OUTPUT_BASE_DIR / dir_name

    print(f"\nProcessing doc directory: {dir_name}")

    if not input_dir.is_dir():
        print(f"  Warning: Input doc directory '{input_dir}' not found. Skipping.", file=sys.stderr)
        return

    # Create output dir only if not reference
    if not is_reference_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Input:  {input_dir}")
    if is_reference_dir:
         print(f"  Output: Appending to {reference_output_path}")
    else:
         print(f"  Output: {output_dir}")

    # --- Find files ---
    txt_files = list(input_dir.glob("*.txt"))
    current_dir_txt_found = len(txt_files)

    if not txt_files:
        print(f"  No .txt files found in {input_dir}.")
        with summary_lock:
            summary_stats['total_txt_files_found'] += current_dir_txt_found
        return

    # --- Process files concurrently ---
    current_dir_processed = 0
    current_dir_succeeded = 0
    current_dir_failed = 0
    reference_pairs_to_append = [] # Collect pairs for reference dir before writing

    # Using ThreadPoolExecutor to manage threads for file processing
    num_workers = min(10, os.cpu_count() + 4 if os.cpu_count() else 8)
    print(f"  Processing {len(txt_files)} files concurrently using {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a dictionary to map futures to their input paths
        future_to_path = {executor.submit(_generate_instructions_for_file, txt_path): txt_path for txt_path in txt_files}

        for future in as_completed(future_to_path):
            txt_path = future_to_path[future]
            current_dir_processed += 1
            try:
                instruction_pairs = future.result() # Get result from the completed future

                if instruction_pairs is not None:
                    if is_reference_dir:
                        # Collect pairs for later batch append
                        reference_pairs_to_append.extend(instruction_pairs)
                        # We count success here, but actual file write happens later
                        current_dir_succeeded += 1
                    else:
                        # Write individual file for non-reference directories immediately
                        output_path = output_dir / txt_path.with_suffix(".json").name
                        try:
                            with open(output_path, 'w', encoding='utf-8') as f_out:
                                json.dump(instruction_pairs, f_out, indent=2, ensure_ascii=False)
                            current_dir_succeeded += 1
                        except Exception as e:
                            print(f"    Error writing output file {output_path}: {e}", file=sys.stderr)
                            current_dir_failed += 1 # Count as failed if write fails
                else:
                    current_dir_failed += 1
            except Exception as exc:
                print(f'    Error processing file {txt_path.name}: {exc}', file=sys.stderr)
                current_dir_failed += 1

    # --- Handle Reference Directory Output (after all files in this dir are processed) ---
    if is_reference_dir and reference_pairs_to_append:
        print(f"  Appending {len(reference_pairs_to_append)} total pairs to reference file: {reference_output_path}")
        try:
            with reference_lock:
                # Ensure directory exists (might be redundant but safe)
                reference_output_dir.mkdir(parents=True, exist_ok=True)
                # Read existing data or initialize
                existing_pairs = []
                if reference_output_path.is_file():
                    try:
                        with open(reference_output_path, 'r', encoding='utf-8') as f_in:
                            content = f_in.read()
                            if content.strip(): # Check if file is not empty
                                existing_pairs = json.loads(content)
                                if not isinstance(existing_pairs, list):
                                    print(f"    Warning: Corrupted reference file {reference_output_path} (not a list). Re-initializing.", file=sys.stderr)
                                    existing_pairs = []
                            else:
                                existing_pairs = [] # Treat empty file as empty list
                    except json.JSONDecodeError:
                        print(f"    Warning: Corrupted reference file {reference_output_path} (invalid JSON). Re-initializing.", file=sys.stderr)
                        existing_pairs = []
                    except FileNotFoundError:
                         # Should not happen if is_file() passed, but handle defensively
                         existing_pairs = []

                # Append new pairs and write back
                existing_pairs.extend(reference_pairs_to_append)
                with open(reference_output_path, 'w', encoding='utf-8') as f_out:
                    json.dump(existing_pairs, f_out, indent=2, ensure_ascii=False)
            print(f"    -> Successfully appended pairs to {reference_output_path}")
            # Note: Success/failure counts were already updated when pairs were generated/collected
        except Exception as e:
            print(f"    Error appending batch to reference file {reference_output_path}: {e}", file=sys.stderr)
            # Adjust counts as saving failed despite generation success
            current_dir_succeeded -= len(reference_pairs_to_append) # Revert success count
            current_dir_failed += len(reference_pairs_to_append)    # Increment fail count

    print(f"  Finished processing {dir_name}: Found={current_dir_txt_found}, Attempted={current_dir_processed}, Succeeded={current_dir_succeeded}, Failed={current_dir_failed}")

    # --- Update shared summary statistics safely ---
    with summary_lock:
        summary_stats['total_files_processed'] += current_dir_processed
        summary_stats['total_files_succeeded'] += current_dir_succeeded
        summary_stats['total_files_failed'] += current_dir_failed
        summary_stats['total_txt_files_found'] += current_dir_txt_found


# --- Main Generate Function (Docs) ---
def generate_from_doc(target_dirs: str | tuple | list | None = None, target_file: str | None = None):
    """
    Generates instruction/answer pairs for .txt files extracted from documentation.
    Processes directories sequentially, but processes files within each directory concurrently.

    Processes either a specific file using --target_file or all .txt files
    within specified directories using --target_dirs. If neither is specified,
    processes all .txt files in all subdirectories under DOC_INPUT_BASE_DIR.

    Args:
        target_file (str | None): Path to a specific .txt file relative to
            DOC_INPUT_BASE_DIR to process (e.g., 'tutorials/quickstart.txt').
            If provided, this takes precedence over target_dirs.
            CLI Example: `python src/curate/generate_inst.py generate_from_doc --target_file='tutorials/quickstart.txt'`
        target_dirs (str | tuple | list | None): Directory name(s) under DOC_INPUT_BASE_DIR
            to process. Can be a single name, comma-separated string, or tuple/list
            (as parsed by `fire`). Ignored if `target_file` is provided.
            CLI Examples:
             - Process 'faq' and 'tutorials': `python src/curate/generate_inst.py generate_from_doc --target_dirs='faq,tutorials'`
             - Process only 'quickstart': `python src/curate/generate_inst.py generate_from_doc --target_dirs='quickstart'`
             - Process all (default): `python src/curate/generate_inst.py generate_from_doc`
    """
    # --- Handle target_file first (takes precedence) ---
    if target_file:
        if target_dirs:
            print("Warning: Both --target_file and --target_dirs provided. --target_file takes precedence.", file=sys.stderr)

        input_path = DOC_INPUT_BASE_DIR / target_file # Use renamed constant
        print(f"\nProcessing single target doc file: {input_path}")

        if not input_path.is_file():
            print(f"Error: Target doc file not found: {input_path}", file=sys.stderr)
            return
        if input_path.suffix.lower() != ".txt":
             print(f"Error: Target doc file must be a .txt file: {input_path}", file=sys.stderr)
             return

        # Construct output path preserving relative structure
        relative_path = Path(target_file)
        output_path = OUTPUT_BASE_DIR / relative_path.parent / relative_path.with_suffix(".json").name

        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

        # Ensure output directory exists for the single file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process single file directly (no concurrency needed)
        instruction_pairs = _generate_instructions_for_file(input_path)

        success = False
        if instruction_pairs is not None:
            try:
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    json.dump(instruction_pairs, f_out, indent=2, ensure_ascii=False)
                print(f"    -> Saved {len(instruction_pairs)} instruction pairs to {output_path}")
                success = True
            except Exception as e:
                print(f"    Error writing output file {output_path}: {e}", file=sys.stderr)

        print("\n--- Single Doc File Processing Summary ---")
        if success:
            print("Total files attempted:                   1")
            print("Total successfully generated:            1")
            print("Total failed:                            0")
            print("\nCompleted successfully.")
        else:
            print("Total files attempted:                   1")
            print("Total successfully generated:            0")
            print("Total failed:                            1")
            print("\nCompleted with errors.")
        return # Exit after processing single file

    # --- Determine Target Directories (only if target_file was not provided) ---
    actual_target_dirs = []
    if target_dirs is None:
        if DOC_INPUT_BASE_DIR.is_dir(): # Use renamed constant
            # Sort directories for consistent processing order
            actual_target_dirs = sorted([d.name for d in DOC_INPUT_BASE_DIR.iterdir() if d.is_dir()]) # Use renamed constant
        if not actual_target_dirs:
             print(f"Error: No subdirectories found in {DOC_INPUT_BASE_DIR}. Cannot determine default targets.", file=sys.stderr) # Use renamed constant
             return
        print(f"No target directories specified, defaulting to all found (in order): {actual_target_dirs}")
    elif isinstance(target_dirs, str):
        # Handle single string or comma-separated string input
        actual_target_dirs = sorted([d.strip() for d in target_dirs.split(',') if d.strip()])
        if not actual_target_dirs:
             print(f"Warning: Invalid or empty 'target_dirs' string provided: '{target_dirs}'. Exiting.", file=sys.stderr)
             return
    elif isinstance(target_dirs, (tuple, list)):
        # Handle tuple/list input (e.g., fire parsing 'a,b' into ('a', 'b'))
        actual_target_dirs = sorted([str(d).strip() for d in target_dirs if str(d).strip()])
        if not actual_target_dirs:
             print(f"Warning: Invalid or empty 'target_dirs' tuple/list provided: {target_dirs}. Exiting.", file=sys.stderr)
             return
    else:
        print(f"Warning: Invalid 'target_dirs' argument type: {type(target_dirs)}. Value: {target_dirs}. Exiting.", file=sys.stderr)
        return

    # --- Processing Setup (only runs if target_file was None) ---
    print(f"\nProcessing target doc directories sequentially: {actual_target_dirs}")
    print(f"Files within each directory will be processed concurrently.")

    # Shared resources and locks
    summary_lock = threading.Lock()
    summary_stats = {
        'total_files_processed': 0,
        'total_files_succeeded': 0,
        'total_files_failed': 0,
        'total_txt_files_found': 0
    }
    reference_lock = threading.Lock() # Lock for reference file access

    # --- Initialize/Clear Reference File (Optional but recommended for clean runs) ---
    # Check if the reference directory is among the targets before clearing
    if REFERENCE_DIR_NAME in actual_target_dirs:
        reference_output_dir = OUTPUT_BASE_DIR / REFERENCE_DIR_NAME
        reference_output_path = reference_output_dir / REFERENCE_OUTPUT_FILENAME
        print(f"\nEnsuring reference directory exists and clearing/initializing {reference_output_path} for this run...")
        try:
            reference_output_dir.mkdir(parents=True, exist_ok=True)
            # Initialize with an empty list
            with open(reference_output_path, 'w', encoding='utf-8') as f_init:
                json.dump([], f_init)
            print(f"  -> Initialized {reference_output_path}")
        except Exception as e:
            print(f"  Warning: Could not initialize reference file {reference_output_path}: {e}", file=sys.stderr)
            # Decide if this is fatal or just a warning. Proceeding for now.

    for dir_name in actual_target_dirs:
        _process_directory_concurrently(dir_name, summary_lock, summary_stats, reference_lock)
        print("-" * 40)


    print("\nAll target doc directory processing finished.")

    # --- Overall Summary ---
    total_txt_files_found = summary_stats['total_txt_files_found']
    total_files_processed = summary_stats['total_files_processed']
    total_files_succeeded = summary_stats['total_files_succeeded']
    total_files_failed = summary_stats['total_files_failed']

    print("\n--- Overall Doc Processing Summary ---")
    print(f"Total .txt files found across all targets: {total_txt_files_found}")
    print(f"Total files attempted:                   {total_files_processed}")
    print(f"Total successfully generated:            {total_files_succeeded}")
    print(f"Total failed:                            {total_files_failed}")

    if total_files_failed > 0:
        print("\nCompleted with errors.")
    else:
        print("\nCompleted successfully.")


# --- Main Generate Function (Code) ---
# TODO: needs evaluation
def generate_from_code(target_years: str | tuple | list | None = None):
    """
    Generates instruction/answer pairs from Manim source code files (.py).
    Processes only year directories (_YYYY) from 2020 onwards within CODE_INPUT_BASE_DIR.
    Processes files sequentially.

    Args:
        target_years (str | tuple | list | None): Specific year(s) (e.g., 2021, '2022,2023')
            to process. If None, processes all years >= MIN_YEAR_CODE found.
            Year format should be YYYY (e.g., 2020).
            CLI Examples:
             - Process 2022 and 2023: `python src/curate/generate_inst.py generate_from_code --target_years='2022,2023'`
             - Process only 2021: `python src/curate/generate_inst.py generate_from_code --target_years=2021`
             - Process all valid years (>=2020): `python src/curate/generate_inst.py generate_from_code`
    """
    print("\nStarting generation from source code...")
    print(f"Source directory: {CODE_INPUT_BASE_DIR}")
    print(f"Output directory: {CODE_OUTPUT_BASE_DIR}")
    print(f"Minimum year:     {MIN_YEAR_CODE}")

    # --- Determine Target Years ---
    all_year_dirs = []
    if CODE_INPUT_BASE_DIR.is_dir():
        # Find directories matching _YYYY pattern
        year_pattern = re.compile(r'^_(\d{4})$')
        for item in CODE_INPUT_BASE_DIR.iterdir():
            if item.is_dir():
                match = year_pattern.match(item.name)
                if match:
                    year = int(match.group(1))
                    if year >= MIN_YEAR_CODE:
                        all_year_dirs.append(item) # Store Path object

    if not all_year_dirs:
        print(f"Error: No year directories found in {CODE_INPUT_BASE_DIR} matching pattern _YYYY with year >= {MIN_YEAR_CODE}.", file=sys.stderr)
        return

    # Filter based on target_years argument
    actual_target_year_dirs = []
    if target_years is None:
        actual_target_year_dirs = sorted(all_year_dirs, key=lambda p: p.name)
        print(f"No target years specified, processing all found valid years: {[d.name for d in actual_target_year_dirs]}")
    else:
        requested_years = set()
        if isinstance(target_years, (int, str)):
            # Handle single int, single string 'YYYY', or comma-separated string 'YYYY,YYYY'
            years_str = str(target_years)
            for year_str in years_str.split(','):
                year_str = year_str.strip()
                if year_str.isdigit():
                    requested_years.add(int(year_str))
                else:
                    print(f"Warning: Invalid year format '{year_str}' in target_years. Skipping.", file=sys.stderr)
        elif isinstance(target_years, (tuple, list)):
             # Handle tuple/list input (e.g., fire parsing '2022,2023' into ('2022', '2023'))
             for year_item in target_years:
                 year_str = str(year_item).strip()
                 if year_str.isdigit():
                     requested_years.add(int(year_str))
                 else:
                     print(f"Warning: Invalid year format '{year_str}' in target_years. Skipping.", file=sys.stderr)
        else:
            print(f"Warning: Invalid 'target_years' argument type: {type(target_years)}. Value: {target_years}. Exiting.", file=sys.stderr)
            return

        if not requested_years:
             print("Warning: No valid target years provided or parsed. Exiting.", file=sys.stderr)
             return

        # Filter the found dirs by the requested years
        for year_dir in all_year_dirs:
            year = int(year_dir.name[1:]) # Extract year from '_YYYY'
            if year in requested_years:
                actual_target_year_dirs.append(year_dir)

        actual_target_year_dirs = sorted(actual_target_year_dirs, key=lambda p: p.name)
        print(f"Processing requested target years: {[d.name for d in actual_target_year_dirs]}")
        if len(actual_target_year_dirs) < len(requested_years):
             found_years_set = {int(d.name[1:]) for d in actual_target_year_dirs}
             missing_years = requested_years - found_years_set
             print(f"Warning: Requested years not found or invalid (>= {MIN_YEAR_CODE}): {sorted(list(missing_years))}", file=sys.stderr)


    if not actual_target_year_dirs:
        print("Error: No valid target year directories to process.", file=sys.stderr)
        return

    # --- Processing Setup ---
    summary_stats = {
        'total_py_files_found': 0,
        'total_files_processed': 0,
        'total_files_succeeded': 0,
        'total_files_failed': 0
    }

    # --- Process each target year directory ---
    for year_dir in actual_target_year_dirs:
        print(f"\nProcessing year directory: {year_dir.name}")
        # Find all .py files recursively within this year directory
        py_files = list(year_dir.rglob("*.py"))
        current_dir_py_found = len(py_files)
        summary_stats['total_py_files_found'] += current_dir_py_found

        if not py_files:
            print(f"  No .py files found in {year_dir}.")
            continue

        print(f"  Found {current_dir_py_found} .py files. Processing sequentially...")

        for input_py_path in py_files:
            print(f"    Processing file: {input_py_path.relative_to(CODE_INPUT_BASE_DIR)}")
            summary_stats['total_files_processed'] += 1

            # Construct output path mirroring the input structure
            relative_path = input_py_path.relative_to(CODE_INPUT_BASE_DIR)
            output_path = CODE_OUTPUT_BASE_DIR / relative_path.with_suffix(".json")

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate instructions
            instruction_pairs = _generate_instructions_for_code_file(input_py_path)

            if instruction_pairs is not None:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f_out:
                        json.dump(instruction_pairs, f_out, indent=2, ensure_ascii=False)
                    print(f"      -> Saved {len(instruction_pairs)} pairs to {output_path.relative_to(OUTPUT_BASE_DIR)}")
                    summary_stats['total_files_succeeded'] += 1
                except Exception as e:
                    print(f"      Error writing output file {output_path}: {e}", file=sys.stderr)
                    summary_stats['total_files_failed'] += 1
            else:
                # Failure already printed in helper function
                summary_stats['total_files_failed'] += 1
        print("-" * 40)


    print("\nAll target year directory processing finished.")

    # --- Overall Summary ---
    total_py_files_found = summary_stats['total_py_files_found']
    total_files_processed = summary_stats['total_files_processed']
    total_files_succeeded = summary_stats['total_files_succeeded']
    total_files_failed = summary_stats['total_files_failed']

    print("\n--- Overall Code Processing Summary ---")
    print(f"Total .py files found across target years: {total_py_files_found}")
    print(f"Total files attempted:                   {total_files_processed}")
    print(f"Total successfully generated:            {total_files_succeeded}")
    print(f"Total failed:                            {total_files_failed}")

    if total_files_failed > 0:
        print("\nCompleted with errors.")
    else:
        print("\nCompleted successfully.")


if __name__ == "__main__":
    fire.Fire({
        "generate_from_doc": generate_from_doc,
        "generate_from_code": generate_from_code
    })
