import sys
import re
import os
from pathlib import Path
from bs4 import BeautifulSoup
import fire

def format_text(text):
    """Cleans up extracted text by normalizing whitespace and removing blank lines"""
    # Split into lines and strip whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    # Remove empty lines
    non_empty_lines = [line for line in lines if line]
    # Replace multiple spaces within lines with a single space
    cleaned_lines = [re.sub(r'\s+', ' ', line) for line in non_empty_lines]
    # Remove the ¶ symbol
    cleaned_lines = [line.replace('¶', '') for line in cleaned_lines]
    # Re-filter empty lines that might have been created by removing ¶
    cleaned_lines = [line.strip() for line in cleaned_lines if line.strip()]
    # Join lines back with a single newline
    return '\n'.join(cleaned_lines)

def extract_text_from_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the main content area, specifically targeting <article role="main">
        main_content = soup.find('article', role='main', id='furo-main-content')

        if not main_content:
            print(f"Warning: Could not find <article role='main'> in {file_path}. No text extracted.", file=sys.stderr)
            return None

        # Remove script and style tags first
        for script_or_style in main_content(['script', 'style']):
            script_or_style.decompose()

        # Find all code blocks (pre tags or divs with highlight class)
        code_blocks = main_content.find_all(
            ['pre', lambda tag: tag.name == 'div' and tag.get('class') and any(c.startswith('highlight-') for c in tag.get('class'))]
        )

        # --- Handle Block Code ---
        # Store original block code, wrapped in fences, and replace with placeholders
        original_block_code = {}
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_content = block.get_text().strip() # Get raw code text

            # Try to detect language from class names like 'highlight-python'
            language = ''
            highlight_div = block.find_parent('div', class_=lambda x: x and x.startswith('highlight-'))
            if highlight_div:
                lang_class = next((c for c in highlight_div['class'] if c.startswith('highlight-')), None)
                if lang_class:
                    language = lang_class.split('-')[-1] # e.g., 'python'

            # Store the block code wrapped in Markdown fences
            original_block_code[placeholder] = f"\n```{language}\n{code_content}\n```\n"

            # Replace the entire block with just the placeholder text
            block.replace_with(soup.new_string(placeholder))

        # --- Handle Inline Code ---
        # Find inline code tags (like <code class="docutils literal notranslate"><span class="pre">...</span></code>)
        # We target the outer <code> tag with the specific classes.
        inline_code_tags = main_content.find_all('code', class_='docutils literal notranslate')

        original_inline_code = {}
        for i, tag in enumerate(inline_code_tags):
            # Check if it's inside a block code placeholder's parent (already handled)
            if tag.find_parent(['pre', lambda t: t.name == 'div' and t.get('class') and any(c.startswith('highlight-') for c in t.get('class'))]):
                continue # Skip if it's part of a larger code block

            placeholder = f"__INLINE_CODE_{i}__"
            # Extract text, strip extra whitespace often found in spans
            inline_content = ' '.join(tag.get_text().split())
            if inline_content:
                 # Store the inline code wrapped in backticks
                original_inline_code[placeholder] = f"`{inline_content}`"
                # Replace the tag with the placeholder
                tag.replace_with(soup.new_string(placeholder))
            else:
                 # If empty, just remove the tag
                 tag.decompose()


        # Extract text from the modified soup (with placeholders for both block and inline code)
        text_with_placeholders = main_content.get_text(separator='\n', strip=True)

        # Format the main text (placeholders should survive this)
        formatted_text_with_placeholders = format_text(text_with_placeholders)

        # Replace block code placeholders first
        text_with_block_code = formatted_text_with_placeholders
        for placeholder, fenced_code in original_block_code.items():
            text_with_block_code = text_with_block_code.replace(placeholder, fenced_code)

        # Then replace inline code placeholders
        final_text = text_with_block_code
        for placeholder, backticked_code in original_inline_code.items():
            final_text = final_text.replace(placeholder, backticked_code)

        return final_text # Return the fully processed text

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        return None

def extract(targets=None):
    """
    Extracts text content from HTML files in specified subdirectories of build/html
    and saves them as .txt files in corresponding subdirectories under build/extract.

    Args:
        targets (tuple[str] | str | None): A tuple/list of directory names under build/html
            to process (e.g., ('faq', 'tutorials')). Can also be a single string for one directory.
            If None or empty, processes the default set: ["faq", "guides", "installation", "reference", "tutorials"].
    """
    default_targets = ["faq", "guides", "installation", "reference", "tutorials"]

    if targets is None:
        target_dirs = default_targets
    elif isinstance(targets, str):
        target_dirs = [targets] # Handle single string input
    elif isinstance(targets, (list, tuple)) and len(targets) > 0:
        target_dirs = list(targets) # Use provided list/tuple
    else:
        print("Warning: Invalid 'targets' argument provided. Using default targets.", file=sys.stderr)
        target_dirs = default_targets

    print(f"Processing target directories: {target_dirs}")

    target_base_dir = Path('../build/html')
    output_base_dir = Path('../build/extract')

    total_html_files_found = 0
    total_files_processed = 0
    total_files_failed = 0

    for dir_name in target_dirs:
        input_dir = target_base_dir / dir_name
        output_dir = output_base_dir / dir_name

        if not input_dir.is_dir():
            print(f"Warning: Input directory {input_dir} not found. Skipping.", file=sys.stderr)
            continue

        # Create the corresponding output directory structure
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing HTML files from: {input_dir}")
        print(f"Saving TXT files to:    {output_dir}")

        current_dir_files_found = 0
        current_dir_files_processed = 0
        current_dir_files_failed = 0

        # Recursively find all .html files in the current target input directory
        for html_path in input_dir.rglob('*.html'):
            # Skip index.html files
            if html_path.name == 'index.html':
                print(f"  Skipping index file: {html_path}")
                continue

            current_dir_files_found += 1
            print(f"  Processing: {html_path}")

            # Determine the relative path from the current input directory
            try:
                relative_path = html_path.relative_to(input_dir)
            except ValueError:
                # This case should ideally not happen with rglob if input_dir exists, but safety first
                print(f"    Warning: File {html_path} is not under {input_dir}? Skipping.", file=sys.stderr)
                current_dir_files_failed += 1
                continue

            # Construct the output path by joining the current output dir with the relative path
            # and changing the suffix to .txt
            output_path = output_dir / relative_path.with_suffix('.txt')

            # Ensure the parent directory for the output file exists (needed for nested structures)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract text using the function
            extracted_text = extract_text_from_html(str(html_path)) # Pass path as string

            if extracted_text is not None:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(extracted_text)
                    # print(f"    -> Saved to: {output_path}") # Optional: reduce verbosity
                    current_dir_files_processed += 1
                except Exception as e:
                    print(f"    Error writing file {output_path}: {e}", file=sys.stderr)
                    current_dir_files_failed += 1
            else:
                # Error message already printed by extract_text_from_html
                print(f"    -> Failed to extract text from {html_path}")
                current_dir_files_failed += 1

        print(f"  Finished processing {dir_name}: Found={current_dir_files_found}, Processed={current_dir_files_processed}, Failed={current_dir_files_failed}")
        total_html_files_found += current_dir_files_found
        total_files_processed += current_dir_files_processed
        total_files_failed += current_dir_files_failed

    print("\nOverall processing complete.")
    print(f"Total HTML files found across all targets: {total_html_files_found}")
    print(f"Total successfully processed:              {total_files_processed}")
    print(f"Total failed:                              {total_files_failed}")

    if total_files_failed > 0:
        print("\nCompleted with errors.", file=sys.stderr)
        sys.exit(1) # Exit with error code if failures occurred
    else:
        print("\nCompleted successfully.")

if __name__ == "__main__":
    fire.Fire(extract)
