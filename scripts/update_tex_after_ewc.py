#!/usr/bin/env python3
"""
Update LaTeX paper with EWC results after grid search.
"""

import argparse
import re
import os
from typing import Tuple

def find_ewc_table(tex_content: str) -> Tuple[int, int, str]:
    """Find the EWC results table in the LaTeX content."""
    # Look for the table with EWC results
    table_pattern = r'\\begin\{table\}\[h\].*?\\caption\{Final average accuracy.*?EWC.*?\*.*?\\end\{table\}'
    match = re.search(table_pattern, tex_content, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find EWC results table")
    
    start_pos = match.start()
    end_pos = match.end()
    table_content = match.group(0)
    
    return start_pos, end_pos, table_content

def update_ewc_value(table_content: str, new_ewc_value: float) -> str:
    """Update the EWC value in the table."""
    # Replace the EWC accuracy value
    updated_table = re.sub(
        r'(EWC \(tuned\)\s*&\s*)\d+\.\d+\*',
        rf'\g<1>{new_ewc_value:.1f}',
        table_content
    )
    
    # Remove the asterisk note
    updated_table = re.sub(
        r'\\\*EWC is a literature value in this setup and will be replaced by our in-protocol run\.',
        'EWC is our in-protocol implementation.',
        updated_table
    )
    
    return updated_table

def update_tex_file(tex_path: str, new_ewc_value: float, backup: bool = True):
    """Update the LaTeX file with new EWC results."""
    # Read the file
    with open(tex_path, 'r') as f:
        content = f.read()
    
    # Create backup if requested
    if backup:
        backup_path = tex_path + '.backup'
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Backup created: {backup_path}")
    
    # Find and update the table
    start_pos, end_pos, old_table = find_ewc_table(content)
    new_table = update_ewc_value(old_table, new_ewc_value)
    
    # Replace the table in the content
    updated_content = content[:start_pos] + new_table + content[end_pos:]
    
    # Write the updated content
    with open(tex_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated {tex_path} with EWC value: {new_ewc_value:.1f}%")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Update LaTeX paper with EWC results')
    parser.add_argument('--tex', type=str, required=True, help='Path to LaTeX file')
    parser.add_argument('--ewc', type=float, required=True, help='New EWC final average accuracy')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.tex):
        print(f"Error: LaTeX file not found: {args.tex}")
        return
    
    try:
        update_tex_file(args.tex, args.ewc, backup=not args.no_backup)
        print(f"\nSuccessfully updated {args.tex}")
        print("You can now rebuild the PDF with: make pdf")
        
    except Exception as e:
        print(f"Error updating LaTeX file: {e}")
        return

if __name__ == "__main__":
    main()
