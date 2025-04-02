#!/usr/bin/env python3
"""
Script to run all experiments for Problems 3a and 3b.
"""

import subprocess
import time
from datetime import datetime

def run_command(command):
    """Run a command and print execution details."""
    print(f"\n=== RUNNING: {command} ===")
    print(f"START TIME: {datetime.now()}")

    start_time = time.time()
    try:
        result = subprocess.run(command.split(), capture_output=True, text=True)
        elapsed = time.time() - start_time
        minutes, seconds = divmod(elapsed, 60)

        print(f"COMPLETED IN: {int(minutes)}m {int(seconds)}s")
        print(f"END TIME: {datetime.now()}")
        print("=" * 50)

        # Logging output for debugging
        with open("experiment_results.log", "a") as log_file:
            log_file.write(f"\n=== RUNNING: {command} ===\n")
            log_file.write(result.stdout)
            log_file.write(result.stderr)
            log_file.write("=" * 50 + "\n")

    except Exception as e:
        print(f"ERROR: {e}")

def main():
    # Problem 3a
    opt_models = [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b"
    ]
    
    for model in opt_models:
        run_command(f"python truthfulqa.py {model}")
    
    # Problem 3b: Prompt Engineering - Test different prompting styles 
    # We already tested opt-1.3b with demos in Problem 3a
    # Zero-Shot (no demos, no system prompt)
    run_command("python truthfulqa.py facebook/opt-1.3b --no-demos")
    
    # System Prompt Only
    run_command("python truthfulqa.py facebook/opt-1.3b --system-prompt 'Actually,' --no-demos")
    
    # Demos + System Prompt
    run_command("python truthfulqa.py facebook/opt-1.3b --system-prompt 'Actually,'")

    #Demos + System Prompt changed for Extra credit
    run_command("python truthfulqa.py facebook/opt-2.7b --system-prompt 'Truthfully,'")

    print("\nAll experiments completed! Results logged in experiment_results.log.")

if __name__ == "__main__":
    main()
