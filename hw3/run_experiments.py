#!/usr/bin/env python3
"""
Script to automatically run all experiments for problems 3a and 3b.
This script is designed for running in a SLURM job without interactive input.
"""

import os
import subprocess
import time
import argparse
from datetime import datetime

def run_command(command, description):
    """Run a command and log the start/end times."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    process = subprocess.run(command, shell=True)
    end_time = time.time()
    
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {description}")
    print(f"END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ELAPSED TIME: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"EXIT CODE: {process.returncode}")
    print(f"{'='*80}\n")
    
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Run all TruthfulQA experiments for problems 3a and 3b")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (only 10 questions)")
    args = parser.parse_args()
    
    debug_flag = "--debug" if args.debug else ""
    
    # Define all experiments to run
    experiments = []
    
    # Problem 3a: Scaling Laws Experiments
    problem_3a_models = [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b"
    ]
    
    for model in problem_3a_models:
        model_name = model.split("/")[-1]
        experiments.append({
            "description": f"Problem 3a - Testing {model_name}",
            "command": f"python truthfulqa.py {model} {debug_flag}"
        })
    
    # Problem 3b: Prompt Engineering Experiments
    problem_3b_configs = [
        {"name": "Zero-Shot (no demos, no system prompt)", 
         "command": f"python truthfulqa.py facebook/opt-1.3b --no-demos {debug_flag}"},
        {"name": "Demos Only", 
         "command": f"python truthfulqa.py facebook/opt-1.3b {debug_flag}"},
        {"name": "System Prompt Only", 
         "command": f"python truthfulqa.py facebook/opt-1.3b --system-prompt 'Actually,' --no-demos {debug_flag}"},
        {"name": "Demos + System Prompt", 
         "command": f"python truthfulqa.py facebook/opt-1.3b --system-prompt 'Actually,' {debug_flag}"}
    ]
    
    # Already run the facebook/opt-1.3b with demos in Problem 3a, skip the second configuration (Demos Only) to avoid duplication
    for i, config in enumerate(problem_3b_configs):
        if i != 1:  # Skip the "Demos Only" configuration as it's duplicated in Problem 3a
            experiments.append({
                "description": f"Problem 3b - {config['name']}",
                "command": config['command']
            })
    
    # Run all experiments
    print(f"\nStarting all experiments. Total experiments: {len(experiments)}")
    print("Debug mode:", "ON" if args.debug else "OFF")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\nExperiment {i}/{len(experiments)}")
        run_command(exp["command"], exp["description"])
    
    print("\nAll experiments completed!")
    print("Results should be available in the 'results' directory.")

if __name__ == "__main__":
    main()