#!/usr/bin/env python3
"""
Interactive script to run selected experiments for problems 3a and 3b.
This script will ask which specific experiments you want to run and execute only those.
"""

import os
import subprocess
import time
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

def get_user_selection(options, prompt):
    """Ask user to select options from a list."""
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    while True:
        try:
            selection_input = input("Enter the numbers of your selections (comma-separated, or 'all' for all options): ")
            
            if selection_input.lower() == 'all':
                return list(range(len(options)))
            
            selections = [int(x.strip()) - 1 for x in selection_input.split(',')]
            valid_selections = [s for s in selections if 0 <= s < len(options)]
            
            if not valid_selections:
                print("No valid selections. Please try again.")
                continue
                
            return valid_selections
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

def main():
    print("\nTruthfulQA Experiment Runner for Problems 3a and 3b\n")
    
    # Ask if debug mode should be used
    debug_mode = input("Run in debug mode (only 10 questions)? (y/n): ").lower().startswith('y')
    debug_flag = "--debug" if debug_mode else ""
    
    # Define experiment options
    problem_3a_models = [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b"
    ]
    
    problem_3b_configs = [
        {"name": "Zero-Shot (no demos, no system prompt)", 
         "command": "python truthfulqa.py facebook/opt-1.3b --no-demos"},
        {"name": "Demos Only", 
         "command": "python truthfulqa.py facebook/opt-1.3b"},
        {"name": "System Prompt Only", 
         "command": "python truthfulqa.py facebook/opt-1.3b --system-prompt 'Actually,' --no-demos"},
        {"name": "Demos + System Prompt", 
         "command": "python truthfulqa.py facebook/opt-1.3b --system-prompt 'Actually,'"}
    ]
    
    # Ask which problems to run
    print("\nWhich problems would you like to run?")
    problems = ["Problem 3a (Scaling Laws)", "Problem 3b (Prompt Engineering)"]
    problem_selections = get_user_selection(problems, "Available problems:")
    
    experiments_to_run = []
    
    # For Problem 3a, ask which models to run
    if 0 in problem_selections:  # Problem 3a selected
        model_selections = get_user_selection(problem_3a_models, "\nWhich OPT models would you like to test for Problem 3a?")
        for idx in model_selections:
            model = problem_3a_models[idx]
            model_name = model.split("/")[-1]
            experiments_to_run.append({
                "description": f"Problem 3a - Testing {model_name}",
                "command": f"python truthfulqa.py {model} {debug_flag}"
            })
    
    # For Problem 3b, ask which prompting configurations to run
    if 1 in problem_selections:  # Problem 3b selected
        config_selections = get_user_selection(
            [config["name"] for config in problem_3b_configs], 
            "\nWhich prompting configurations would you like to test for Problem 3b?"
        )
        for idx in config_selections:
            config = problem_3b_configs[idx]
            experiments_to_run.append({
                "description": f"Problem 3b - {config['name']}",
                "command": f"{config['command']} {debug_flag}"
            })
    
    # Check if we have experiments to run
    if not experiments_to_run:
        print("\nNo experiments selected. Exiting.")
        return
    
    # Confirm the experiments to run
    print("\nConfirm running the following experiments:")
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"{i}. {exp['description']}")
    
    confirm = input("\nProceed with these experiments? (y/n): ")
    if not confirm.lower().startswith('y'):
        print("Aborted. No experiments will be run.")
        return
    
    # Run the selected experiments
    print("\nStarting selected experiments...")
    for exp in experiments_to_run:
        run_command(exp["command"], exp["description"])
    
    print("\nAll selected experiments completed!")
    print("Results should be available in the 'results' directory.")

if __name__ == "__main__":
    main()