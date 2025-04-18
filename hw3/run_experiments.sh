#!/bin/bash
# Script to run all TruthfulQA experiments

# Problem 3a: Test different model sizes
echo "Running Problem 3a experiments..."
python truthfulqa.py facebook/opt-125m
python truthfulqa.py facebook/opt-350m
python truthfulqa.py facebook/opt-1.3b
python truthfulqa.py facebook/opt-2.7b
python truthfulqa.py facebook/opt-6.7b

# Problem 3b: Test different prompting styles
echo -e "\nRunning Problem 3b experiments..."
# Zero-Shot (no demos, no system prompt)
python truthfulqa.py facebook/opt-1.3b --no-demos

# System Prompt Only
python truthfulqa.py facebook/opt-1.3b --system-prompt "Actually," --no-demos

# Demos + System Prompt
python truthfulqa.py facebook/opt-1.3b --system-prompt "Actually,"

# Extra credit
python truthfulqa.py facebook/opt-1.3b --system-prompt "Truthfully" -d extrademos.txt

echo -e "\nAll experiments completed!"

# Push results to GitHub
echo -e "\nPushing results to GitHub..."
git add .
git commit -m "Ran the experiments again"
git push origin main

echo -e "\nResults pushed to GitHub!"