#!/bin/bash

# SLURM JOB CONFIGURATION

#SBATCH --partition=kamiak            # Partition (queue) to run the job on
#SBATCH --job-name=sudoku_scale       # Name of the job (used in output filenames)
#SBATCH --output=%x_%j.out            # Standard output file (%x=job name, %j=job ID)
#SBATCH --error=%x_%j.err             # Standard error file
#SBATCH --time=08:00:00               # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                     # Number of nodes requested
#SBATCH --ntasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4             # Number of CPU cores allocated per task
#SBATCH --mem=32G                     # Amount of RAM requested
#SBATCH --gres=gpu:tesla:1            # Request 1 Tesla GPU

# ENVIRONMENT SETUP

# Navigate to project directory where .sh and .py files are located
# Replace with your folder name if different
cd ~/CPTS440

# Load Anaconda module
module load anaconda3

# Activate conda environment
source activate sudoku

# RUN TRAINING SCRIPT

# Run Python script with unbuffered output (-u for real-time logs)
srun python3 -u train_sudoku.py \

    # Path to dataset (CSV with puzzle/solution columns)
    --csv sudoku.csv \

    # Number of rows to load (dataset scaling experiment)
    --nrows 2000000 \

    # Number of training epochs
    --epochs 20 \

    # Learning rate
    --lr 0.0005 \

    # Batch size (kept constant for controlled experiments)
    --batch_size 128 \

    # Use masked loss (focus training on blank cells)
    --use_masked_loss \

    # Resume training if checkpoint exists
    --resume \

    # File to save/load model checkpoint
    # IMPORTANT: change this for each experiment OR increase epochs
    --save_path sudoku_scale_2M.pt \

    # Constraint loss weight (0.0 = baseline CNN, no constraints)
    --constraint_weight 0.0 \

    # Confidence threshold for iterative inference
    --iter_confidence 0.95 \

    # Maximum number of iterative refinement steps
    --iter_max_iters 20 \

    # Skip iterative evaluation entirely
    # Recommended for large datasets (1M+) because iterative testing is VERY slow
    # This allows the job to finish after standard test results
    --skip_iterative_eval

    # OPTIONAL: Limit iterative evaluation to a subset of test data
    # Use this INSTEAD of --skip_iterative_eval if you still want iterative results
    # Example: only evaluate 1000 puzzles instead of entire test set
    # --iter_eval_limit 1000
