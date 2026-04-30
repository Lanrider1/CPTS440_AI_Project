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

cd ~/CPTS440                          # Navigate to project directory
module load anaconda3                # Load Anaconda module
source activate sudoku               # Activate conda environment

# RUN TRAINING SCRIPT

srun python3 -u train_sudoku.py \    # Run Python script with unbuffered output (-u for real-time logs)
    --csv sudoku.csv \               # Path to dataset (CSV with puzzle/solution columns)
    --nrows 2000000 \               # Number of rows to load (2 million)
    --epochs 20 \                   # Number of training epochs
    --lr 0.0005 \                   # Learning rate
    --batch_size 128 \              # Batch size (kept constant for controlled experiments)
    --use_masked_loss \             # Use masked loss (focus training on blank cells)
    --resume \                      # Resume training if checkpoint exists
    --save_path sudoku_scale_2M.pt \ # File to save/load model checkpoint, 
                                    # you'll need to change the file name or increase the number 
                                    # of epochs to continue learning or start over
    --constraint_weight 0.0 \       # Disable constraint loss (baseline CNN experiment)
    --iter_confidence 0.95 \        # Confidence threshold for iterative inference
    --iter_max_iters 20             # Maximum number of iterative refinement steps
