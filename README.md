# CPTS440_AI_Project

This project implements a Convolutional Neural Network (CNN) to solve Sudoku puzzles. It explores how different training techniques and inference strategies affect performance on structured reasoning tasks.

Project Overview

The model:

Uses a CNN to predict missing Sudoku values
Applies masked loss to focus on blank cells
Optionally includes constraint-based regularization
Uses iterative constrained inference to improve full-board solving

Files
1. train_sudoku.py - Main training + evaluation script
2. run_sudoku_gpu.sh - Slurm batch script for GPU training
3. sudoku.csv - Dataset (puzzle, solution)
4. *.out - Example training outputs

Installation
1. Load environment
module load anaconda3
conda create -n sudoku python=3.10 -y
source activate sudoku
2. Install dependencies
pip install pandas numpy torch torchvision torchaudio
3. Verify GPU (optional)
python3 -c "import torch; print(torch.cuda.is_available())"

Expected:

True
Running the Code
Run locally using (CPU or GPU if available)
python3 train_sudoku.py \
  --csv sudoku.csv \
  --nrows 10000 \
  --epochs 5 \
  --lr 0.0005 \
  --batch_size 128 \
  --use_masked_loss \
  --save_path sudoku_test.pt
Run on cluster (GPU)

Submit job:

sbatch run_sudoku_gpu.sh

Watch output:

tail -f sudoku_scale_<JOBID>.out

Output Explanation

Below is an example of training output:

Epoch 1/20
  Train Total Loss: 2.1106 | Train CE: 2.1106 | Train Constraint: 0.2388
  Train Cell Acc: 57.13% | Train Blank Acc: 17.55%
  Val Total Loss: 1.9651 | Val CE: 1.9651 | Val Constraint: 0.2863
  Val Cell Acc: 59.91% | Val Blank Acc: 23.21% | Val Board Acc: 0.0000%

Training Metrics

Train Total Loss
Combined loss used for optimization

1. Total Loss = Cross Entropy + Constraint Weight × Constraint Loss
2. Train CE (Cross Entropy Loss) - Measures prediction error for each cell
3. Train Constraint Loss - Penalizes violations of Sudoku rules (row/column/subgrid duplicates)
4. Train Cell Accuracy - % of all 81 cells predicted correctly
5. Train Blank Accuracy - % of only missing cells predicted correctly

Validation Metrics
Same as training, but computed on unseen data
Val Board Accuracy - % of entire Sudoku boards solved perfectly

Test Results
Standard Test Results
  Test Total Loss: 1.3337
  Test CE Loss:    1.3337
  Test Constraint: 1.6823
  Test Cell Acc:   69.21%
  Test Blank Acc:  40.68%
  Test Board Acc:  0.0000%

Interpretation
Test Cell Accuracy (~69%) - model is good at predicting individual digits
Test Blank Accuracy (~40%) - moderate performance on missing cells
Test Board Accuracy (0%) - almost no full puzzles solved

High cell accuracy ≠ solving Sudoku correctly

Iterative Constrained Inference
Running iterative constrained inference on test set...

Iterative Test Results
  Iter Cell Acc:   48.41%
  Iter Blank Acc:  0.68%
  Iter Board Acc:  0.0000%
  Iter Settings: confidence_threshold=0.95, max_iters=20

What this does:
Instead of predicting all cells at once, the model:

Predicts probabilities, Selects only high-confidence predictions,
Inserts them if they do not violate constraints,
Repeats until convergence

Metrics
Iter Cell Accuracy - Accuracy after iterative filling
Iter Blank Accuracy - Accuracy on blanks after iterative process
Iter Board Accuracy - % of boards fully solved
