#!/bin/bash
#SBATCH --partition=kamiak
#SBATCH --job-name=sudoku_scale
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:tesla:1

cd ~/CPTS440

module load anaconda3
source activate sudoku

srun python3 -u train_sudoku.py \
    --csv sudoku.csv \
    --nrows 2000000 \
    --epochs 20 \
    --lr 0.0005 \
    --batch_size 128 \
    --use_masked_loss \
    --resume \
    --save_path sudoku_scale_2M.pt \
    --constraint_weight 0.0 \
    --iter_confidence 0.95 \
    --iter_max_iters 20