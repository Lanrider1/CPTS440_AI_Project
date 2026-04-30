````markdown
# CPTS440_AI_Project

This project implements a Convolutional Neural Network (CNN) to solve Sudoku puzzles.
It explores how different training techniques and inference strategies affect performance on structured reasoning tasks.

---

## Project Overview

The model:

- Uses a CNN to predict missing Sudoku values  
- Applies masked loss to focus on blank cells  
- Optionally includes constraint-based regularization  
- Uses iterative constrained inference to improve full-board solving  

---

## Environment

This project was developed and executed on the <u>Kamiak High Performance Computing (HPC) cluster</u>
at <u>Washington State University (Pullman)</u>.

If you are running this project on Kamiak, refer to the official documentation:

- [Kamiak Quick Start Guide](https://hpc.wsu.edu/users-guide/quick-start-guide/)
- Cheat Sheets and User Resources (available on the same site)
- “Welcome to Kamiak” documentation

These resources explain:
- how to log in
- how to request compute nodes
- how to run jobs with `srun` and `sbatch`
- how GPU resources are allocated

---

## Files

- `train_sudoku.py` - Main training + evaluation script  
- `run_sudoku_gpu.sh` - Slurm batch script for GPU training  
- `sudoku.csv` - Dataset (puzzle, solution) found at https://www.kaggle.com/datasets/rohanrao/sudoku
- `sudoku_scale_24900417.out` - Example training outputs  

---
````
## Installation

### 1. Load environment

```bash
module load anaconda3
conda create -n sudoku python=3.10 -y
source activate sudoku

````
---
### 2. Install dependencies

```bash
pip install pandas numpy torch torchvision torchaudio
```

---

### 3. Verify GPU (optional)

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```text
True
```

---

## Running the Code

### Run locally (CPU or GPU if available)

```bash
python3 train_sudoku.py \
  --csv sudoku.csv \
  --nrows 10000 \
  --epochs 5 \
  --lr 0.0005 \
  --batch_size 128 \
  --use_masked_loss \
  --save_path sudoku_test.pt
```

---

### Run on Kamiak cluster (GPU)

Submit job:

```bash
sbatch run_sudoku_gpu.sh
```

Watch output:

```bash
tail -f sudoku_scale_<JOBID>.out
```

---

## Output Explanation

Below is an example of training output:

```text
Epoch 1/20
  Train Total Loss: 2.1106 | Train CE: 2.1106 | Train Constraint: 0.2388
  Train Cell Acc: 57.13% | Train Blank Acc: 17.55%
  Val Total Loss: 1.9651 | Val CE: 1.9651 | Val Constraint: 0.2863
  Val Cell Acc: 59.91% | Val Blank Acc: 23.21% | Val Board Acc: 0.0000%
```

---

## Training Metrics

* **Train Total Loss**
  Combined loss used for optimization

  ```
  Total Loss = Cross Entropy + Constraint Weight × Constraint Loss
  ```

* **Train CE (Cross Entropy Loss)**
  Measures prediction error for each cell

* **Train Constraint Loss**
  Penalizes violations of Sudoku rules (row/column/subgrid duplicates)

* **Train Cell Accuracy**
  % of all 81 cells predicted correctly

* **Train Blank Accuracy**
  % of only missing cells predicted correctly

---

## Validation Metrics

* Same as training, but computed on unseen data
* **Val Board Accuracy** = % of entire Sudoku boards solved perfectly

---

## Test Results

### Standard Test Results

```text
Test Total Loss: 1.3337
Test CE Loss:    1.3337
Test Constraint: 1.6823
Test Cell Acc:   69.21%
Test Blank Acc:  40.68%
Test Board Acc:  0.0000%
```

---

### Interpretation

* **Cell Accuracy (~69%)** → model is good at predicting individual digits
* **Blank Accuracy (~40%)** → moderate performance on missing cells
* **Board Accuracy (0%)** → almost no full puzzles solved

> Note: High cell accuracy does NOT mean the model can solve Sudoku correctly.

---

## Iterative Constrained Inference

```text
Running iterative constrained inference on test set...

Iterative Test Results
  Iter Cell Acc:   48.41%
  Iter Blank Acc:  0.68%
  Iter Board Acc:  0.0000%
  Iter Settings: confidence_threshold=0.95, max_iters=20
```

---

## What this does

Instead of predicting all cells at once, the model:

1. Predicts probabilities
2. Selects only high-confidence predictions
3. Inserts them if they do not violate constraints
4. Repeats until convergence

---

## Iterative Metrics

* **Iter Cell Accuracy** → Accuracy after iterative filling
* **Iter Blank Accuracy** → Accuracy on blanks after iterative process
* **Iter Board Accuracy** → % of boards fully solved
