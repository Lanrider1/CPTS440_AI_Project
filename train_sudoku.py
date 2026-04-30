import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN to solve Sudoku puzzles")
    parser.add_argument("--csv", type=str, required=True, help="Path to sudoku CSV file")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to read from the CSV")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--save_path", type=str, default="sudoku_checkpoint.pt", help="Checkpoint path")
    parser.add_argument("--use_masked_loss", action="store_true", help="Compute CE loss only on blank cells")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if it exists")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--constraint_weight", type=float, default=0.0, help="Weight for Sudoku constraint loss")
    parser.add_argument(
        "--iter_confidence",
        type=float,
        default=0.95,
        help="Confidence threshold for iterative constrained inference"
    )
    parser.add_argument(
        "--iter_max_iters",
        type=int,
        default=20,
        help="Maximum number of iterative inference passes"
    )
    return parser.parse_args()


def puzzle_to_grid(puzzle_str):
    return np.array([int(c) for c in puzzle_str], dtype=np.int64).reshape(9, 9)


def solution_to_grid(solution_str):
    # Convert digits 1-9 into class labels 0-8
    return np.array([int(c) - 1 for c in solution_str], dtype=np.int64).reshape(9, 9)


def grid_to_one_hot(grid):
    """
    grid: (9, 9) values 0-9
    returns: torch tensor (10, 9, 9)
    """
    onehot = np.eye(10, dtype=np.float32)[grid]   # (9, 9, 10)
    return torch.tensor(onehot, dtype=torch.float32).permute(2, 0, 1)


def pct(x):
    return x * 100.0


class SudokuDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        puzzle = puzzle_to_grid(self.df.loc[idx, "puzzle"])         # values 0-9
        solution = solution_to_grid(self.df.loc[idx, "solution"])   # values 0-8

        puzzle_tensor = grid_to_one_hot(puzzle)                     # (10, 9, 9)
        solution_tensor = torch.tensor(solution, dtype=torch.long)  # (9, 9)

        return puzzle_tensor, solution_tensor


class SudokuCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 9, kernel_size=1)
        )

    def forward(self, x):
        # (B, 10, 9, 9) -> (B, 9, 9, 9)
        return self.net(x)


def masked_cross_entropy_loss(outputs, targets, puzzles):
    """
    outputs: (B, 9, 9, 9)
    targets: (B, 9, 9)
    puzzles: (B, 10, 9, 9)
    """
    blank_mask = (puzzles[:, 0, :, :] == 1)

    outputs = outputs.permute(0, 2, 3, 1).reshape(-1, 9)
    targets = targets.reshape(-1)
    blank_mask = blank_mask.reshape(-1)

    losses = F.cross_entropy(outputs, targets, reduction="none")
    masked_losses = losses[blank_mask]

    return masked_losses.mean()


def sudoku_constraint_loss(outputs):
    """
    Soft differentiable constraint loss.

    outputs: logits (B, 9, 9, 9)
    """
    probs = F.softmax(outputs, dim=1)  # (B, 9, 9, 9)

    # Row sums: for each digit and row, probabilities across columns should sum to 1
    row_sums = probs.sum(dim=3)  # (B, 9, 9)

    # Column sums: for each digit and column, probabilities across rows should sum to 1
    col_sums = probs.sum(dim=2)  # (B, 9, 9)

    # Box sums
    B = probs.size(0)
    box_probs = probs.view(B, 9, 3, 3, 3, 3)         # (B,9,3,3,3,3)
    box_probs = box_probs.permute(0, 1, 2, 4, 3, 5)  # (B,9,3,3,3,3)
    box_sums = box_probs.reshape(B, 9, 3, 3, 9).sum(dim=4)  # (B,9,3,3)

    row_loss = ((row_sums - 1.0) ** 2).mean()
    col_loss = ((col_sums - 1.0) ** 2).mean()
    box_loss = ((box_sums - 1.0) ** 2).mean()

    return row_loss + col_loss + box_loss


def constrained_predictions(outputs, puzzles):
    """
    Keep original puzzle clues fixed.
    outputs: (B, 9, 9, 9)
    puzzles: (B, 10, 9, 9)
    returns: (B, 9, 9) class labels 0-8
    """
    preds = torch.argmax(outputs, dim=1)  # (B, 9, 9)

    original_digits = torch.argmax(puzzles, dim=1)  # 0-9
    filled_mask = (original_digits != 0)
    original_labels = torch.clamp(original_digits - 1, min=0)

    preds = torch.where(filled_mask, original_labels, preds)
    return preds


def cell_accuracy(outputs, targets, puzzles):
    preds = constrained_predictions(outputs, puzzles)
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


def blank_cell_accuracy(outputs, targets, puzzles):
    preds = constrained_predictions(outputs, puzzles)
    blank_mask = (puzzles[:, 0, :, :] == 1)

    correct = ((preds == targets) & blank_mask).float().sum()
    total = blank_mask.float().sum()

    if total.item() == 0:
        return 0.0

    return (correct / total).item()


def board_accuracy(outputs, targets, puzzles):
    preds = constrained_predictions(outputs, puzzles)
    correct_boards = (preds == targets).view(targets.size(0), -1).all(dim=1).float().sum()
    return (correct_boards / targets.size(0)).item()


def train_one_epoch(model, loader, optimizer, device, use_masked_loss, constraint_weight):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_constraint_loss = 0.0
    total_cell_acc = 0.0
    total_blank_acc = 0.0

    for puzzles, solutions in loader:
        puzzles = puzzles.to(device)
        solutions = solutions.to(device)

        optimizer.zero_grad()
        outputs = model(puzzles)

        if use_masked_loss:
            ce_loss = masked_cross_entropy_loss(outputs, solutions, puzzles)
        else:
            ce_loss = F.cross_entropy(outputs, solutions)

        constraint_loss = sudoku_constraint_loss(outputs)
        total_objective = ce_loss + constraint_weight * constraint_loss

        total_objective.backward()
        optimizer.step()

        total_loss += total_objective.item()
        total_ce_loss += ce_loss.item()
        total_constraint_loss += constraint_loss.item()
        total_cell_acc += cell_accuracy(outputs, solutions, puzzles)
        total_blank_acc += blank_cell_accuracy(outputs, solutions, puzzles)

    return (
        total_loss / len(loader),
        total_ce_loss / len(loader),
        total_constraint_loss / len(loader),
        total_cell_acc / len(loader),
        total_blank_acc / len(loader),
    )


def evaluate(model, loader, device, use_masked_loss, constraint_weight):
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_constraint_loss = 0.0
    total_cell_acc = 0.0
    total_blank_acc = 0.0
    total_board_acc = 0.0

    with torch.no_grad():
        for puzzles, solutions in loader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)

            outputs = model(puzzles)

            if use_masked_loss:
                ce_loss = masked_cross_entropy_loss(outputs, solutions, puzzles)
            else:
                ce_loss = F.cross_entropy(outputs, solutions)

            constraint_loss = sudoku_constraint_loss(outputs)
            total_objective = ce_loss + constraint_weight * constraint_loss

            total_loss += total_objective.item()
            total_ce_loss += ce_loss.item()
            total_constraint_loss += constraint_loss.item()
            total_cell_acc += cell_accuracy(outputs, solutions, puzzles)
            total_blank_acc += blank_cell_accuracy(outputs, solutions, puzzles)
            total_board_acc += board_accuracy(outputs, solutions, puzzles)

    return (
        total_loss / len(loader),
        total_ce_loss / len(loader),
        total_constraint_loss / len(loader),
        total_cell_acc / len(loader),
        total_blank_acc / len(loader),
        total_board_acc / len(loader),
    )


def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    # backward compatibility with old weight-only save
    if "model_state_dict" not in checkpoint:
        print("Old checkpoint format detected; loading model weights only.", flush=True)
        model.load_state_dict(checkpoint)
        return 0, float("inf")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    return start_epoch, best_val_loss


# ---------------------------
# Iterative constrained inference
# ---------------------------

def is_valid_move(grid, row, col, digit):
    """
    grid: (9, 9), values 0-9
    digit: 1-9
    """
    if digit in grid[row, :]:
        return False

    if digit in grid[:, col]:
        return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    if digit in grid[box_row:box_row+3, box_col:box_col+3]:
        return False

    return True


def iterative_fill_step(model, grid, device, confidence_threshold=0.95):
    """
    One iterative inference pass.
    grid: numpy array (9, 9), values 0-9

    returns:
        new_grid, num_filled
    """
    model.eval()

    input_tensor = grid_to_one_hot(grid).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)           # (1, 9, 9, 9)
        probs = F.softmax(outputs, dim=1)[0]    # (9, 9, 9)

    candidates = []

    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                continue

            digit_probs = probs[:, r, c].detach().cpu().numpy()
            ranked = np.argsort(-digit_probs)

            best_digit = None
            best_prob = None

            for cls in ranked:
                digit = cls + 1  # class 0-8 -> digit 1-9
                prob = float(digit_probs[cls])

                if is_valid_move(grid, r, c, digit):
                    best_digit = digit
                    best_prob = prob
                    break

            if best_digit is not None:
                candidates.append((best_prob, r, c, best_digit))

    candidates.sort(reverse=True, key=lambda x: x[0])

    new_grid = grid.copy()
    num_filled = 0

    for prob, r, c, digit in candidates:
        if prob < confidence_threshold:
            continue

        if new_grid[r, c] == 0 and is_valid_move(new_grid, r, c, digit):
            new_grid[r, c] = digit
            num_filled += 1

    return new_grid, num_filled


def iterative_constrained_inference(
    model,
    puzzle_grid,
    device,
    confidence_threshold=0.95,
    max_iters=20,
):
    """
    puzzle_grid: numpy array (9, 9), values 0-9
    returns final_grid
    """
    grid = puzzle_grid.copy()

    for _ in range(max_iters):
        new_grid, num_filled = iterative_fill_step(
            model,
            grid,
            device,
            confidence_threshold=confidence_threshold,
        )

        if num_filled == 0:
            break

        grid = new_grid

        if np.all(grid != 0):
            break

    return grid


def evaluate_iterative(model, dataset, device, confidence_threshold=0.95, max_iters=20):
    """
    Evaluate iterative constrained inference over a dataset split.
    """
    model.eval()

    total_cell_acc = 0.0
    total_blank_acc = 0.0
    total_board_acc = 0.0
    total_examples = 0

    for puzzle_tensor, solution_tensor in dataset:
        # Recover original puzzle grid from one-hot tensor
        puzzle_grid = torch.argmax(puzzle_tensor, dim=0).cpu().numpy()  # (9, 9), values 0-9
        target_grid = (solution_tensor.cpu().numpy() + 1)               # (9, 9), values 1-9

        pred_grid = iterative_constrained_inference(
            model,
            puzzle_grid,
            device,
            confidence_threshold=confidence_threshold,
            max_iters=max_iters,
        )

        filled_mask = (puzzle_grid != 0)
        blank_mask = (puzzle_grid == 0)

        # Keep original clues fixed for evaluation
        pred_grid_eval = pred_grid.copy()
        pred_grid_eval[filled_mask] = target_grid[filled_mask]

        cell_acc = (pred_grid_eval == target_grid).mean()

        if blank_mask.sum() > 0:
            blank_acc = (pred_grid_eval[blank_mask] == target_grid[blank_mask]).mean()
        else:
            blank_acc = 0.0

        board_acc = float(np.array_equal(pred_grid_eval, target_grid))

        total_cell_acc += cell_acc
        total_blank_acc += blank_acc
        total_board_acc += board_acc
        total_examples += 1

    return (
        total_cell_acc / total_examples,
        total_blank_acc / total_examples,
        total_board_acc / total_examples,
    )


def main():
    args = parse_args()

    print("Loading CSV...", flush=True)
    df = pd.read_csv(args.csv, dtype=str, nrows=args.nrows)

    if "puzzle" not in df.columns or "solution" not in df.columns:
        raise ValueError(
            f"Expected columns 'puzzle' and 'solution', but found: {list(df.columns)}"
        )

    print(f"Loaded {len(df)} rows", flush=True)
    print("Columns:", list(df.columns), flush=True)

    dataset = SudokuDataset(df)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    model = SudokuCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.save_path):
        print(f"Loading checkpoint from {args.save_path}...", flush=True)
        start_epoch, best_val_loss = load_checkpoint(args.save_path, model, optimizer, device)
        print(f"Resuming from epoch {start_epoch}/{args.epochs}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_ce_loss, train_constraint_loss, train_cell_acc, train_blank_acc = train_one_epoch(
            model, train_loader, optimizer, device, args.use_masked_loss, args.constraint_weight
        )

        val_loss, val_ce_loss, val_constraint_loss, val_cell_acc, val_blank_acc, val_board_acc = evaluate(
            model, val_loader, device, args.use_masked_loss, args.constraint_weight
        )

        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
        print(
            f"  Train Total Loss: {train_loss:.4f} | "
            f"Train CE: {train_ce_loss:.4f} | "
            f"Train Constraint: {train_constraint_loss:.4f}",
            flush=True,
        )
        print(
            f"  Train Cell Acc: {pct(train_cell_acc):.2f}% | "
            f"Train Blank Acc: {pct(train_blank_acc):.2f}%",
            flush=True,
        )
        print(
            f"  Val Total Loss: {val_loss:.4f} | "
            f"Val CE: {val_ce_loss:.4f} | "
            f"Val Constraint: {val_constraint_loss:.4f}",
            flush=True,
        )
        print(
            f"  Val Cell Acc: {pct(val_cell_acc):.2f}% | "
            f"Val Blank Acc: {pct(val_blank_acc):.2f}% | "
            f"Val Board Acc: {pct(val_board_acc):.4f}%",
            flush=True,
        )

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(args.save_path, model, optimizer, epoch, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.save_path, model, optimizer, epoch, best_val_loss)
            print(f"  Saved best model to {args.save_path}", flush=True)

    if os.path.exists(args.save_path):
        print(f"Loading best checkpoint from {args.save_path} for testing...", flush=True)
        _, _ = load_checkpoint(args.save_path, model, optimizer, device)

    # Standard one-shot evaluation
    test_loss, test_ce_loss, test_constraint_loss, test_cell_acc, test_blank_acc, test_board_acc = evaluate(
        model, test_loader, device, args.use_masked_loss, args.constraint_weight
    )

    print("\nStandard Test Results", flush=True)
    print(f"  Test Total Loss: {test_loss:.4f}", flush=True)
    print(f"  Test CE Loss:    {test_ce_loss:.4f}", flush=True)
    print(f"  Test Constraint: {test_constraint_loss:.4f}", flush=True)
    print(f"  Test Cell Acc:   {pct(test_cell_acc):.2f}%", flush=True)
    print(f"  Test Blank Acc:  {pct(test_blank_acc):.2f}%", flush=True)
    print(f"  Test Board Acc:  {pct(test_board_acc):.4f}%", flush=True)

    # Iterative constrained inference evaluation
    print("\nRunning iterative constrained inference on test set...", flush=True)
    iter_cell_acc, iter_blank_acc, iter_board_acc = evaluate_iterative(
        model,
        test_dataset,
        device,
        confidence_threshold=args.iter_confidence,
        max_iters=args.iter_max_iters,
    )

    print("\nIterative Test Results", flush=True)
    print(f"  Iter Cell Acc:   {pct(iter_cell_acc):.2f}%", flush=True)
    print(f"  Iter Blank Acc:  {pct(iter_blank_acc):.2f}%", flush=True)
    print(f"  Iter Board Acc:  {pct(iter_board_acc):.4f}%", flush=True)
    print(
        f"  Iter Settings: confidence_threshold={args.iter_confidence}, max_iters={args.iter_max_iters}",
        flush=True,
    )


if __name__ == "__main__":
    main()