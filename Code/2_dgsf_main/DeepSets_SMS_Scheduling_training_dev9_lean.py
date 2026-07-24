# -*- coding: utf-8 -*-
"""
Train an SMS DeepSets ranker using the lean Dev9 architecture and pairwise loss.

This is an experimental single-machine variant of the PMS Dev9-lean model:
  - keeps the Dev9-lean DeepSets encoder, attention pooling, post block, and
    priority head;
  - uses the original SMS per-job feature columns;
  - replaces the old SMS Margin-ListNet objective with a Dev9-style pairwise
    margin ranking loss over all valid job pairs in each single-machine
    sequence.

The script is intentionally separate from the submitted SMS trainer so the old
and revised ML models can be compared cleanly during the R1 revision.
"""

import argparse
import ast
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


SMS_FEATURE_COLS = [
    "rel_proc_time",
    "window_tightness_jobs",
    "slack_time",
    "release_percentile",
    "due_percentile",
    "atc_t0_rank",
    "atc_at_release_rank",
    "load_before_due_norm",
    "myopic_lateness",
]

# Default cluster paths matching the currently used SMS cluster trainer.
# To train Type B instead, either edit these two lines or pass --input and
# --output-model on the command line.
DEFAULT_TRAIN_FILE = "Dev3_50k_singlemachine_instances_30_theta_max_6Itau_train.csv"
DEFAULT_OUTPUT_MODEL = "30_theta_max_6Itau_Dev3_50k_dev9_lean.pth"
DEFAULT_METRICS_CSV = "30_theta_max_6Itau_Dev3_50k_dev9_lean_metrics.csv"
DEFAULT_LOSS_PLOT = "30_theta_max_6Itau_Dev3_50k_dev9_lean_loss.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SMS model with Dev9-lean architecture and pairwise loss."
    )
    parser.add_argument("--input", default=DEFAULT_TRAIN_FILE, help="Training CSV/XLSX file.")
    parser.add_argument("--output-model", default=DEFAULT_OUTPUT_MODEL, help="Output .pth checkpoint.")
    parser.add_argument("--metrics-csv", default=DEFAULT_METRICS_CSV, help="Epoch metrics CSV.")
    parser.add_argument("--loss-plot", default=DEFAULT_LOSS_PLOT, help="Loss-curve PNG.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--select-by",
        choices=["val_loss", "val_gamma", "val_spearman"],
        default="val_gamma",
        help="Criterion for saving the best checkpoint.",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    return parser.parse_args()


def parse_1d(cell, dtype=np.float32) -> np.ndarray:
    return np.array(ast.literal_eval(str(cell)), dtype=dtype).flatten()


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SMSScheduleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_cols: list[str], zscore: bool = True):
        self.samples = []

        for _, row in dataframe.iterrows():
            features = []
            for col in feature_cols:
                if col not in row.index:
                    raise KeyError(f"Missing required feature column: {col}")
                features.append(parse_1d(row[col]).reshape(-1, 1))
            job_x = np.concatenate(features, axis=1).astype(np.float32)

            if zscore:
                mu = job_x.mean(axis=0, keepdims=True)
                sd = job_x.std(axis=0, keepdims=True) + 1e-8
                job_x = (job_x - mu) / sd

            ranks = parse_1d(row["rank_vector_dtime"], dtype=np.float32)
            tau = parse_1d(row["tau"], dtype=np.float32)
            eps = parse_1d(row["eps"], dtype=np.float32)
            rho = parse_1d(row["rho"], dtype=np.float32)
            total_tardiness = float(row["tardiness_dtime"])

            n_jobs = len(ranks)
            if job_x.shape[0] != n_jobs:
                raise ValueError(
                    f"Feature length {job_x.shape[0]} does not match rank length {n_jobs}."
                )

            self.samples.append(
                {
                    "job_x": torch.from_numpy(job_x),
                    "ranks": torch.from_numpy(ranks),
                    "tau": torch.from_numpy(tau),
                    "eps": torch.from_numpy(eps),
                    "rho": torch.from_numpy(rho),
                    "total_tardiness": torch.tensor(total_tardiness, dtype=torch.float32),
                    "N": n_jobs,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def sms_collate(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_n = max(sample["N"] for sample in batch)
    feat_dim = batch[0]["job_x"].shape[1]

    job_x = torch.zeros(batch_size, max_n, feat_dim, dtype=torch.float32)
    ranks = torch.full((batch_size, max_n), -1.0, dtype=torch.float32)
    tau = torch.zeros(batch_size, max_n, dtype=torch.float32)
    eps = torch.zeros(batch_size, max_n, dtype=torch.float32)
    rho = torch.zeros(batch_size, max_n, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_n, dtype=torch.float32)
    total_tardiness = torch.zeros(batch_size, dtype=torch.float32)

    for b, sample in enumerate(batch):
        n_jobs = sample["N"]
        job_x[b, :n_jobs] = sample["job_x"]
        ranks[b, :n_jobs] = sample["ranks"]
        tau[b, :n_jobs] = sample["tau"]
        eps[b, :n_jobs] = sample["eps"]
        rho[b, :n_jobs] = sample["rho"]
        mask[b, :n_jobs] = 1.0
        total_tardiness[b] = sample["total_tardiness"]

    return {
        "job_x": job_x,
        "ranks": ranks,
        "tau": tau,
        "eps": eps,
        "rho": rho,
        "mask": mask,
        "total_tardiness": total_tardiness,
    }


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "hidden dimension must be divisible by heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_jobs, dim = x.shape
        q = self.q(x).view(batch_size, n_jobs, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, n_jobs, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, n_jobs, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask[:, None, None, :] == 0, -1e4)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, n_jobs, dim)
        out = self.out(out)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (out * mask.unsqueeze(-1)).sum(dim=1) / denom


class DeepSetsEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2, heads: int = 8):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
        )
        self.pool = AttentionPool(hidden, num_heads=heads)
        self.post = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden, dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.phi(x)
        g = self.pool(h, mask)
        h = h + g.unsqueeze(1)
        h = self.post(h)
        return h, g


class SMSDev9LeanRanker(nn.Module):
    def __init__(self, job_in: int, hidden: int = 128, dropout: float = 0.2, heads: int = 8):
        super().__init__()
        self.job_enc = DeepSetsEncoder(job_in, hidden, dropout, heads)
        self.prio_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, job_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        job_h, _ = self.job_enc(job_x, mask)
        prio = self.prio_head(job_h).squeeze(-1)
        return prio.masked_fill(mask == 0, -1e4)


def sms_pairwise_ranking_loss(
    prio: torch.Tensor,
    ranks: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    batch_size = prio.shape[0]
    total_loss = prio.new_zeros(())
    count = 0

    for b in range(batch_size):
        valid = (mask[b] > 0) & (ranks[b] > 0)
        if valid.sum() < 2:
            continue

        r = ranks[b][valid]
        p = prio[b][valid]
        precedes = (r.unsqueeze(1) < r.unsqueeze(0)).float()
        if precedes.sum() == 0:
            continue

        score_diff = p.unsqueeze(1) - p.unsqueeze(0)
        margin_viol = torch.clamp(margin - score_diff, min=0.0)
        total_loss = total_loss + (margin_viol * precedes).sum() / precedes.sum().clamp_min(1e-6)
        count += 1

    return total_loss / max(count, 1)


def simulate_tardiness(scores, tau, eps, rho, mask) -> torch.Tensor:
    batch_size, max_n = scores.shape
    masked_scores = scores.masked_fill(mask == 0, -1e9)
    order = torch.argsort(masked_scores, dim=1, descending=True)
    batch_idx = torch.arange(batch_size, device=scores.device).unsqueeze(1).expand(batch_size, max_n)
    ordered_tau = tau[batch_idx, order]
    ordered_eps = eps[batch_idx, order]
    ordered_rho = rho[batch_idx, order]
    ordered_mask = mask[batch_idx, order]

    start = torch.zeros(batch_size, max_n, device=scores.device, dtype=scores.dtype)
    start[:, 0] = ordered_rho[:, 0]
    for pos in range(1, max_n):
        start[:, pos] = torch.maximum(start[:, pos - 1] + ordered_tau[:, pos - 1], ordered_rho[:, pos])

    completion = start + ordered_tau
    return (torch.clamp(completion - ordered_eps, min=0.0) * ordered_mask).sum(dim=1)


@torch.no_grad()
def evaluate(model, loader, device, margin: float) -> dict:
    model.eval()
    losses = []
    spearman_scores = []
    gammas = []

    for batch in loader:
        for key, val in batch.items():
            if torch.is_tensor(val):
                batch[key] = val.to(device, non_blocking=True)

        scores = model(batch["job_x"], batch["mask"])
        loss = sms_pairwise_ranking_loss(scores, batch["ranks"], batch["mask"], margin=margin)
        losses.append(float(loss.item()))

        pred_tardiness = simulate_tardiness(
            scores, batch["tau"], batch["eps"], batch["rho"], batch["mask"]
        )
        gamma = (pred_tardiness - batch["total_tardiness"]) / pred_tardiness.clamp_min(1e-6) * 100.0
        gammas.extend(gamma.detach().cpu().numpy().tolist())

        for i in range(scores.shape[0]):
            valid = batch["mask"][i].bool()
            if valid.sum() < 2:
                continue
            order = torch.argsort(-scores[i][valid])
            pred_rank = torch.argsort(order).detach().cpu().numpy() + 1
            true_rank = batch["ranks"][i][valid].detach().cpu().numpy()
            corr = spearmanr(true_rank, pred_rank).correlation
            if corr is not None and not np.isnan(corr):
                spearman_scores.append(float(corr))

    return {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "gamma": float(np.mean(gammas)) if gammas else float("inf"),
        "spearman": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
    }


def criterion_value(metrics: dict, select_by: str) -> float:
    if select_by == "val_loss":
        return metrics["loss"]
    if select_by == "val_gamma":
        return metrics["gamma"]
    if select_by == "val_spearman":
        return -metrics["spearman"]
    raise ValueError(select_by)


def plot_history(history: list[dict], save_path: str) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    ax1.plot(epochs, [row["val_loss"] for row in history], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Pairwise margin loss")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["val_gamma"] for row in history], color="tab:red", label="Val gamma")
    ax2.set_ylabel("Validation gamma (%)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
    device = torch.device(args.device)

    df = read_table(args.input)
    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
    train_ds = SMSScheduleDataset(train_df, SMS_FEATURE_COLS)
    val_ds = SMSScheduleDataset(val_df, SMS_FEATURE_COLS)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=sms_collate,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=sms_collate,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = SMSDev9LeanRanker(
        job_in=len(SMS_FEATURE_COLS),
        hidden=args.hidden,
        dropout=args.dropout,
        heads=args.heads,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"Input rows: {len(df)} | train={len(train_ds)} | val={len(val_ds)}")
    print(f"Device: {device} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Selection: {args.select_by}")

    best_score = float("inf")
    best_metrics = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0

        for batch in train_loader:
            for key, val in batch.items():
                if torch.is_tensor(val):
                    batch[key] = val.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                scores = model(batch["job_x"], batch["mask"])
                loss = sms_pairwise_ranking_loss(
                    scores, batch["ranks"], batch["mask"], margin=args.margin
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            batches += 1

        scheduler.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            train_loss = running / max(batches, 1)
            val_metrics = evaluate(model, val_loader, device, margin=args.margin)
            row = {
                "epoch": epoch,
                "minutes": (time.time() - start_time) / 60.0,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_gamma": val_metrics["gamma"],
                "val_spearman": val_metrics["spearman"],
            }
            history.append(row)

            score = criterion_value(val_metrics, args.select_by)
            improved = score < best_score
            if improved:
                best_score = score
                best_metrics = row
                Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "architecture": "dev9_lean",
                        "model_module": "DeepSets_SMS_Scheduling_training_dev9_lean",
                        "model_class": "SMSDev9LeanRanker",
                        "model_kwargs": {
                            "job_in": len(SMS_FEATURE_COLS),
                            "hidden": args.hidden,
                            "dropout": args.dropout,
                            "heads": args.heads,
                        },
                        "feature_cols": SMS_FEATURE_COLS,
                        "hidden": args.hidden,
                        "dropout": args.dropout,
                        "heads": args.heads,
                        "margin": args.margin,
                        "select_by": args.select_by,
                        "best_metrics": best_metrics,
                    },
                    args.output_model,
                )

            tag = " saved" if improved else ""
            print(
                f"Epoch {epoch:4d} | {row['minutes']:6.1f}m | "
                f"train={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"val_gamma={val_metrics['gamma']:.3f} | "
                f"val_rho={val_metrics['spearman']:.4f}{tag}"
            )

    if args.metrics_csv:
        Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(args.metrics_csv, index=False)
    if args.loss_plot:
        Path(args.loss_plot).parent.mkdir(parents=True, exist_ok=True)
        plot_history(history, args.loss_plot)

    print(f"Done. Best checkpoint: {args.output_model}")
    if best_metrics:
        print(f"Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()
