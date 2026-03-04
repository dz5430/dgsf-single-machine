# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr 
import math  
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================
# Load and process data
# =============================
# ---- Load dataset (solved optimal schedules + features) ----
df = pd.read_csv("Training_file.csv") # Update with own file

# Selected input features 
input_feature_cols = [
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
input_size = len(input_feature_cols)

class ScheduleDataset(Dataset):
    def __init__(self, dataframe, input_feature_cols, use_instance_zscore=True):
        self.samples = []
        for _, row in dataframe.iterrows():
            # Parse per-job feature arrays
            feature_vectors = [np.array(ast.literal_eval(str(row[col])), dtype=np.float32)
                               for col in input_feature_cols]
            features = np.column_stack(feature_vectors).astype(np.float32)  # [N_jobs, F]

            # Optional: instance-level standardization (lighter than sklearn per-row)
            if use_instance_zscore:
                mu = features.mean(axis=0, keepdims=True)
                sd = features.std(axis=0, keepdims=True) + 1e-8
                features = (features - mu) / sd

            ranks            = np.array(ast.literal_eval(str(row["rank_vector_dtime"])), dtype=np.float32)
            processing_times = np.array(ast.literal_eval(str(row["tau"])), dtype=np.float32)
            due_dates        = np.array(ast.literal_eval(str(row["eps"])), dtype=np.float32)
            release_times    = np.array(ast.literal_eval(str(row["rho"])), dtype=np.float32)
            total_tardiness  = float(row["tardiness_dtime"])

            self.samples.append({
                "features":         torch.from_numpy(features),          # [N,F]
                "ranks":            torch.from_numpy(ranks),             # [N]
                "processing_times": torch.from_numpy(processing_times),  # [N]
                "due_dates":        torch.from_numpy(due_dates),         # [N]
                "release_times":    torch.from_numpy(release_times),     # [N]
                "total_tardiness":  torch.tensor(total_tardiness, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    features         = [b["features"] for b in batch]
    ranks            = [b["ranks"] for b in batch]
    processing_times = [b["processing_times"] for b in batch]
    due_dates        = [b["due_dates"] for b in batch]
    release_times    = [b["release_times"] for b in batch]
    total_tardiness  = [b["total_tardiness"] for b in batch]

    padded_features         = nn.utils.rnn.pad_sequence(features,         batch_first=True)              # [B,N,F]
    padded_ranks            = nn.utils.rnn.pad_sequence(ranks,            batch_first=True, padding_value=-1)  # [B,N]
    padded_processing_times = nn.utils.rnn.pad_sequence(processing_times, batch_first=True, padding_value=0)   # [B,N]
    padded_due_dates        = nn.utils.rnn.pad_sequence(due_dates,        batch_first=True, padding_value=0)   # [B,N]
    padded_release_times    = nn.utils.rnn.pad_sequence(release_times,    batch_first=True, padding_value=0)   # [B,N]
    total_tardiness_tensor  = torch.stack(total_tardiness)  # [B]

    # Precompute mask here for convenience downstream (1 for real jobs, 0 for pad)
    mask = (padded_ranks != -1).float()

    return {
        "features":         padded_features,
        "ranks":            padded_ranks,
        "processing_times": padded_processing_times,
        "due_dates":        padded_due_dates,
        "release_times":    padded_release_times,
        "total_tardiness":  total_tardiness_tensor,
        "mask":             mask,
    }

# =============================
# Model: DeepSetsRanker
# =============================
# Residual block
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

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    [B, N, D]
        mask: [B, N] with 1 for real jobs, 0 for pad
        returns pooled: [B, D] (permutation-invariant)
        """
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,hd]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,hd]
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,hd]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)      # [B,H,N,N]
        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
            scores = scores.masked_fill(m == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)                                         # [B,H,N,N]
        out = torch.matmul(attn, v)                                                  # [B,H,N,hd]
        out = out.transpose(1, 2).contiguous().view(B, N, D)                         # [B,N,D]
        out = self.out_proj(out)                                                     # [B,N,D]

        # permutation-invariant pooling over the set (mask pads)
        if mask is None:
            return out.mean(dim=1)                                                   # [B,D]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)                         # [B,1]
        return (out * mask.unsqueeze(-1)).sum(dim=1) / denom                         # [B,D]


class DeepSetsRanker(nn.Module):
    """
    Input:  x    [B, N, F]  (per-job features)
            mask [B, N]     (1 = real, 0 = pad)
    Output: schedule_scores [B, N] (higher => earlier in schedule)

    Architecture matches Dev3_30_theta_max_6Itau_50k_old_3_sp.pth:
      embedding : Linear(input_size→hidden) → LayerNorm → GELU → Dropout
      encoder   : 2× ResidualBlock(hidden)  [hidden→2*hidden→hidden]
      attn_pool : MultiHeadAttentionPooling(hidden)
      rank_head : Linear(hidden→1)
    """
    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.3, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # φ: per-item embedding (permutation-equivariant)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # deeper φ via residual blocks
        self.encoder = nn.Sequential(
            ResidualBlock(hidden_size, dropout),
            ResidualBlock(hidden_size, dropout),
        )

        # attention pooling to get a global context
        self.attn_pool = MultiHeadAttentionPooling(hidden_size, num_heads=num_heads)

        # per-item scorer
        self.rank_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        h = self.embedding(x)            # [B,N,H]
        h = self.encoder(h)              # [B,N,H]

        g = self.attn_pool(h, mask)      # [B,H] (invariant)
        g = g.unsqueeze(1).expand_as(h)  # [B,N,H] broadcast global context

        h = h + g                        # context injection (equivariant)
        scores = self.rank_head(h).squeeze(-1)  # [B,N]

        # mask out pads so they won't affect losses downstream
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        return {"schedule_scores": scores}

# -----------------------------
# Loss: Margin ListNet
# -----------------------------
def margin_listnet_loss(preds, targets, margin=0.5, margin_weight=0.1):
    mask = targets != -1
    preds, targets = preds[mask], targets[mask]

    if len(preds) <= 1:
        return torch.tensor(0.0, device=preds.device)

    targets = -targets  # Lower rank = more urgent

    pred_prob = torch.softmax(preds, dim=0)
    target_prob = torch.softmax(targets, dim=0)
    listnet_loss = -torch.sum(target_prob * torch.log(pred_prob + 1e-8))

    pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    should_rank_higher = (target_diff > 0).float()
    margin_violations = torch.clamp(margin - pred_diff, min=0) * should_rank_higher
    margin_loss = margin_violations.sum() / torch.clamp(should_rank_higher.sum(), min=1)

    return listnet_loss + margin_weight * margin_loss

# -----------------------------
# Schedule simulation
# -----------------------------
def simulate_schedule_tardiness(scores, proc, due, release, mask):
    """
    All tensors [B, N]; mask in {0,1}.
    Returns total tardiness per instance: [B]
    """
    B, N = scores.shape
    mask_bool = mask.bool()

    # Push padded items to the end with a large negative
    masked_scores = scores.masked_fill(~mask_bool, -1e9)

    # Sort by predicted priority (desc)
    sort_indices = torch.argsort(masked_scores, dim=1, descending=True)  # [B,N]
    batch_idx = torch.arange(B, device=scores.device).unsqueeze(1).expand(B, N)

    sp = proc[batch_idx, sort_indices]
    sd = due[batch_idx, sort_indices]
    sr = release[batch_idx, sort_indices]
    sm = mask[batch_idx, sort_indices].float()

    # Build start/finish times with release constraints
    start = torch.zeros(B, N, device=scores.device, dtype=scores.dtype)
    start[:, 0] = sr[:, 0]
    for i in range(1, N):
        prev_end = start[:, i-1] + sp[:, i-1]
        start[:, i] = torch.maximum(prev_end, sr[:, i])

    completion = start + sp
    tardiness = torch.clamp(completion - sd, min=0.0) * sm
    return tardiness.sum(dim=1)  # [B]

# -----------------------------
# Loss
# -----------------------------
def compute_loss(model_outputs, batch_data):
    scores = model_outputs['schedule_scores']              # [B,N]
    ranks  = batch_data['ranks']                           # [B,N]
    mask   = (ranks != -1).float()                         # [B,N]
    B      = scores.size(0)

    # Average margin-ListNet loss over instances with >1 valid job
    rank_accum = scores.new_zeros(())
    valid_inst = 0
    for i in range(B):
        valid = mask[i].bool()
        if valid.sum() > 1:
            rank_accum = rank_accum + margin_listnet_loss(scores[i][valid], ranks[i][valid])
            valid_inst += 1

    return rank_accum / max(valid_inst, 1)

# -----------------------------
# Training Epoch
# -----------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, clip_norm=1.0):
    model.train()
    running = 0.0
    n_batches = 0

    for batch in dataloader:
        # move tensors to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # mask for padding-aware layers
        mask = (batch['ranks'] != -1).float()
        outputs = model(batch['features'], mask=mask)

        loss = compute_loss(outputs, batch)
        loss.backward()

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()

        running += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return running / max(n_batches, 1)


# -----------------------------
# Evaluation Metrics
# -----------------------------
@torch.no_grad()

# -----------------------------
# Validation Loss
# -----------------------------
@torch.no_grad()
def compute_val_loss(model, dataloader, device):
    """Compute average loss on the validation/test set (no gradients)."""
    model.eval()
    running = 0.0
    n_batches = 0

    for batch in dataloader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        mask = (batch['ranks'] != -1).float()
        outputs = model(batch['features'], mask=mask)
        loss = compute_loss(outputs, batch)
        running += loss.item()
        n_batches += 1

    return running / max(n_batches, 1)

def evaluate_rank_metrics(model, dataloader, device):
    model.eval()
    spearman_scores = []
    tardiness_deltas = []

    for batch in dataloader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        mask = (batch['ranks'] != -1).float()

        # forward (use mask for attention/pooling)
        outputs = model(batch['features'], mask=mask)
        scores = outputs['schedule_scores']        # [B, N]
        y_rank = batch['ranks']                    # [B, N]

        # Spearman per instance
        for i in range(scores.size(0)):
            valid = mask[i].bool()
            if valid.sum() > 1:
                # convert scores -> rank order (1 = highest priority)
                order = torch.argsort(-scores[i][valid])
                pred_rank = torch.argsort(order).cpu().numpy() + 1
                true_rank = y_rank[i][valid].cpu().numpy()

                if len(np.unique(true_rank)) > 1:
                    s = spearmanr(true_rank, pred_rank).correlation
                    if s is not None:
                        spearman_scores.append(float(s))

        # Hard (non-diff) simulator for evaluation/selection
        pred_tardiness = simulate_schedule_tardiness(
            scores,
            batch['processing_times'],
            batch['due_dates'],
            batch['release_times'],
            mask
        )  # [B]

        # % improvement vs oracle tardiness
        denom = (batch['total_tardiness'] + 1e-6)
        delta = (batch['total_tardiness'] - pred_tardiness) / denom
        tardiness_deltas.extend(delta.detach().cpu().numpy())

    return {
        "spearman": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
        "tardiness_improvement": float(np.mean(tardiness_deltas)) if tardiness_deltas else 0.0,
    }

# =============================
# Data Preparation
# =============================
# Reproducibility
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
train_dataset = ScheduleDataset(train_df, input_feature_cols)
test_dataset  = ScheduleDataset(test_df,  input_feature_cols)

# Tweak num_workers/pin_memory to your machine
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True,
    collate_fn=collate_fn, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False,
    collate_fn=collate_fn, num_workers=0, pin_memory=True
)

# =============================
# Model + Optimizer
# =============================
device = torch.device("cuda")
model = DeepSetsRanker(input_size=input_size).to(device)

max_epochs = 1000
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs*0.8)

# =============================
# Training Loop with Model Selection
# =============================
print(f"\n Training on device: {device}")
print("Starting Training...\n")

best_loss = float("inf")
best_rho = -float("inf")
best_epoch = -1
best_model_state = None
eps = 1e-9


# ---- Loss curve history ----
train_loss_history = []
val_loss_history   = []

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss   = compute_val_loss(model, test_loader, device)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    metrics = evaluate_rank_metrics(model, test_loader, device)  # still just for logging
    spearman = metrics["spearman"]
    tardiness_delta = metrics["tardiness_improvement"]

    # # --- Select purely on min training loss ---
    # if train_loss + eps < best_loss:
    #     best_loss = train_loss
    #     best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    #     best_epoch = epoch
    #     print(f" New best @ epoch {epoch:4d} | loss={train_loss:.6f} | "
    #           f"ρ={spearman:.4f} | ΔTard={tardiness_delta:.2%}")
    
    # --- Select based on max Spearman’s rho ---
    if spearman - eps > best_rho:
        best_rho = spearman
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_epoch = epoch
        print(f" New best @ epoch {epoch:4d} | ρ={spearman:.4f} | "
              f"loss={train_loss:.6f} | ΔTard={tardiness_delta:.2%}")

    if epoch % 100 == 0:
        print(f"[Epoch {epoch:4d}] loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"ρ={spearman:.4f} | ΔTard={tardiness_delta:.2%}")


# =============================
# Plot and save loss curves
# =============================
def plot_loss_curves(train_losses, val_losses, best_epoch, save_path='loss_curves.png'):
    epochs = range(len(train_losses))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Training Loss",   color="#2C6FBF", linewidth=1.8, alpha=0.9)
    ax.plot(epochs, val_losses,   label="Validation Loss", color="#E05A2B", linewidth=1.8, alpha=0.9)
    if best_epoch >= 0:
        ax.axvline(x=best_epoch, color="#2CA02C", linestyle="--", linewidth=1.4,
                   label=f"Best model (epoch {best_epoch})")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("DeepSets Ranker — Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Loss curve saved to {save_path}")

plot_loss_curves(train_loss_history, val_loss_history, best_epoch,
                 save_path='loss_curves_0max_40tau_avg.png')

# =============================
# Save best model
# =============================
if best_model_state is not None:
    filename = "Model_name.pth" # Update with own file
    torch.save(best_model_state, filename)
    print(f"Best model saved to {filename} (epoch {best_epoch})")
else:
    print("No model improvement detected. Final model not saved.")
