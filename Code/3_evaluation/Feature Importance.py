# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import math
import random

# =========================
# Config
# =========================
MODEL_PATH = "Chosen_model.pth"   # Update with own file
DATA_PATH  = "DGSF_processed_file.xlsx"  # Update with own file

# Chosen input features
input_feature_cols = [
    "rel_proc_time",
    "window_tightness_jobs",
    "slack_time",
    "release_percentile",
    "due_percentile",
    "atc_t0_rank",
    "atc_at_release_rank",
    "load_before_due_norm",
    "myopic_lateness"
]

input_size = len(input_feature_cols)

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

# =========================
# Utilities
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def spearman_safe(a, b):
    val = spearmanr(a, b).correlation
    return np.nan if val is None else val

def predict_scores(model, feat_tensor):
    # feat_tensor: (N, F) -> returns numpy (N,)
    with torch.no_grad():
        out = model(feat_tensor.unsqueeze(0))  # (1, N, F)
        scores = out["schedule_scores"].squeeze(0).cpu().numpy()
    return scores

# =========================
# Load data (per-instance scaling)
# =========================
def load_instances(df, feature_cols):
    X_list, y_list, names = [], [], []
    for idx, row in df.iterrows():
        try:
            feats = []
            for col in feature_cols:
                feats.append(np.array(ast.literal_eval(str(row[col]))).reshape(-1, 1))
            X = np.concatenate(feats, axis=1)
            X = StandardScaler().fit_transform(X)        # per-instance scale

            ranks = np.array(ast.literal_eval(str(row["rank_vector_dtime"])))  # target ranks
            X_list.append(torch.tensor(X, dtype=torch.float32))
            y_list.append(ranks)
            names.append(idx)
        except Exception as e:
            print(f"⚠️ Skipping row {idx} due to parse error: {e}")
    return X_list, y_list, names

# =========================
# Main
# =========================
if __name__ == "__main__":
    set_seed(123)

    # Load model
    device = torch.device("cuda")
    model = DeepSetsRanker(input_size=input_size, hidden_size=128, dropout=0.3)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load data
    df = pd.read_excel(DATA_PATH)
    X_all, y_all, _ = load_instances(df, input_feature_cols)

    # Baseline predictions & Spearman
    baseline_preds = [predict_scores(model, x.to(device)) for x in X_all]
    baseline_rhos = [spearman_safe(-p, t) for p, t in zip(baseline_preds, y_all)]
    baseline_perf = np.nanmean(baseline_rhos)
    print(f"Baseline Spearman ρ (mean across instances): {baseline_perf:.4f}")

    # Permutation importance (percent drop in ρ)
    n_repeats = 5  # increase to 10+ for more stable estimates
    importances = []

    for f_idx, col in enumerate(input_feature_cols):
        drops = []
        for _ in range(n_repeats):
            perturbed_preds = []
            for x in X_all:
                x_pert = x.clone()
                # permute feature within the instance (keeps marginal distribution)
                idx_perm = torch.randperm(x_pert.size(0))
                x_pert[:, f_idx] = x_pert[idx_perm, f_idx]
                perturbed_preds.append(predict_scores(model, x_pert.to(device)))
            perf = np.nanmean([spearman_safe(-p, t) for p, t in zip(perturbed_preds, y_all)])
            drop = 0.0 if (baseline_perf is None or np.isnan(baseline_perf) or baseline_perf == 0) else \
                   100.0 * (baseline_perf - perf) / abs(baseline_perf)
            drops.append(drop)
        importances.append((col, np.mean(drops), np.std(drops)))

    # Report
    importances.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Feature Importance by % Spearman ρ Drop (mean ± std over repeats) ---")
    for col, mean_drop, std_drop in importances:
        print(f"{col:<24} {mean_drop:6.2f}%  ± {std_drop:5.2f}%")