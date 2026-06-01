# -*- coding: utf-8 -*-
"""
Load ML model and apply local swap heuristic

"""

import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import math
import time
import argparse

# =============================
# Model: DeepSetsRanker
# =============================
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

        if mask is None:
            return out.mean(dim=1)                                                   # [B,D]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)                         # [B,1]
        return (out * mask.unsqueeze(-1)).sum(dim=1) / denom                         # [B,D]


class DeepSetsRanker(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.3, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.Sequential(
            ResidualBlock(hidden_size, dropout),
            ResidualBlock(hidden_size, dropout),
        )
        self.attn_pool = MultiHeadAttentionPooling(hidden_size, num_heads=num_heads)
        self.rank_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        h = self.embedding(x)            # [B,N,H]
        h = self.encoder(h)              # [B,N,H]
        g = self.attn_pool(h, mask)      # [B,H]
        g = g.unsqueeze(1).expand_as(h)  # [B,N,H]
        h = h + g
        scores = self.rank_head(h).squeeze(-1)  # [B,N]
        return {"schedule_scores": scores}

# -----------------------------
# Tardiness + Swap Postprocessing
# -----------------------------
def compute_tardiness(rho, tau, eps, ranks, batch_list):
    I = len(ranks)
    predicted_ranks = {batch_list[i]: ranks[i] for i in range(I)}
    sorted_jobs = sorted(batch_list, key=lambda x: predicted_ranks[x])
    sorted_indices = [int(job.replace('job', '')) - 1 for job in sorted_jobs]

    ordered_rho = rho[sorted_indices]
    ordered_tau = tau[sorted_indices]
    ordered_eps = eps[sorted_indices]

    start_times = np.zeros(I)
    start_times[0] = ordered_rho[0]
    for i in range(1, I):
        start_times[i] = max(ordered_rho[i], start_times[i - 1] + ordered_tau[i - 1])

    completion_times = start_times + ordered_tau
    tardiness = np.maximum(completion_times - ordered_eps, 0)
    return float(np.sum(tardiness))


def refine_by_local_swaps(rho, tau, eps, ranks, batch_list, max_lookahead=4):
    improved = True
    current_ranks = ranks.copy()
    best_tardiness = compute_tardiness(rho, tau, eps, current_ranks, batch_list)

    while improved:
        improved = False
        best_swap = None
        best_swap_tardiness = best_tardiness

        for i in range(len(current_ranks)):
            for offset in range(1, max_lookahead + 1):
                j = i + offset
                if j >= len(current_ranks):
                    continue

                swapped = current_ranks.copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                new_tardiness = compute_tardiness(rho, tau, eps, swapped, batch_list)

                if new_tardiness < best_swap_tardiness:
                    best_swap = swapped
                    best_swap_tardiness = new_tardiness

        if best_swap is not None:
            current_ranks = best_swap
            best_tardiness = best_swap_tardiness
            improved = True

    return current_ranks, best_tardiness

# -----------------------------
# Main Inference Loop
# -----------------------------
parser = argparse.ArgumentParser(description="Run DeepSets inference and local-swap refinement.")
parser.add_argument("--input", default="Updated_file.xlsx", help="Input Excel file.")
parser.add_argument("--model", default="Model_name.pth", help="Trained PyTorch model path.")
parser.add_argument("--output", default="Post_ML_file.xlsx", help="Output Excel file.")
parser.add_argument("--max-lookahead", type=int, default=4, help="Maximum local-swap lookahead.")
args = parser.parse_args()

df = pd.read_excel(args.input)

# Input features must align with ones used to train the model
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

model = DeepSetsRanker(input_size=input_size)
model.load_state_dict(torch.load(args.model, map_location="cpu"))
model.eval()

total_start_time = time.time()

ml_rank_values = []
ml_tardiness_values = []
ml_inference_times = []

rank_swap_values = []
tardiness_swap_values = []
post_swap_times = []

for _, row in df.iterrows():
    I = int(row["I"])
    batch_list = [f"job{i+1}" for i in range(I)]

    feature_vectors = [
        np.array(ast.literal_eval(str(row[col]))).reshape(-1, 1)
        for col in input_feature_cols
    ]
    features = np.concatenate(feature_vectors, axis=1)
    features = StandardScaler().fit_transform(features)
    X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    ml_start_time = time.time()
    with torch.no_grad():
        scores = model(X_tensor)['schedule_scores'][0]
    predicted_ranks = torch.argsort(torch.argsort(-scores)).cpu().numpy() + 1
    ml_duration = time.time() - ml_start_time

    rho = np.array(ast.literal_eval(str(row["rho"]))).flatten()
    tau = np.array(ast.literal_eval(str(row["tau"]))).flatten()
    eps = np.array(ast.literal_eval(str(row["eps"]))).flatten()

    ml_tardiness = compute_tardiness(rho, tau, eps, predicted_ranks, batch_list)
    
    # Apply local pairwise swap
    post_start_time = time.time()
    best_ranks, best_tardiness = refine_by_local_swaps(
        rho, tau, eps, predicted_ranks, batch_list, max_lookahead=args.max_lookahead
    )
    post_duration = time.time() - post_start_time

    ml_rank_values.append(predicted_ranks.tolist())
    ml_tardiness_values.append(ml_tardiness)
    ml_inference_times.append(ml_duration)

    rank_swap_values.append(best_ranks.tolist())
    tardiness_swap_values.append(best_tardiness)
    post_swap_times.append(post_duration)

df["ml_rank"] = ml_rank_values
df["tardiness_ml"] = ml_tardiness_values
df["ml_inference_time"] = ml_inference_times

df["rank_swap"] = rank_swap_values
df["tardiness_swap"] = tardiness_swap_values
df["postprocessing_time_swap"] = post_swap_times

output_filename = args.output
df.to_excel(output_filename, index=False)
print(f"Saved results to {output_filename} in {time.time() - total_start_time:.2f}s")
