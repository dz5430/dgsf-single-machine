# -*- coding: utf-8 -*-
"""
Generic SMS model evaluator.

Unlike the original Load_ML_swap.py, this script does not duplicate a model
definition. It loads supported model classes from their training modules and
chooses the architecture from checkpoint metadata when available.

Supported architectures:
  - dev9_lean: checkpoints from DeepSets_SMS_Scheduling_training_dev9_lean.py

Legacy checkpoints from the original SMS trainer can still be evaluated with
Load_ML_swap.py. That training script runs at import time, so it is not a clean
import target for this metadata-driven evaluator.
"""

import argparse
import ast
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SMS DGSF model and local-swap output.")
    parser.add_argument("--input", default="Updated_file.xlsx", help="Input CSV/XLSX file.")
    parser.add_argument("--model", required=True, help="Model checkpoint path.")
    parser.add_argument("--output", default="Post_ML_file.xlsx", help="Output CSV/XLSX file.")
    parser.add_argument("--architecture", choices=["auto", "dev9_lean"], default="auto")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-lookahead", type=int, default=4)
    parser.add_argument("--legacy-standard-scaler", action="store_true",
                        help="Use sklearn StandardScaler per instance, matching the original loader.")
    return parser.parse_args()


def import_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def parse_1d(cell) -> np.ndarray:
    return np.array(ast.literal_eval(str(cell)), dtype=float).flatten()


def prepare_features(row: pd.Series, feature_cols: list[str], use_standard_scaler: bool) -> np.ndarray:
    vectors = [parse_1d(row[col]).reshape(-1, 1) for col in feature_cols]
    features = np.concatenate(vectors, axis=1).astype(np.float32)
    if use_standard_scaler:
        return StandardScaler().fit_transform(features).astype(np.float32)
    mu = features.mean(axis=0, keepdims=True)
    sd = features.std(axis=0, keepdims=True) + 1e-8
    return ((features - mu) / sd).astype(np.float32)


def compute_tardiness(rho, tau, eps, ranks):
    n_jobs = len(ranks)
    order = np.argsort(np.asarray(ranks), kind="mergesort")
    time_now = 0.0
    total = 0.0
    for job_idx in order:
        start = max(time_now, float(rho[job_idx]))
        finish = start + float(tau[job_idx])
        total += max(0.0, finish - float(eps[job_idx]))
        time_now = finish
    return float(total)


def refine_by_local_swaps(rho, tau, eps, ranks, max_lookahead=4):
    current = np.asarray(ranks, dtype=int).copy()
    best_tardiness = compute_tardiness(rho, tau, eps, current)

    improved = True
    while improved:
        improved = False
        best_swap = None
        best_swap_tardiness = best_tardiness

        for i in range(len(current)):
            for offset in range(1, max_lookahead + 1):
                j = i + offset
                if j >= len(current):
                    continue
                swapped = current.copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                tardiness = compute_tardiness(rho, tau, eps, swapped)
                if tardiness < best_swap_tardiness:
                    best_swap = swapped
                    best_swap_tardiness = tardiness

        if best_swap is not None:
            current = best_swap
            best_tardiness = best_swap_tardiness
            improved = True

    return current, best_tardiness


def infer_architecture(raw_checkpoint, requested: str) -> str:
    if requested != "auto":
        return requested
    if isinstance(raw_checkpoint, dict) and raw_checkpoint.get("architecture"):
        return str(raw_checkpoint["architecture"])
    if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
        return "dev9_lean"
    raise ValueError(
        "Could not infer architecture. This evaluator expects metadata-rich "
        "Dev9-lean SMS checkpoints. Use Load_ML_swap.py for legacy original checkpoints."
    )


def load_model(model_path: str, architecture: str, device: torch.device):
    script_dir = Path(__file__).resolve().parent
    raw = torch.load(model_path, map_location=device)
    arch = infer_architecture(raw, architecture)

    if arch == "dev9_lean":
        module = import_from_file(
            "sms_dev9_lean_trainer",
            script_dir / "DeepSets_SMS_Scheduling_training_dev9_lean.py",
        )
        state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        feature_cols = raw.get("feature_cols", DEFAULT_FEATURE_COLS) if isinstance(raw, dict) else DEFAULT_FEATURE_COLS
        kwargs = (
            raw.get("model_kwargs")
            if isinstance(raw, dict) and isinstance(raw.get("model_kwargs"), dict)
            else {
                "job_in": len(feature_cols),
                "hidden": raw.get("hidden", 128) if isinstance(raw, dict) else 128,
                "dropout": raw.get("dropout", 0.2) if isinstance(raw, dict) else 0.2,
                "heads": raw.get("heads", 8) if isinstance(raw, dict) else 8,
            }
        )
        model = module.SMSDev9LeanRanker(**kwargs).to(device)
        model.load_state_dict(state)
        return model, arch, feature_cols

    raise ValueError(f"Unsupported architecture: {arch}")


def score_instance(model, architecture: str, features: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, features.shape[0], dtype=torch.float32, device=device)
    with torch.no_grad():
        if architecture == "dev9_lean":
            scores = model(x, mask)[0]
        else:
            scores = model(x, mask=mask)["schedule_scores"][0]
    return scores.detach().cpu().numpy()


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-scores)) + 1


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
    device = torch.device(args.device)

    model, architecture, feature_cols = load_model(args.model, args.architecture, device)
    model.eval()
    df = read_table(args.input)

    print(f"Loaded {architecture} model from {args.model}")
    print(f"Features: {feature_cols}")
    print(f"Rows: {len(df)}")

    ml_rank_values = []
    ml_tardiness_values = []
    ml_inference_times = []
    rank_swap_values = []
    tardiness_swap_values = []
    post_swap_times = []
    score_values = []

    total_start = time.time()
    for _, row in df.iterrows():
        features = prepare_features(
            row,
            feature_cols,
            use_standard_scaler=(args.legacy_standard_scaler or architecture == "original"),
        )
        t0 = time.time()
        scores = score_instance(model, architecture, features, device)
        ml_time = time.time() - t0

        predicted_ranks = ranks_from_scores(scores)
        rho = parse_1d(row["rho"])
        tau = parse_1d(row["tau"])
        eps = parse_1d(row["eps"])
        ml_tardiness = compute_tardiness(rho, tau, eps, predicted_ranks)

        post_start = time.time()
        best_ranks, best_tardiness = refine_by_local_swaps(
            rho, tau, eps, predicted_ranks, max_lookahead=args.max_lookahead
        )
        post_time = time.time() - post_start

        score_values.append(scores.tolist())
        ml_rank_values.append(predicted_ranks.tolist())
        ml_tardiness_values.append(ml_tardiness)
        ml_inference_times.append(ml_time)
        rank_swap_values.append(best_ranks.tolist())
        tardiness_swap_values.append(best_tardiness)
        post_swap_times.append(post_time)

    df["ml_scores"] = score_values
    df["ml_rank"] = ml_rank_values
    df["tardiness_ml"] = ml_tardiness_values
    df["ml_inference_time"] = ml_inference_times
    df["rank_swap"] = rank_swap_values
    df["tardiness_swap"] = tardiness_swap_values
    df["postprocessing_time_swap"] = post_swap_times
    df["model_architecture"] = architecture
    df["model_path"] = args.model

    write_table(df, args.output)
    print(f"Saved results to {args.output} in {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
