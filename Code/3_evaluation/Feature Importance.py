"""Permutation importance for a trained DGSF single-machine ranker.

The model class and feature list are imported from the training script supplied
on the command line.  This avoids duplicating a neural-network architecture in
the evaluator and keeps the analysis tied to the checkpoint being studied.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DGSF feature-permutation importance.")
    parser.add_argument("--model-script", type=Path, required=True, help="Training script defining the model class.")
    parser.add_argument("--model-class", default="SMSDev9LeanRanker", help="Class to import from --model-script.")
    parser.add_argument("--type-a-data", type=Path, required=True)
    parser.add_argument("--type-a-model", type=Path, required=True)
    parser.add_argument("--type-b-data", type=Path, required=True)
    parser.add_argument("--type-b-model", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5, help="Within-instance shuffles per feature.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--heads", type=int, default=8, help="Attention-head count used by the checkpoint.")
    return parser.parse_args()


def import_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model definition from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def parse_array(value: object) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return checkpoint


def build_model(module: Any, class_name: str, checkpoint: Path, feature_cols: list[str], heads: int,
                device: torch.device) -> torch.nn.Module:
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise AttributeError(f"{checkpoint.name}: {class_name} is not defined in the supplied model script.")
    checkpoint_data = load_checkpoint(checkpoint, device)
    state = checkpoint_data.get("model_state_dict", checkpoint_data.get("state_dict", checkpoint_data))
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported model state in {checkpoint}")
    input_weight = state.get("job_enc.phi.0.weight")
    if input_weight is None:
        raise KeyError(f"{checkpoint.name}: cannot infer input size from job_enc.phi.0.weight")
    hidden, input_size = input_weight.shape
    if input_size != len(feature_cols):
        raise ValueError(
            f"{checkpoint.name}: checkpoint expects {input_size} features, but the imported model defines "
            f"{len(feature_cols)}."
        )
    checkpoint_features = checkpoint_data.get("feature_cols")
    if checkpoint_features is not None and list(checkpoint_features) != feature_cols:
        raise ValueError(f"{checkpoint.name}: imported feature list differs from checkpoint metadata.")
    kwargs = dict(checkpoint_data.get("model_kwargs", {}))
    kwargs.setdefault("job_in", input_size)
    kwargs.setdefault("hidden", hidden)
    kwargs.setdefault("dropout", 0.0)
    kwargs.setdefault("heads", heads)
    model = model_cls(**kwargs)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def prepare_instances(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    missing = [column for column in feature_cols + ["rank_vector_dtime"] if column not in df.columns]
    if missing:
        raise KeyError(f"Data are missing required columns: {missing}")
    inputs, reference_ranks = [], []
    for row_index, row in df.iterrows():
        x = np.column_stack([parse_array(row[column]) for column in feature_cols]).astype(np.float32)
        ranks = parse_array(row["rank_vector_dtime"])
        if x.shape[0] != len(ranks):
            raise ValueError(f"Row {row_index}: feature and rank-vector lengths do not agree.")
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
        inputs.append(x)
        reference_ranks.append(ranks)
    return inputs, reference_ranks


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    features = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones((1, x.shape[0]), dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(features, mask)
    scores = output["schedule_scores"] if isinstance(output, dict) else output
    return scores.squeeze(0).detach().cpu().numpy()


def spearman_to_reference(scores: np.ndarray, ranks: np.ndarray) -> float:
    value = spearmanr(-scores, ranks).correlation
    return float(value) if value is not None else float("nan")


def run_family(label: str, data_path: Path, model_path: Path, module: Any, args: argparse.Namespace,
               feature_cols: list[str], device: torch.device, seed_offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = build_model(module, args.model_class, model_path, feature_cols, args.heads, device)
    inputs, ranks = prepare_instances(read_table(data_path), feature_cols)
    baseline_scores = [predict(model, x, device) for x in inputs]
    baseline_rhos = np.asarray([spearman_to_reference(scores, target) for scores, target in zip(baseline_scores, ranks)])
    baseline = float(np.nanmean(baseline_rhos))
    if not np.isfinite(baseline) or baseline == 0.0:
        raise ValueError(f"{label}: invalid baseline Spearman value {baseline}.")

    rng = np.random.default_rng(args.seed + seed_offset)
    rows = []
    for feature_index, feature in enumerate(feature_cols):
        repeat_rhos = []
        for _ in range(args.repeats):
            perturbed = []
            for x, target in zip(inputs, ranks):
                shuffled = x.copy()
                shuffled[:, feature_index] = shuffled[rng.permutation(len(shuffled)), feature_index]
                perturbed.append(spearman_to_reference(predict(model, shuffled, device), target))
            repeat_rhos.append(float(np.nanmean(perturbed)))
        drops = 100.0 * (baseline - np.asarray(repeat_rhos)) / abs(baseline)
        rows.append(
            {
                "type": label,
                "instances": len(inputs),
                "feature": feature,
                "baseline_spearman": baseline,
                "perturbed_spearman_mean": float(np.mean(repeat_rhos)),
                "importance_drop_percent_mean": float(np.mean(drops)),
                "importance_drop_percent_std": float(np.std(drops, ddof=1)),
                "repeats": args.repeats,
                "model_file": model_path.name,
                "data_file": data_path.name,
            }
        )
    importance = pd.DataFrame(rows).sort_values("importance_drop_percent_mean", ascending=False)
    per_instance = pd.DataFrame({"type": label, "instance": np.arange(len(inputs)), "baseline_spearman": baseline_rhos})
    return importance, per_instance


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to report a standard deviation.")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    module = import_module(args.model_script)
    feature_cols = list(getattr(module, "SMS_FEATURE_COLS"))
    type_a, type_a_instances = run_family("Type A", args.type_a_data, args.type_a_model, module, args,
                                          feature_cols, device, seed_offset=0)
    type_b, type_b_instances = run_family("Type B", args.type_b_data, args.type_b_model, module, args,
                                          feature_cols, device, seed_offset=10_000)
    importance = pd.concat([type_a, type_b], ignore_index=True)
    per_instance = pd.concat([type_a_instances, type_b_instances], ignore_index=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.output_csv, index=False, float_format="%.6f")
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        importance.to_excel(writer, sheet_name="Importance", index=False)
        per_instance.to_excel(writer, sheet_name="Instance Spearman", index=False)
    print(importance.to_string(index=False))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_xlsx}")


if __name__ == "__main__":
    main()
