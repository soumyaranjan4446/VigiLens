"""
Results Summarizer for VigiLens Experiments.

Reads experiments/results.csv and computes mean +/- std across folds and seeds,
generating Markdown tables, paired comparisons against baseline, and anomaly-type breakdowns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd


def summarize_results(csv_path: Union[str, Path] = "experiments/results.csv",
                      group: Optional[str] = None,
                      run_id: Optional[str] = None) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    df = pd.read_csv(path)
    if group:
        df = df[df["group"] == group]
    if run_id:
        df = df[df["run_id"] == run_id]

    # Convert metric columns to numeric
    metric_cols = [
        "frame_auc", "frame_ap", "best_f1", "f1_at_thr", "fpr_at_95tpr",
        "normal_frame_fpr", "normal_video_fpr", "false_alarms_per_hour",
        "event_recall", "mean_delay_frames"
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def generate_markdown_summary(df: pd.DataFrame,
                              baseline_group: str = "baseline_v0") -> str:
    """Format experiment results into comparison markdown tables."""
    lines = ["# Experiment Results Summary", ""]

    # Filter to macro headline rows
    slice_col = "slice" if "slice" in df.columns else ("anomaly_type" if "anomaly_type" in df.columns else None)
    if slice_col:
        macro_df = df[df[slice_col] == "macro"]
        if macro_df.empty:
            macro_df = df
    else:
        macro_df = df

    # Group by run_id and modality
    grouped = macro_df.groupby(["run_id", "modality"]).agg({
        "frame_auc": ["mean", "std", "count"],
        "frame_ap": ["mean", "std"],
        "fpr_at_95tpr": ["mean", "std"],
        "normal_video_fpr": ["mean", "std"],
        "false_alarms_per_hour": ["mean", "std"],
        "event_recall": ["mean", "std"],
        "mean_delay_frames": ["mean", "std"],
    }).reset_index()

    lines.append("## Headline Metrics Across Folds (Mean ± Std)")
    lines.append("")
    lines.append("| Run ID | Modality | Folds/Seeds | Frame AUC | Frame AP | FPR@95%TPR | Normal Video FPR | False Alarms/hr | Event Recall | Mean Delay |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for _, row in grouped.iterrows():
        run_name = row[("run_id", "")]
        mod = row[("modality", "")]
        n_runs = int(row[("frame_auc", "count")])

        def fmt(col):
            m = row[(col, "mean")]
            s = row[(col, "std")]
            if np.isnan(m):
                return "NaN"
            if np.isnan(s) or s == 0.0:
                return f"{m:.4f}"
            return f"{m:.4f} ± {s:.4f}"

        auc_str = fmt("frame_auc")
        ap_str = fmt("frame_ap")
        fpr95_str = fmt("fpr_at_95tpr")
        vid_fpr_str = fmt("normal_video_fpr")
        fa_str = fmt("false_alarms_per_hour")
        ev_str = fmt("event_recall")
        del_str = fmt("mean_delay_frames")

        lines.append(
            f"| `{run_name}` | `{mod}` | {n_runs} | {auc_str} | {ap_str} | {fpr95_str} | {vid_fpr_str} | {fa_str} | {ev_str} | {del_str} |"
        )

    # Per-anomaly-type breakdown if present
    type_df = (
        df[df[slice_col].isin(["reverse", "speedup_x2", "speedup_x4", "freeze", "shuffle", "jumpcut"])]
        if slice_col and slice_col in df.columns
        else pd.DataFrame()
    )
    if not type_df.empty:
        lines.append("")
        lines.append("## Per-Anomaly-Type Breakdown (Frame AUC)")
        lines.append("")
        type_grouped = type_df.groupby(["run_id", "modality", slice_col])["frame_auc"].agg(["mean", "std"]).reset_index()
        lines.append("| Run ID | Modality | Anomaly Type | Frame AUC |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for _, row in type_grouped.iterrows():
            m = row["mean"]
            s = row["std"]
            val_str = f"{m:.4f} ± {s:.4f}" if np.isfinite(s) and s > 0 else (f"{m:.4f}" if np.isfinite(m) else "NaN")
            lines.append(f"| `{row['run_id']}` | `{row['modality']}` | `{row[slice_col]}` | {val_str} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VigiLens experiment results")
    parser.add_argument("--csv", default="experiments/results.csv")
    parser.add_argument("--group", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    df = summarize_results(args.csv, group=args.group, run_id=args.run_id)
    md = generate_markdown_summary(df)
    print(md)

    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
        print(f"\nWrote summary markdown to {args.out_md}")


if __name__ == "__main__":
    main()
