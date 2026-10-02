#!/usr/bin/env python
"""Analyze cross-validation results.

Reads ``<results_dir>/*_cv_metrics.json``, then:
  * builds a summary table (mean +/- std across folds),
  * runs paired significance tests between every architecture and its
    corresponding transfer-learning (feature extraction + SVM) counterpart,
  * renders figures (accuracy bars, paired bars per metric, per-fold lines,
    confusion matrices) into ``<results_dir>/figures``,
  * writes ``<results_dir>/statistical_tests.csv`` and a text summary.

Usage:
    python scripts/analyze_results.py --results-dir brain_tumor_results
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision (weighted)",
    "recall": "Recall (weighted)",
    "f1": "F1 (macro)",
    "roc_auc": "ROC-AUC",
}

PAIRS = [
    ("googlenet", "googlenet+svm"),
    ("resnet18", "resnet18+svm"),
    ("vgg19", "vgg19+svm"),
]

MODEL_ORDER = [
    "cnn_baseline",
    "googlenet",
    "resnet18",
    "vgg19",
    "googlenet+svm",
    "resnet18+svm",
    "vgg19+svm",
]

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

COLOR_FINETUNE = "#2a6fdb"
COLOR_TRANSFER = "#e08a1e"
COLOR_BASELINE = "#7f8c8d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CV results.")
    parser.add_argument("--results-dir", default="brain_tumor_results")
    return parser.parse_args()


def load_payloads(results_dir: Path) -> dict:
    payloads = {}
    for path in sorted(results_dir.glob("*_cv_metrics.json")):
        with open(path) as handle:
            payload = json.load(handle)
        if "model" in payload and "per_fold" in payload:
            payloads[payload["model"]] = payload
    if not payloads:
        raise SystemExit(f"No *_cv_metrics.json found in {results_dir}")
    return payloads


def summary_table(payloads: dict) -> pd.DataFrame:
    rows = []
    for model, payload in payloads.items():
        agg = payload["aggregate"]
        row = {
            "model": model,
            "approach": payload["approach"],
            "folds": payload["num_folds"],
        }
        for key in METRIC_KEYS:
            row[key] = agg[f"{key}_mean"]
            row[f"{key}_std"] = agg[f"{key}_std"]
        rows.append(row)

    order = [m for m in MODEL_ORDER if m in [r["model"] for r in rows]]
    order += [r["model"] for r in rows if r["model"] not in order]
    return pd.DataFrame(rows).set_index("model").loc[order].reset_index()


def per_fold_series(payload: dict, metric: str) -> np.ndarray:
    folds = sorted(payload["per_fold"], key=lambda f: f["fold"])
    return np.array([f["metrics"][metric] for f in folds], dtype=float)


def sign_flip_permutation(diffs: np.ndarray, alternative: str = "two-sided") -> float:
    """Exact paired sign-flip permutation test (2^n sign vectors)."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    observed = diffs.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    perm_means = (signs @ diffs) / n

    tolerance = 1e-12
    if alternative == "two-sided":
        return float(np.mean(np.abs(perm_means) >= abs(observed) - tolerance))
    return float(np.mean(perm_means >= observed - tolerance))


def cohens_d_paired(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return float("inf") if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / sd)


def holm_adjust(pvalues: list) -> list:
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def run_tests(payloads: dict) -> pd.DataFrame:
    rows = []
    for fine, transfer in PAIRS:
        if fine not in payloads or transfer not in payloads:
            continue
        for metric in METRIC_KEYS:
            x = per_fold_series(payloads[fine], metric)
            y = per_fold_series(payloads[transfer], metric)
            diff = x - y

            t_two = stats.ttest_rel(x, y)
            t_greater = stats.ttest_rel(x, y, alternative="greater")
            try:
                w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                w_stat, w_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")

            rows.append(
                {
                    "architecture": fine,
                    "transfer": transfer,
                    "metric": metric,
                    "fine_tune_mean": float(x.mean()),
                    "transfer_mean": float(y.mean()),
                    "mean_diff": float(diff.mean()),
                    "pct_diff": float(diff.mean() / y.mean() * 100) if y.mean() else float("nan"),
                    "t_stat": float(t_two.statistic),
                    "p_ttest_two_sided": float(t_two.pvalue),
                    "p_ttest_greater": float(t_greater.pvalue),
                    "wilcoxon_stat": w_stat,
                    "p_wilcoxon": w_p,
                    "p_perm_two_sided": sign_flip_permutation(diff, "two-sided"),
                    "p_perm_greater": sign_flip_permutation(diff, "greater"),
                    "cohens_d_paired": cohens_d_paired(diff),
                    "n_folds": len(diff),
                }
            )

    table = pd.DataFrame(rows)
    table["p_perm_two_sided_holm"] = holm_adjust(table["p_perm_two_sided"].tolist())
    table["p_perm_greater_holm"] = holm_adjust(table["p_perm_greater"].tolist())
    return table


def figure_all_models(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ordered = summary.sort_values("accuracy", ascending=True)

    for ax, metric in zip(axes, ["accuracy", "roc_auc"]):
        colors = [
            COLOR_BASELINE if m == "cnn_baseline"
            else (COLOR_FINETUNE if row.approach == "fine_tune" else COLOR_TRANSFER)
            for m, row in ordered.set_index("model").iterrows()
        ]
        y = np.arange(len(ordered))
        ax.barh(
            y,
            ordered[metric],
            xerr=ordered[f"{metric}_std"],
            color=colors,
            capsize=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(ordered["model"])
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} (mean +/- std over folds)")
        ax.set_xlim(0, 1.05)
        for i, value in enumerate(ordered[metric]):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BASELINE),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FINETUNE),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TRANSFER),
    ]
    fig.legend(
        handles,
        ["Baseline (from scratch)", "Fine-tuned CNN", "CNN features + SVM"],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = out_dir / "all_models_accuracy_roc_auc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_pairs(summary: pd.DataFrame, tests: pd.DataFrame, out_dir: Path) -> None:
    lookup = summary.set_index("model")
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(5 * len(PAIRS), 5), sharey=True)
    metric = "accuracy"
    width = 0.38

    for ax, (fine, transfer) in zip(axes, PAIRS):
        values = [
            lookup.loc[fine, metric], lookup.loc[transfer, metric]
        ]
        errors = [
            lookup.loc[fine, f"{metric}_std"], lookup.loc[transfer, f"{metric}_std"]
        ]
        x = np.arange(2)
        ax.bar(
            x - width / 2, [values[0]], yerr=[errors[0]], width=width,
            color=COLOR_FINETUNE, capsize=4, label="Fine-tuned",
        )
        ax.bar(
            x + width / 2, [values[1]], yerr=[errors[1]], width=width,
            color=COLOR_TRANSFER, capsize=4, label="Features + SVM",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([fine, transfer], rotation=10)
        ax.set_title(f"{fine} vs {transfer}")
        ax.set_ylim(0, 1.08)

        row = tests[(tests.architecture == fine) & (tests.metric == metric)]
        if not row.empty:
            p_two = float(row.iloc[0]["p_ttest_two_sided"])
            p_greater = float(row.iloc[0]["p_ttest_greater"])
            stars = "***" if p_two < 0.001 else "**" if p_two < 0.01 else "*" if p_two < 0.05 else "ns"
            ax.text(
                0.5, 1.01,
                f"paired t-test p={p_two:.4f} ({stars}); one-sided p={p_greater:.4f}",
                ha="center", va="bottom", fontsize=8, transform=ax.transAxes,
            )

    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Fine-tuned CNN vs transfer learning (features + SVM) - accuracy", y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = out_dir / "architecture_pairs_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def figure_metric_grid(summary: pd.DataFrame, out_dir: Path) -> None:
    lookup = summary.set_index("model")
    metrics = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    width = 0.36
    x = np.arange(len(PAIRS))

    for ax, metric in zip(axes.ravel(), metrics):
        fine_vals = [lookup.loc[f, metric] for f, _ in PAIRS]
        trans_vals = [lookup.loc[t, metric] for _, t in PAIRS]
        fine_err = [lookup.loc[f, f"{metric}_std"] for f, _ in PAIRS]
        trans_err = [lookup.loc[t, f"{metric}_std"] for _, t in PAIRS]
        ax.bar(x - width / 2, fine_vals, yerr=fine_err, width=width,
               color=COLOR_FINETUNE, capsize=3, label="Fine-tuned")
        ax.bar(x + width / 2, trans_vals, yerr=trans_err, width=width,
               color=COLOR_TRANSFER, capsize=3, label="Features + SVM")
        ax.set_xticks(x)
        ax.set_xticklabels([f for f, _ in PAIRS])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, 1.08)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Per-architecture comparison across metrics (mean +/- std over 5 folds)", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = out_dir / "architecture_metric_grid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_per_fold(payloads: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(5 * len(PAIRS), 4.5), sharey=True)
    for ax, (fine, transfer) in zip(axes, PAIRS):
        folds = np.arange(len(payloads[fine]["per_fold"])) + 1
        ax.plot(folds, per_fold_series(payloads[fine], "accuracy"), "o-",
                color=COLOR_FINETUNE, label="Fine-tuned")
        ax.plot(folds, per_fold_series(payloads[transfer], "accuracy"), "s--",
                color=COLOR_TRANSFER, label="Features + SVM")
        ax.set_title(f"{fine} vs {transfer}")
        ax.set_xlabel("Fold")
        ax.set_ylim(0.8, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    fig.suptitle("Per-fold accuracy", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "per_fold_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_confusion_matrices(payloads: dict, out_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in payloads]
    cols = 4
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.array(axes).ravel()

    for ax, model in zip(axes, models):
        cm = np.array(payloads[model]["confusion_matrix"], dtype=float)
        cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, rotation=20)
        ax.set_yticks(range(3))
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_title(model)
        if ax is axes[0]:
            ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    for ax in axes[len(models):]:
        ax.axis("off")

    fig.suptitle("Pooled out-of-fold confusion matrices", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def write_summary(summary: pd.DataFrame, tests: pd.DataFrame, out_dir: Path) -> None:
    summary_path = out_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    tests_path = out_dir / "statistical_tests.csv"
    tests.to_csv(tests_path, index=False)
    print(f"Saved {tests_path}")

    lines = []
    lines.append("Paired significance tests: fine-tuned vs transfer (features + SVM)")
    lines.append("=" * 72)
    for _, row in tests.iterrows():
        lines.append(
            f"{row['architecture']:>9} | {row['metric']:<17} | "
            f"FT={row['fine_tune_mean']:.4f} SVM={row['transfer_mean']:.4f} "
            f"diff=+{row['mean_diff']:.4f} ({row['pct_diff']:.2f}%) | "
            f"t p2={row['p_ttest_two_sided']:.4f} p1={row['p_ttest_greater']:.4f} | "
            f"perm p2={row['p_perm_two_sided']:.4f} p1={row['p_perm_greater']:.4f} "
            f"(holm p1={row['p_perm_greater_holm']:.4f}) | d={row['cohens_d_paired']:.2f}"
        )
    text = "\n".join(lines)
    (out_dir / "statistical_tests.txt").write_text(text + "\n")
    print(f"Saved {out_dir / 'statistical_tests.txt'}")
    print("\n" + text)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_payloads(results_dir)
    summary = summary_table(payloads)
    tests = run_tests(payloads)

    pd.set_option("display.float_format", lambda value: f"{value:.4f}")
    print("=" * 72)
    print("Summary (mean over folds)")
    print("=" * 72)
    print(summary[["model", "approach", "folds", "accuracy", "f1", "roc_auc"]].to_string(index=False))

    figure_all_models(summary, figures_dir)
    figure_pairs(summary, tests, figures_dir)
    figure_metric_grid(summary, figures_dir)
    figure_per_fold(payloads, figures_dir)
    figure_confusion_matrices(payloads, figures_dir)
    write_summary(summary, tests, results_dir)


if __name__ == "__main__":
    main()
