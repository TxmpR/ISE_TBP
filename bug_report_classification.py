"""
Bug report classification experiments.

Runs the lab baseline, TF-IDF + Naive Bayes, against two simple linear
alternatives. Results are written to CSV/PNG files so they can be checked and
reused in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC



TEXT_COLUMNS = ("title", "body", "labels", "comments", "codes", "commands")
METRICS = ("precision", "recall", "f1")
MODEL_NAMES = ("NB", "LR", "SVM")


def cliff_delta(left: np.ndarray, right: np.ndarray) -> float:
    """Return Cliff's delta for two score lists."""
    if len(left) ==0 or len(right) == 0:
        return 0.0

    wins = 0
    losses = 0
    for x in left:
        wins += int(np.sum(x > right))
        losses += int(np.sum(x < right))
    return (wins - losses) / (len(left) * len(right))


def read_csvs(folder: Path) -> dict[str, pd.DataFrame]:
    """Read every project CSV in the data folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder.resolve()}")

    tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df.columns = [name.strip().lower() for name in df.columns]

        if "class" not in df.columns:
            raise ValueError(f"{csv_path.name} has no 'class' column")

        df["class"] = pd.to_numeric(df["class"], errors="raise").astype(int)
        bad_labels = sorted(set(df["class"]) - {0, 1})
        if bad_labels:
            raise ValueError(f"{csv_path.name} has labels outside 0/1: {bad_labels}")

        df["project"] = csv_path.stem
        tables[csv_path.stem] = df

    if not tables:
        raise FileNotFoundError(f"No CSV files found in {folder.resolve()}")
    return tables


def merge_text(df: pd.DataFrame) -> pd.Series:
    """Join the report fields used by the classifier."""
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    available = [col for col in TEXT_COLUMNS if col in df.columns]
    if not available:
        raise ValueError(f"Expected at least one text column from: {', '.join(TEXT_COLUMNS)}")

    text = df.reindex(columns=TEXT_COLUMNS, fill_value="")
    return text.fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def prep_data(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one tidy table with project, text and class columns."""
    parts = []
    for project, df in tables.items():
        part = pd.DataFrame(
            {
                "project": project,
                "text": merge_text(df),
                "class": df["class"].astype(int),
            }
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def check_split(df: pd.DataFrame, label: str, test_size: float) -> None:
    """Fail early when a stratified split would be invalid."""
    counts = df["class"].value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError(f"{label} must contain both classes 0 and 1")
    
    if counts.min()< 2:
        raise ValueError(f"{label} needs at least two rows in each class")

    smallest_test_class =int(np.floor(counts.min() * test_size))
    if smallest_test_class< 1:
        
        raise ValueError(
            f"{label} has too few minority-class rows for test_size={test_size}. "
            "Use a larger dataset or a larger test split."
        )


def make_vectorizer(min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(

        ngram_range= (1, 2),
        min_df=min_df,
        max_df= 0.95,
        sublinear_tf=True,
        max_features= 50_000,
    )


def make_models(seed: int) -> dict[str, Any]:
    return {

        "NB": MultinomialNB(),
        "LR": LogisticRegression(
            class_weight="balanced",
            max_iter= 5000,
            random_state=seed,
        ),
        "SVM": LinearSVC(
            class_weight="balanced",
            max_iter= 10000,
            random_state=seed,
        ),
    }


def score_run(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float |int]:
    tn, fp, fn, tp =confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {

        "precision":precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp ),
        "fp": int(fp ),
        "fn": int(fn ),
        "tn": int(tn ),
    }


def run_splits(
        
    df: pd.DataFrame,
    label: str,
    n_runs: int,
    seed: int,
    test_size: float,
    min_df: int,
) -> pd.DataFrame:
    """Run repeated stratified train/test splits for one dataset"""
    check_split(df, label, test_size)
    rows: list[dict[str, Any]] = []

    for run in range(n_runs):
        split_seed = seed + run
        print(f"[{label}] run {run + 1}/{n_runs}", flush= True)
        x_train, x_test, y_train, y_test =train_test_split(
            df["text" ],
            df["class" ],
            test_size=test_size,
            stratify=df["class" ],
            random_state=split_seed,
        )

        vectorizer = make_vectorizer(min_df )
        x_train_vec = vectorizer.fit_transform( x_train )
        x_test_vec = vectorizer.transform( x_test)

        for model_name, model in make_models(split_seed).items():
            model.fit(x_train_vec, y_train)
            scores = score_run(y_test, model.predict(x_test_vec))
            rows.append(
                {
                    "scope":label,
                    "run":run + 1,
                    "split_seed":split_seed,
                    "model":model_name,
                    **scores,
                }
            )

    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby(["scope", "model"], as_index=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    return summary.sort_values(["scope", "f1_mean"], ascending=[True, False])


def compare(
        
    results: pd.DataFrame,
    baseline: str,
    proposed: str,
    scope: str = "all",
) -> pd.DataFrame:
    """Compare two models on matched runs"""
    rows = []
    scoped = results[results["scope"] == scope]

    for metric in METRICS:
        scores = scoped.pivot(index="run", columns="model", values=metric)
        baseline_scores = scores[baseline].to_numpy()
        proposed_scores = scores[proposed].to_numpy()

        try:
            _, p_value = wilcoxon(
                proposed_scores,
                baseline_scores,
                alternative="greater",
                zero_method="zsplit",
            )
        except ValueError:
            p_value = 1.0

        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "baseline": baseline,
                "proposed": proposed,
                "baseline_mean": baseline_scores.mean(),
                "proposed_mean": proposed_scores.mean(),
                "mean_difference": proposed_scores.mean() - baseline_scores.mean(),
                "p_value": p_value,
                "cliffs_delta": cliff_delta(proposed_scores, baseline_scores),
                "proposed_wins": int(np.sum(proposed_scores > baseline_scores)),
                "ties": int(np.sum(proposed_scores == baseline_scores)),
                "baseline_wins": int(np.sum(proposed_scores < baseline_scores)),
            }
        )

    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
    }
    versions = {}
    for label, package_name in names.items():
        try:
            versions[label] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[label] = "not installed"
    return versions


def save_metadata(args: argparse.Namespace, csv_paths: list[Path], output_dir: Path) -> None:
    meta = {
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "settings": vars(args),
        "data_files": [
            {
                "file": str(path),
                "sha256": sha256(path),
            }
            for path in csv_paths
        ],
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def save_tables(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "per_run_results.csv", index=False)
    summary[summary["scope"] == "all"].to_csv(output_dir / "overall_summary.csv", index=False)
    summary[summary["scope"] != "all"].to_csv(output_dir / "per_project_summary.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)


def plot_scores(results: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    overall = summary[summary["scope"] == "all"].set_index("model").loc[list(MODEL_NAMES)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(overall.index, overall["f1_mean"], yerr=overall["f1_std"], capsize=4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1" )
    ax.set_title("Overall F1 by model" )
    for idx, value in enumerate(overall["f1_mean" ]):
        ax.text(idx, min(value + 0.02, 0.98), f"{value:.3f}", ha="center" )
    fig.tight_layout()
    fig.savefig(output_dir / "overall_f1_bar.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [
        results[(results["scope"] == "all") & (results["model"] == model)]["f1"]
        for model in MODEL_NAMES
    ]
    ax.boxplot(data)
    ax.set_xticks(range(1, len(MODEL_NAMES) + 1))
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("F1 over repeated splits")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_f1_box.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run bug report classification experiments.")
    parser.add_argument("--data_dir", default="datasets", help="Folder containing the lab CSV files")
    parser.add_argument("--output_dir", default="results", help="Folder for CSV and plot outputs")
    parser.add_argument("--n_runs", type=int, default=30, help="Number of repeated splits")
    parser.add_argument("--seed", type=int, default=7, help="First random seed")
    parser.add_argument("--test_size", type=float, default=0.30, help="Test split size")
    parser.add_argument("--min_df", type=int, default=2, help="Minimum document frequency for TF-IDF")
    parser.add_argument("--baseline", choices=MODEL_NAMES, default="NB")
    parser.add_argument("--proposed", choices=MODEL_NAMES, default="SVM")
    parser.add_argument("--no_plots", action="store_true", help="Skip PNG plot generation")
    return parser.parse_args()


def main() -> None:

    args= parse_args()
    if args.n_runs < 1:
        raise ValueError("--n_runs must be at least 1")
    if not 0 < args.test_size < 1:
        raise ValueError("--test_size must be between 0 and 1")
    if args.baseline == args.proposed:
        raise ValueError("--baseline and --proposed must be different models")

    data_dir= Path(args.data_dir)
    output_dir= Path(args.output_dir)
    csv_paths= sorted(data_dir.glob("*.csv"))

    tables= read_csvs(data_dir)
    data= prep_data(tables)

    all_results = [run_splits(data, "all", args.n_runs, args.seed, args.test_size, args.min_df)]
    for project in sorted(data["project"].unique()):
        project_data = data[data["project"] == project].reset_index(drop=True)
        all_results.append(
            run_splits(project_data, project, args.n_runs, args.seed, args.test_size, args.min_df)
        )

    results = pd.concat(all_results, ignore_index=True)
    summary = summarise(results)
    comparison = compare(results, args.baseline, args.proposed)

    save_tables(results, summary, comparison, output_dir)
    save_metadata(args, csv_paths, output_dir)
    if not args.no_plots:
        plot_scores(results, summary, output_dir)

    print("\nOverall summary" )

    print(summary[summary["scope"] =="all"].to_string(index=False, float_format="{:.3f}".format))

    print("\nBaseline comparison" )

    print(comparison.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSaved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
