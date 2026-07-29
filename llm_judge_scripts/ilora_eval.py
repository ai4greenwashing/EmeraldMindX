"""Radar charts of the ILORA scores written by ilora_score.py.

Each chart plots one polygon per pipeline over the five ILORA criteria, using
the mean score across the claims of a dataset. Scores run from 1 to 5, so the
radial axis is fixed to that range and the polygons are directly comparable
across datasets.

Reads <input-dir>/<pipeline>/<dataset>.csv and writes one chart per dataset plus
a combined figure with the datasets side by side.

Example
-------
    python ilora_radar.py \\
        --input-dir path/to/ilora \\
        --output-dir path/to/figures \\
        --dataset green_claims:GreenClaims \\
        --dataset emerald_data:EmeraldData \\
        --pipelines baseline emx-rag emx-kgrag emx-voting emx-svoting
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Directory written by ilora_score.py, holding <pipeline>/<dataset>.csv files.
INPUT_DIR = "path/to/ilora"

# Directory for the generated charts.
OUTPUT_DIR = "path/to/figures"

CRITERIA = ("Informativeness", "Logicality", "Objectivity", "Readability", "Accuracy")

# Directory name -> label drawn in the legend.
PIPELINE_LABELS = {
    "baseline": "Baseline",
    "emx-rag": "EMX-RAG",
    "emx-kgrag": "EMX-KGRAG",
    "emx-voting": "EMX-VOTING",
    "emx-svoting": "EMX-SVOTING",
}

# Colourblind-safe palette (Okabe-Ito).
PIPELINE_COLORS = {
    "baseline": "#999999",
    "emx-rag": "#E69F00",
    "emx-kgrag": "#0072B2",
    "emx-voting": "#000066",
    "emx-svoting": "#009E73",
}

SCORE_MIN, SCORE_MAX = 1, 5


def parse_dataset_spec(spec: str) -> Dict[str, str]:
    """Parse ``DIRNAME`` or ``DIRNAME:DISPLAY NAME``."""
    name, _, display = spec.partition(":")
    if not name.strip():
        raise argparse.ArgumentTypeError(f"Empty dataset name in {spec!r}")
    return {"name": name.strip(), "display": (display or name).strip()}


def mean_scores(input_dir: str, pipeline: str, dataset: str) -> Optional[List[float]]:
    """Mean score per criterion, or None if the file is absent or incomplete."""
    path = os.path.join(input_dir, pipeline, f"{dataset}.csv")
    if not os.path.exists(path):
        print(f"    {pipeline}: no file at {path}")
        return None

    frame = pd.read_csv(path)
    missing = [c for c in CRITERIA if c not in frame.columns]
    if missing:
        print(f"    {pipeline}: missing column(s) {missing}")
        return None

    means = [float(frame[c].mean()) for c in CRITERIA]
    summary = "  ".join(f"{c[:3]}={m:.2f}" for c, m in zip(CRITERIA, means))
    print(
        f"    {PIPELINE_LABELS.get(pipeline, pipeline):<12s} "
        f"({len(frame):>4d} claims)  {summary}"
    )
    return means


def collect(
    input_dir: str, pipelines: List[str], dataset: str
) -> Dict[str, List[float]]:
    return {
        pipeline: means
        for pipeline in pipelines
        if (means := mean_scores(input_dir, pipeline, dataset)) is not None
    }


def criterion_angles() -> List[float]:
    """One angle per criterion, with the first repeated to close the polygon."""
    angles = [i / len(CRITERIA) * 2 * math.pi for i in range(len(CRITERIA))]
    return angles + angles[:1]


def draw_axis(ax, scale: float) -> None:
    """Grid, radial ticks and criterion labels for one polar axis."""
    angles = criterion_angles()

    ax.set_ylim(0, SCORE_MAX)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    ax.set_yticks(range(SCORE_MIN, SCORE_MAX + 1))
    ax.set_yticklabels(
        [str(v) for v in range(SCORE_MIN, SCORE_MAX + 1)], size=6 * scale, weight="bold"
    )

    ax.grid(True, color="#000000", linestyle="-", linewidth=0.3 * scale)
    ax.spines["polar"].set_edgecolor("#000000")
    ax.spines["polar"].set_linewidth(0.3 * scale)

    for angle, criterion in zip(angles[:-1], CRITERIA):
        alignment = "center"
        if 0.1 < angle < math.pi - 0.1:
            alignment = "left"
        elif math.pi + 0.1 < angle < 2 * math.pi - 0.1:
            alignment = "right"
        ax.text(
            angle,
            SCORE_MAX * 1.18,
            criterion,
            size=7 * scale,
            weight="bold",
            ha=alignment,
            va="center",
        )


def draw_polygons(ax, means_by_pipeline: Dict[str, List[float]], scale: float) -> List:
    angles = criterion_angles()
    handles = []
    for pipeline, means in means_by_pipeline.items():
        (line,) = ax.plot(
            angles,
            means + means[:1],
            "--",
            linewidth=1.1 * scale,
            color=PIPELINE_COLORS.get(pipeline, "#000000"),
            label=PIPELINE_LABELS.get(pipeline, pipeline),
            alpha=0.9,
        )
        handles.append(line)
    return handles


def single_chart(
    input_dir: str,
    pipelines: List[str],
    dataset: Dict[str, str],
    path: str,
    scale: float,
) -> None:
    print(f"\n  {dataset['display']}")
    means_by_pipeline = collect(input_dir, pipelines, dataset["name"])
    if not means_by_pipeline:
        print("    no data, chart skipped")
        return

    fig, ax = plt.subplots(figsize=(scale, scale), subplot_kw=dict(projection="polar"))
    draw_polygons(ax, means_by_pipeline, scale)
    draw_axis(ax, scale)
    ax.set_title(dataset["display"], size=9 * scale, weight="bold", pad=12 * scale)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        fontsize=7 * scale,
        frameon=True,
        edgecolor="#000000",
        framealpha=1.0,
    )
    for line in legend.get_lines():
        line.set_linewidth(1.5 * scale)

    save(fig, path)


def combined_chart(
    input_dir: str,
    pipelines: List[str],
    datasets: List[Dict[str, str]],
    path: str,
    scale: float,
) -> None:
    print("\n  Combined figure")
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(scale * len(datasets) * 1.15, scale),
        subplot_kw=dict(projection="polar"),
    )
    if len(datasets) == 1:
        axes = [axes]

    handles: Dict[str, object] = {}
    for ax, dataset in zip(axes, datasets):
        print(f"\n  {dataset['display']}")
        means_by_pipeline = collect(input_dir, pipelines, dataset["name"])
        for pipeline, line in zip(
            means_by_pipeline, draw_polygons(ax, means_by_pipeline, scale)
        ):
            handles.setdefault(PIPELINE_LABELS.get(pipeline, pipeline), line)
        draw_axis(ax, scale)
        ax.set_title(dataset["display"], size=9 * scale, weight="bold", pad=12 * scale)

    if handles:
        legend = fig.legend(
            handles.values(),
            handles.keys(),
            loc="lower center",
            ncol=len(handles),
            fontsize=7 * scale,
            frameon=True,
            edgecolor="#000000",
            framealpha=1.0,
        )
        for line in legend.get_lines():
            line.set_linewidth(1.5 * scale)

    fig.subplots_adjust(bottom=0.16, wspace=0.45)
    save(fig, path)


def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Radar charts of ILORA scores.")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        type=parse_dataset_spec,
        metavar="DIRNAME[:DISPLAY NAME]",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=list(PIPELINE_LABELS),
        help="Pipeline directories to plot, in legend order.",
    )
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument(
        "--scale",
        type=float,
        default=20.0,
        help="Figure size in inches; all fonts scale with it.",
    )
    parser.add_argument(
        "--no-combined", action="store_true", help="Skip the side-by-side figure."
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        path = os.path.join(args.output_dir, f"ilora_{dataset['name']}.{args.format}")
        single_chart(args.input_dir, args.pipelines, dataset, path, args.scale)

    if not args.no_combined and len(args.datasets) > 1:
        path = os.path.join(args.output_dir, f"ilora_combined.{args.format}")
        combined_chart(args.input_dir, args.pipelines, args.datasets, path, args.scale)


if __name__ == "__main__":
    main()
