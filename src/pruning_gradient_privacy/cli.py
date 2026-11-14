"""CLI and portable artifacts; SVG requires no plotting dependency."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np

from . import __version__
from .core import run_experiments


def _svg(results: dict) -> str:
    records = results["inversion"]
    budget = max(results["config"]["budgets"])
    rhos = results["config"]["rhos"]
    series = {}
    for strategy in ("taylor", "magnitude", "random"):
        series[strategy] = [float(np.mean([r["relative_l2_error"] for r in records
                            if r["strategy"] == strategy and r["rho"] == rho and r["budget"] == budget]))
                            for rho in rhos]
    ymax = max(0.1, max(value for values in series.values() for value in values) * 1.15)
    x = lambda rho: 75 + 530 * rho
    y = lambda error: 320 - 235 * error / ymax
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="410" viewBox="0 0 700 410">',
             '<rect width="700" height="410" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#222">',
             '<text x="75" y="27" font-size="18">Synthetic gradient reconstruction</text>',
             f'<text x="75" y="50">Mean relative L2 error; {budget} Adam updates; {results["config"]["trials"]} trials</text>',
             '<path d="M75 80V320H605" fill="none" stroke="#333"/>']
    for tick in range(6):
        value = tick * ymax / 5
        parts.append(f'<text x="65" y="{y(value)+4:.1f}" text-anchor="end">{value:.2f}</text>')
    for rho in rhos:
        parts.append(f'<text x="{x(rho):.1f}" y="341" text-anchor="middle">{rho:g}</text>')
    for index, (strategy, color) in enumerate(zip(series, ("#1460aa", "#c34325", "#20844d"))):
        points = " ".join(f"{x(rho):.1f},{y(v):.1f}" for rho, v in zip(rhos, series[strategy]))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for rho, value in zip(rhos, series[strategy]):
            parts.append(f'<circle cx="{x(rho):.1f}" cy="{y(value):.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{90+170*index}" y="385" fill="{color}">{strategy}</text>')
    return "\n".join(parts + ['<text x="340" y="363" text-anchor="middle">Requested retention fraction</text>', '</g></svg>'])


def write_artifacts(results: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    results["environment"] = {"package": __version__, "python": platform.python_version(), "numpy": np.__version__}
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = results["inversion"]
    with (output / "inversion.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    theory = results["theory"]
    lines = ["# Synthetic pruning and gradient privacy lab", "",
             "All inputs are generated from the recorded seed. Budgets count Adam updates, not FLOPs.", "",
             "## Numerical checks", "",
             "| Pruning scale | Absolute score error | Remainder bound |", "| --- | --- | --- |"]
    lines += [f'| {r["scale"]:g} | {r["absolute_score_error"]:.8g} | {r["remainder_bound"]:.8g} |'
              for r in theory["quadratic_taylor"]]
    lines += ["", f'Wanda single-weight output norm maximum error: {theory["wanda_max_output_norm_error"]:.3g}.',
              f'Maximum full-gradient analytic reconstruction L2 error: {max(results["analytic_full_gradient_l2_errors"]):.3g}.', "",
              "## Reconstruction", "", "![Synthetic reconstruction error](reconstruction.svg)", "",
              "Raw results for every trial, retention fraction, strategy and budget are in inversion.csv.", "",
              "## Scope", "", "Labels are known; unquantized weight and bias gradients come from one synthetic record.",
              "The optimizer sees selected indices and values, but does not enforce top-k selection consistency.",
              "One initialization per trial is shared across all strategies and masks. Results do not establish attack optimality.",
              "At rho=0 no gradient coordinates are observed: the initialization is retained and zero updates are performed.",
              "The empty-observation matching objective is conventionally zero and does not measure reconstruction quality.",
              "Epiplexity, MDL and formal privacy guarantees are not measured."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "reconstruction.svg").write_text(_svg(results) + "\n", encoding="utf-8")


def _csv_numbers(text: str, converter):
    try:
        return [converter(part.strip()) for part in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pruning sensitivity and synthetic gradient leakage demonstrations")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run seeded synthetic numerical experiments")
    run.add_argument("--output", type=Path, default=Path("outputs/demo"))
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--dimension", type=int, default=6)
    run.add_argument("--trials", type=int, default=3)
    run.add_argument("--rhos", type=lambda s: _csv_numbers(s, float), default=[0, 0.25, 0.5, 0.75, 1.0])
    run.add_argument("--budgets", type=lambda s: _csv_numbers(s, int), default=[0, 25, 100])
    args = parser.parse_args(argv)
    try:
        results = run_experiments(args.seed, args.dimension, args.trials, args.rhos, args.budgets)
        write_artifacts(results, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {len(results['inversion'])} trial results to {args.output}")
    return 0
