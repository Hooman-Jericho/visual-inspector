"""CLI entry point using Typer."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="visual-inspector",
    help="Unsupervised industrial defect detection with PatchCore.",
    add_completion=False,
)


@app.command()
def data_check(
    root: str = typer.Option("data/mvtec_ad", help="Path to MVTec AD dataset root"),
) -> None:
    """Validate MVTec AD data directory and print counts per category."""
    from visual_inspector.data.mvtec import check_mvtec_data

    check_mvtec_data(Path(root))


@app.command()
def fit(
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML"),
) -> None:
    """Fit PatchCore model on training data."""
    from visual_inspector.config import AppConfig
    from visual_inspector.models.patchcore import PatchCoreModel

    cfg = AppConfig.from_yaml(config)
    model = PatchCoreModel(cfg)
    model.fit()


@app.command()
def evaluate(
    config: str = typer.Option("configs/smoke.yaml", help="Path to config YAML"),
    profile: str = typer.Option("smoke", help="Profile to evaluate: smoke, default, full"),
    category: str | None = typer.Option(None, help="Category name (optional)"),
    max_train: int | None = typer.Option(None, help="Max train images"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Force synthetic dataset"),
) -> None:
    """Evaluate PatchCore on test data."""
    from visual_inspector.eval.evaluate import run_evaluation

    results = run_evaluation(
        config_path=config,
        profile=profile,
        category=category,
        max_train=max_train,
        use_synthetic=synthetic,
    )
    for res in results:
        m = res["metrics"]
        p_str = f"{m['pixel_auroc']:.4f}" if m["pixel_auroc"] is not None else "N/A"
        pro_str = f"{m['au_pro']:.4f}" if m["au_pro"] is not None else "N/A"
        typer.echo(
            f"[{res['category']:12s}] Image AUROC: {m['image_auroc']:.4f} | "
            f"Pixel AUROC: {p_str} | AU-PRO: {pro_str}"
        )


@app.command()
def benchmark(
    config: str = typer.Option("configs/smoke.yaml", help="Path to config YAML"),
    output: str = typer.Option("results/latency.json", help="Output path for latency JSON"),
    warmup: int = typer.Option(10, help="Warmup iterations"),
    runs: int = typer.Option(50, help="Benchmark runs"),
) -> None:
    """Run latency benchmarks across coreset ratios."""
    from visual_inspector.eval.latency import run_latency_benchmark

    typer.echo("Running latency benchmarks...")
    res = run_latency_benchmark(
        config_path=config,
        output_path=output,
        n_warmup=warmup,
        n_runs=runs,
    )
    for ratio_key, data in res["ratios"].items():
        tot = data["total_ms"]
        typer.echo(
            f"[{ratio_key}] Median: {tot['median']:.1f}ms | P95: {tot['p95']:.1f}ms | Memory: {data['memory_bank_mb']}MB"
        )


@app.command()
def report() -> None:
    """Generate report from results/*.json. (Stage B)"""
    typer.echo("Report command — to be implemented in Stage B.")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
