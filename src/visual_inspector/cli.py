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
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML"),
) -> None:
    """Evaluate fitted model on test data. (Stage B)"""
    typer.echo("Evaluate command — to be implemented in Stage B.")
    raise typer.Exit(code=0)


@app.command()
def benchmark(
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML"),
) -> None:
    """Run latency/throughput benchmarks. (Stage B)"""
    typer.echo("Benchmark command — to be implemented in Stage B.")
    raise typer.Exit(code=0)


@app.command()
def report() -> None:
    """Generate report from results/*.json. (Stage B)"""
    typer.echo("Report command — to be implemented in Stage B.")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
