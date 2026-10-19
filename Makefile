.PHONY: setup lint test smoke eval bench report serve docker verify clean

PYTHON ?= python
UV ?= uv

setup:
	$(UV) venv .venv
	$(UV) pip install -e ".[dev]" --quiet
	@echo "✓ setup complete"

lint:
	$(UV) run ruff check src/ tests/
	$(UV) run ruff format --check src/ tests/
	$(UV) run mypy src/visual_inspector/ --ignore-missing-imports
	@echo "✓ lint passed"

test:
	$(UV) run pytest tests/ -v --tb=short
	@echo "✓ tests passed"

smoke:
	$(UV) run visual-inspector fit --config configs/smoke.yaml
	@echo "✓ smoke test passed"

eval:
	$(UV) run visual-inspector evaluate --config configs/default.yaml
	@echo "✓ evaluation complete"

bench:
	$(UV) run visual-inspector benchmark --config configs/default.yaml
	@echo "✓ benchmark complete"

report:
	$(UV) run visual-inspector report
	@echo "✓ report generated"

serve:
	$(UV) run uvicorn visual_inspector.serve.app:app --reload --port 8000

docker:
	docker build -t visual-inspector .
	@echo "✓ docker build complete"

verify: lint test
	@echo "✓ all checks passed"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info
	@echo "✓ cleaned"
