# Contributing

## Dev setup

```bash
uv venv
uv pip install -e ".[dev,openai,mcp]"
pre-commit install
```

## Checks before opening a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

## Adding a new specialty to the data pipeline

1. Add `data_pipeline/ingest/<specialty>/<society>.py` implementing `Ingester`.
2. Register the society + native grading vocabulary in `data_pipeline/registry.py`.
3. Provide a normalization function mapping the society's grades onto `evidence_strength_score ∈ [0,1]`.
4. Drop one example guideline document into `tests/fixtures/` and write an ingest test.

## Adding a new RoB tool

1. Define an `XAssessment` pydantic model alongside `RoB2Assessment` / `RobinsIAssessment`.
2. Implement the tool in `src/weighted_evidence/rubric/<x>.py`.
3. Wire `aggregate.py` to dispatch on `StudyDesign`.
4. Add a fixture and an integration test.
