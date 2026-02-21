# Behavioral Bias Detection System for LLM Financial Agents

Systematic benchmark framework for measuring cognitive bias in LLM financial recommendations.

## What It Includes

- Bias scenario generator (anchoring, recency, loss aversion, overconfidence)
- NVIDIA-first LLM evaluation runner (with optional extra providers)
- Bias detection and scoring engine
- FastAPI service for benchmark orchestration and results
- Timescale/Postgres persistence
- Dash reporting dashboard
- Point-in-time validation guardrail for scenario timestamps

## Project Layout

```text
behavioral-bias-detector/
  src/
    agents/
    api/
    config/
    core/
    dashboard/
    db/
    detectors/
    models/
    scenarios/
    utils/
  scripts/
  tests/
```

## Quick Start

1. Copy env template:

```bash
cp .env.example .env
```

Then set at least:

```bash
NVIDIA_API_KEY=...
```

2. Start infrastructure:

```bash
docker compose up -d postgres redis
```

3. Install deps locally (optional if running in containers):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Initialize DB with baseline agents/scenarios:

```bash
python scripts/init_db.py
```

5. Run API:

```bash
uvicorn src.main:app --reload --port 8000
```

6. Run dashboard:

```bash
python -m src.dashboard.app
```

## API Endpoints

- `GET /health`
- `POST /api/v1/scenarios/generate`
- `GET /api/v1/scenarios`
- `POST /api/v1/agents`
- `GET /api/v1/agents`
- `POST /api/v1/benchmark/run`
- `GET /api/v1/results/by-model`
- `GET /api/v1/runs`

## Point-in-Time Data Policy

- Every scenario includes `as_of` timestamp in `historical_context`.
- `PointInTimeController` rejects future-dated scenario context.
- Scenario generation uses deterministic timestamps anchored to generation time.

## Notes

- Minimum key needed for this setup: `NVIDIA_API_KEY`.
- `NVIDIA_BASE_URL` defaults to `https://integrate.api.nvidia.com/v1`.
- Other provider keys are optional.
- For statistically meaningful results, run at least 30 evaluations per bias type and model.
- Anchoring bias is computed pairwise across high/low anchor twins per run and agent.
