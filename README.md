# StockPilot AI — v1.0.0

StockPilot AI is an educational research project for exploring Indian equities, technical indicators, financial-news sentiment, and historical market outcomes. It combines a React dashboard with a FastAPI service, an XGBoost BUY/HOLD/SELL classifier, FinBERT article sentiment, and a persistent news-outcome worker.

> **Educational use only. Not financial advice.** StockPilot is not an investment adviser or broker. It does not provide personalized recommendations, and its outputs may be incomplete, delayed, or wrong. Do your own research and consult a qualified professional before making investment decisions. You are responsible for your decisions and any losses.

## Problem statement

Market information is spread across price histories, technical indicators, and news sources. StockPilot provides one place to inspect those inputs and explore their historical associations with market movement. Its sentiment and business-impact labels are descriptive; they are not price forecasts and do not establish that news caused a return.

## Features

- Indian stock overview, historical prices, and technical indicators.
- XGBoost BUY/HOLD/SELL model inference using the project's saved model and feature data.
- Financial-news collection and FinBERT sentiment scoring.
- Optional AI business-impact analysis through a configured OpenAI-compatible LLM endpoint.
- Persistent article-symbol intelligence events and a scheduled worker that calculates 1D, 3D, and 5D outcomes against NIFTY 50.
- Historical performance analytics with filters, return summaries, sentiment-return alignment, and business-impact return groupings.
- Account, watchlist, portfolio, and market pages.

## Architecture

```text
React + Vite frontend (frontend/)
           │ /api proxy in development
           ▼
FastAPI backend (Backend/)
 ├─ SQLAlchemy models and database (SQLite or PostgreSQL)
 ├─ API routes and services
 ├─ XGBoost inference and FinBERT sentiment (Src/)
 └─ News outcome worker (runs with API lifespan by default)

Local data and model assets (not distributed in Git)
 ├─ data/raw, data/processed, data/cleaned
 ├─ models/*.pkl
 └─ reports/ model metadata and generated diagnostics
```

`Backend/App/database/init_db.py` creates missing ORM tables. The project currently uses SQLAlchemy `create_all`; it does not include a database migration history. The Vite development server proxies `/api` to `http://127.0.0.1:8000`.

## Technology stack

- Python, FastAPI, Pydantic, SQLAlchemy
- SQLite for local development or PostgreSQL via Psycopg
- PyTorch and Hugging Face Transformers for FinBERT
- XGBoost, NumPy, pandas, and scikit-learn for model/data work
- React, React Router, Vite, Tailwind CSS, Recharts, and Lucide
- pytest for backend tests; npm lockfile for frontend dependency resolution

The backend dependency versions are pinned in [`requirements.txt`](requirements.txt). Frontend dependencies are locked by [`frontend/package-lock.json`](frontend/package-lock.json); use the manifest in `frontend/`, not the legacy root-level npm manifest.

## Screenshots

No product screenshots are currently included. Add genuine captures under `docs/screenshots/` and link them here before publishing the GitHub release. Do not use mock or fabricated metrics.

## Requirements and local assets

- Python 3.14.4 is the version used for backend release validation in this workspace.
- Node.js 22.18.0 and npm 10.9.3 were used for frontend validation; use a Node release compatible with Vite 8.
- Git (the current workspace does not contain Git repository metadata).
- The project datasets and model weights. They are intentionally excluded from Git; see [`data/README.md`](data/README.md) for required files and layouts.
- Network access is needed the first time FinBERT weights are loaded and when running the RSS refresh pipeline. The optional LLM integration requires a provider endpoint and key.

Predictions require the saved XGBoost model, class-label metadata, and stock-feature file. Historical news outcomes require historical stock prices and NIFTY 50 prices. Missing data is reported as unavailable; the application does not fabricate prices or returns.

## Installation and setup

Run commands from the repository root unless a command changes directory.

### 1. Create the Python environment

PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch wheels and hardware support vary by platform. If the pinned PyTorch wheel is unavailable for your platform, install a compatible PyTorch build first, then install the remaining requirements.

### 2. Configure the environment

Copy `.env.example` to `.env` and replace the JWT placeholder with a locally generated random value:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Paste the generated value into `JWT_SECRET_KEY` in `.env`. On macOS/Linux, use `test -e .env || cp .env.example .env` for the copy step. Never commit `.env` or a real provider key. The sample database URL uses a local SQLite file. For PostgreSQL, set a URL such as:

```text
postgresql+psycopg://stockpilot:YOUR_PASSWORD@localhost:5432/stockpilot
```

URL-encode reserved characters in database usernames or passwords.

Optional frontend configuration can be copied from `frontend/.env.example` to `frontend/.env.local`. The default `/api` URL uses the Vite development proxy.

### 3. Provision local datasets and model assets

Place the required datasets under `data/` using the paths in [`data/README.md`](data/README.md). Place the trained model at `models/xgboost_model.pkl`; prediction metadata is read from `reports/class_label_mapping.json`. These files are not in the public source tree and must be obtained, generated, or licensed separately.

### 4. Initialize the database

With `.env` configured and the database available:

```powershell
python -m Backend.App.database.init_db
```

The command creates tables that are missing. It does not migrate an existing schema.

## Run StockPilot

Start the backend from the repository root:

```powershell
python -m uvicorn Backend.App.main:app --reload --host 127.0.0.1 --port 8000
```

Open a second terminal, then start the frontend:

```powershell
Set-Location frontend
npm.cmd ci
npm.cmd run dev -- --host 127.0.0.1
```

Open the URL printed by Vite (normally `http://127.0.0.1:5173`). The backend API documentation is at `http://127.0.0.1:8000/docs`.

For a production frontend bundle, run from `frontend/`:

```powershell
npm.cmd ci
npm.cmd run build
```

The generated `frontend/dist/` directory is a build artifact and is ignored by Git. The backend is a separate service; configure the production web server and `VITE_API_BASE_URL` to point to it.

## News pipeline and outcome worker

The RSS/FinBERT refresh pipeline is an explicit command. Run it from the repository root:

```powershell
python scripts/refresh_market_news.py
```

It fetches Google News RSS articles, scores them with FinBERT, validates both outputs, then publishes the CSV pair under `data/raw/` and `data/processed/`. It requires network access and may download the FinBERT model on first use. Pipeline logs are written to `reports/market_news_pipeline.log`.

The persistent news-outcome worker starts automatically with the backend by default. It runs once on startup, then waits `NEWS_OUTCOME_WORKER_INTERVAL_SECONDS` (default 3600 seconds) between passes. It uses the existing stock-price and NIFTY datasets and the configured database. To disable the scheduler, set `NEWS_OUTCOME_WORKER_ENABLED=false`. To run a one-off pass manually:

```powershell
python -c 'from Backend.App.workers.news_outcome_worker import NewsOutcomeWorker; print(NewsOutcomeWorker().run_once().as_dict())'
```

The worker reports counts and skips outcomes when required market prices are unavailable. Do not run the RSS refresh unless you intend to retrieve and replace the local news/sentiment CSV outputs.

## Environment variables

Backend settings are loaded from the root `.env` file or the process environment. Defaults shown below reflect the application where defined.

| Variable | Required | Purpose / default |
|---|---:|---|
| `DATABASE_URL` | Yes | SQLAlchemy URL. Example: `sqlite:///./stockpilot.db`. |
| `JWT_SECRET_KEY` | Yes | Secret used to sign authentication tokens; generate a private random value. |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No | Token lifetime; default `30`. |
| `APP_NAME` | No | API display name; default `AI Stock Recommendation System`. |
| `APP_VERSION` | No | API version; default `1.0.0`. |
| `DEBUG` | No | Enables debug mode and reload when launched through `python -m Backend.App.main`; default `false`. |
| `CORS_ORIGINS` | No | Comma-separated browser origins; defaults to localhost ports 3000. |
| `MODEL_PATH` | No | XGBoost model path; default `models/xgboost_model.pkl`. |
| `DATA_DIR` | No | Data root; default `data`. |
| `PROCESSED_DATA_DIR` | No | Processed-data root; default `data/processed`. |
| `REPORTS_DIR` | No | Reports and model metadata root; default `reports`. |
| `FINBERT_MODEL_NAME` | No | Hugging Face model ID; default `ProsusAI/finbert`. |
| `FINBERT_DEVICE` | No | Inference device such as `cpu` or `cuda`; automatic CPU/CUDA selection when blank. |
| `NEWS_IMPACT_LLM_URL` | Optional | OpenAI-compatible chat-completions endpoint for AI business-impact analysis. |
| `NEWS_IMPACT_LLM_API_KEY` | Optional | Provider credential; keep private. |
| `NEWS_IMPACT_LLM_MODEL` | Optional | Provider model name. |
| `NEWS_IMPACT_LLM_TIMEOUT_SECONDS` | No | Provider request timeout; default `25`. |
| `NEWS_OUTCOME_WORKER_ENABLED` | No | Start the outcome scheduler with the API; default `true`. |
| `NEWS_OUTCOME_WORKER_INTERVAL_SECONDS` | No | Scheduler interval; default `3600`. |
| `NEWS_OUTCOME_WORKER_LOCK_PATH` | No | Cross-process lock file; default `data/processed/.news_outcome_worker.lock`. |

Frontend-only optional variable: `VITE_API_BASE_URL` (defaults to `/api` and uses the Vite proxy during development). Set it in `frontend/.env.local`, not the backend `.env`, when pointing the built frontend at a separately hosted API.

## Testing

From the repository root with the virtual environment activated:

```powershell
python -m pytest tests -q
```

The test configuration uses an in-memory SQLite database and disables the scheduled worker. Frontend production build:

```powershell
Set-Location frontend
npm.cmd ci
npm.cmd run build
```

On macOS/Linux, use `npm` instead of `npm.cmd`.

## Known limitations

- StockPilot is an educational research project, not financial advice. Model outputs are not guaranteed and must not be treated as personalized investment guidance.
- Local datasets and model weights are not distributed in this repository. Coverage and licensing depend on the assets supplied by the operator.
- Persisted historical outcomes do not store price-basis provenance. Legacy outcomes may have adjusted/unadjusted ambiguity and cannot be reliably reclassified from the stored fields alone.
- Current stock outcomes use `Adj Close` consistently when that column exists, otherwise `Close`; NIFTY uses its `Close` index level. These are not a matched total-return basis.
- A news article mapped to multiple stocks creates multiple article-symbol observations. Those observations share an article and may be correlated; they are not independent article-level samples.
- Sentiment-return alignment and business-impact return groupings are historical associations, not price predictions. They do not establish that news caused stock movements.
- Daily close data cannot measure intraday reactions. Missing prices and insufficient historical coverage leave outcomes unavailable.
- News impact analysis requires a configured external LLM provider. Without one, that endpoint is unavailable.
- Database initialization uses `create_all`; there is no migration system for evolving an existing production database.
- Screenshots have not yet been supplied for this release.
- No `LICENSE` file has been supplied. Choose and add the intended license before public distribution; no license is implied by making the repository public.

## Release notes

The GitHub tag planned for this source release is `v1.0.0`. The current workspace does not contain `.git` metadata, so a Git status and tracked-file review cannot be produced here. No commit, push, tag, publication, or file deletion has been performed.
