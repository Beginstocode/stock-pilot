# Local data assets

Data files are intentionally excluded from Git. Obtain or generate them from sources you are authorized to use, and keep this directory layout:

| Path | Used for | Required fields / notes |
|---|---|---|
| `processed/stock_features.csv` | XGBoost inference | Must match the model's feature columns. |
| `processed/stock_price_combined.csv` | Historical stock outcomes | `Date`, `Symbol`, `Close`; `Adj Close` is preferred as a consistent dataset-wide basis when the column exists. |
| `cleaned/cleaned_nifty50.csv` | NIFTY benchmark outcomes and trading-session calendar | `Date`, `Close`. The worker can fall back to `raw/nifty50.csv` when the cleaned file is absent. |
| `raw/financial_news.csv` | Historical news analysis | `Date`, `Symbol`, `Headline`, `URL` and the source/article fields used by the news analysis. |
| `processed/news_sentiment.csv` | Historical FinBERT sentiment | `article_id`, `symbol`, `sentiment`; optional score columns are used when present. |

The market refresh and news refresh scripts may produce or update some of these files, but the project does not bundle source datasets. Availability, coverage, redistribution rights, and any third-party usage terms must be verified by the operator. Missing data is not synthesized by the analytics.
