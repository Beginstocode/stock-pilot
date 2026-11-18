"""Read-only article-level financial news and processed FinBERT results."""
from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from datetime import datetime, timezone
from typing import Annotated
from urllib.parse import urlsplit

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from Backend.App.config import DATA_DIR, PROCESSED_DATA_DIR
from Backend.App.database.connection import get_db
from Backend.App.schemas.news_impact import NewsImpactRequest, NewsImpactResponse
from Backend.App.services.news_impact_service import (
    NewsImpactAnalysisService,
    NewsImpactContext,
    NewsImpactProviderError,
)
from Backend.App.services.news_intelligence_service import (
    NewsIntelligencePersistenceError,
    persist_impact_analysis,
)
from Backend.App.stock_catalog import STOCK_CATALOG

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/news", tags=["News"])


class NewsArticle(BaseModel):
    symbol: str
    headline: str
    source: str | None
    published_at: str
    article_url: HttpUrl
    sentiment: str | None
    sentiment_score: float | None
    positive_probability: float | None
    negative_probability: float | None
    neutral_probability: float | None
    content_source: str = "rss_summary"
    content_quality: str = "medium"
    article_title: str | None = None
    article_author: str | None = None
    article_published_at: str | None = None


class NewsArticlesPage(BaseModel):
    total_articles: int
    current_page: int
    page_size: int
    total_pages: int
    sentiment_counts: dict[str, int]
    articles: list[NewsArticle]


@lru_cache(maxsize=1)
def get_news_impact_service() -> NewsImpactAnalysisService:
    """Return the process-wide analysis service and its bounded cache."""
    return NewsImpactAnalysisService()


def _normalized_article_url(value: str) -> str:
    parts = urlsplit(value.strip())
    return f"{parts.scheme.casefold()}://{parts.netloc.casefold()}{parts.path}?{parts.query}"


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and math.isfinite(number) else None


def _load_impact_context(article_url: str) -> NewsImpactContext:
    """Load trusted article, RSS text, association, and FinBERT data from CSVs."""
    raw = pd.read_csv(DATA_DIR / "raw" / "financial_news.csv")
    scored = pd.read_csv(PROCESSED_DATA_DIR / "news_sentiment.csv")
    raw.columns = raw.columns.str.strip()
    scored.columns = scored.columns.str.strip()
    raw = raw.rename(
        columns={
            "Date": "published_at",
            "Symbol": "symbol",
            "Headline": "headline",
            "Text": "article_text",
            "Source": "source",
            "URL": "article_id",
        }
    )
    required_raw = {"article_id", "symbol", "headline", "published_at"}
    required_scored = {"article_id", "symbol", "sentiment"}
    if not required_raw.issubset(raw.columns) or not required_scored.issubset(scored.columns):
        raise ValueError("News CSV files are missing columns required for impact analysis.")

    wanted_url = _normalized_article_url(article_url)
    raw["_normalized_url"] = raw.article_id.astype(str).map(_normalized_article_url)
    selected = raw.loc[raw["_normalized_url"].eq(wanted_url)]
    if selected.empty:
        raise LookupError("The requested article was not found in the Market News feed.")

    symbols = tuple(
        sorted(
            {
                str(value).strip().upper()
                for value in selected.symbol
                if str(value).strip().upper() in STOCK_CATALOG
            }
        )
    )
    first = selected.iloc[0]
    article_id = str(first.article_id)
    matching_scores = scored.loc[scored.article_id.astype(str).eq(article_id)]
    score = matching_scores.iloc[0] if not matching_scores.empty else {}
    headline_value = first.get("headline")
    headline = "" if pd.isna(headline_value) else str(headline_value).strip()
    text_value = first.get("article_text")
    article_text = None if text_value is None or pd.isna(text_value) else str(text_value).strip()
    content_source_value = _optional_csv_string(first.get("content_source"))
    content_source = (
        str(content_source_value).strip()
        if content_source_value in {"full_article", "rss_summary", "headline_only"}
        else ("rss_summary" if article_text else "headline_only")
    )
    content_quality_value = _optional_csv_string(first.get("content_quality"))
    content_quality = (
        str(content_quality_value).strip()
        if content_quality_value in {"high", "medium", "low"}
        else {"full_article": "high", "rss_summary": "medium", "headline_only": "low"}[content_source]
    )
    published_value = score.get("published_at", first.get("published_at", ""))
    sentiment_value = score.get("sentiment")
    source_value = first.get("source")

    return NewsImpactContext(
        article_url=article_id,
        headline=headline,
        source=None if source_value is None or pd.isna(source_value) else str(source_value),
        published_at=str(published_value),
        associated_symbols=symbols,
        sentiment=None if sentiment_value is None or pd.isna(sentiment_value) else str(sentiment_value),
        sentiment_score=_optional_float(score.get("sentiment_score")),
        positive_probability=_optional_float(score.get("positive_probability")),
        negative_probability=_optional_float(score.get("negative_probability")),
        neutral_probability=_optional_float(score.get("neutral_probability")),
        article_text=article_text or None,
        content_source=content_source,
        content_quality=content_quality,
        article_title=_optional_csv_string(first.get("article_title")),
        article_author=_optional_csv_string(first.get("author")),
        article_published_at=_optional_csv_string(first.get("article_published_at")),
    )


def _optional_csv_string(value: object) -> str | None:
    return None if value is None or pd.isna(value) or not str(value).strip() else str(value).strip()


@router.get("", response_model=NewsArticlesPage, summary="List processed financial news")
def list_news(
    symbol: Annotated[str | None, Query()] = None,
    sentiment: Annotated[str | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=50)] = 20,
) -> NewsArticlesPage:
    """Join existing raw news metadata to existing FinBERT article scores."""
    if symbol and symbol.strip().upper() not in STOCK_CATALOG:
        raise HTTPException(status_code=404, detail="Unsupported stock symbol.")
    if sentiment and sentiment.title() not in {"Positive", "Negative", "Neutral"}:
        raise HTTPException(status_code=422, detail="Invalid sentiment filter.")
    raw_path = DATA_DIR / "raw" / "financial_news.csv"
    scored_path = PROCESSED_DATA_DIR / "news_sentiment.csv"
    try:
        raw = pd.read_csv(raw_path)
        scored = pd.read_csv(scored_path)
        raw.columns = raw.columns.str.strip()
        scored.columns = scored.columns.str.strip()
        raw = raw.rename(columns={"Date": "published_at", "Symbol": "symbol", "Headline": "headline", "Source": "source", "URL": "article_id", "Text": "article_text", "ContentSource": "content_source", "ContentQuality": "content_quality", "ArticleTitle": "article_title", "Author": "author", "ArticlePublishedAt": "article_published_at"})
        scored["article_id"] = scored["article_id"].astype(str)
        raw["article_id"] = raw["article_id"].astype(str)
        source_columns = [column for column in ("article_id", "source", "content_source", "content_quality", "article_title", "author", "article_published_at") if column in raw.columns]
        sources = raw[source_columns].drop_duplicates("article_id", keep="first")
        rows = scored.merge(sources, on="article_id", how="left", validate="many_to_one")
        rows["published_at"] = pd.to_datetime(rows["published_at"], errors="coerce", utc=True)
        rows = rows.dropna(subset=["published_at", "headline", "article_id"])
        rows = rows[rows.article_id.str.startswith(("https://", "http://"))]
        if symbol:
            rows = rows[rows.symbol.astype(str).str.upper().eq(symbol.strip().upper())]
        if sentiment:
            rows = rows[rows.sentiment.astype(str).str.casefold().eq(sentiment.casefold())]
        if search:
            query = search.casefold()
            rows = rows[rows.headline.astype(str).str.casefold().str.contains(query, regex=False) | rows.symbol.astype(str).str.casefold().str.contains(query, regex=False)]
        rows = rows.sort_values(
            ["published_at", "symbol", "article_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        total_articles = len(rows)
        sentiment_counts = {
            label: int(rows.sentiment.astype(str).str.casefold().eq(label.casefold()).sum())
            for label in ("Positive", "Negative", "Neutral")
        }
        total_pages = (total_articles + page_size - 1) // page_size
        start = (page - 1) * page_size
        rows = rows.iloc[start : start + page_size]
        result = []
        for row in rows.to_dict("records"):
            result.append(NewsArticle(
                symbol=str(row["symbol"]).upper(), headline=str(row["headline"]),
                source=None if pd.isna(row.get("source")) else str(row["source"]),
                published_at=row["published_at"].isoformat(), article_url=str(row["article_id"]),
                sentiment=None if pd.isna(row.get("sentiment")) else str(row["sentiment"]),
                sentiment_score=None if pd.isna(row.get("sentiment_score")) else float(row["sentiment_score"]),
                positive_probability=None if pd.isna(row.get("positive_probability")) else float(row["positive_probability"]),
                negative_probability=None if pd.isna(row.get("negative_probability")) else float(row["negative_probability"]),
                neutral_probability=None if pd.isna(row.get("neutral_probability")) else float(row["neutral_probability"]),
                content_source=(str(row.get("content_source")) if _optional_csv_string(row.get("content_source")) in {"full_article", "rss_summary", "headline_only"} else ("rss_summary" if _optional_csv_string(row.get("article_text")) else "headline_only")),
                content_quality=(str(row.get("content_quality")) if _optional_csv_string(row.get("content_quality")) in {"high", "medium", "low"} else ("medium" if _optional_csv_string(row.get("article_text")) else "low")),
                article_title=_optional_csv_string(row.get("article_title")),
                article_author=_optional_csv_string(row.get("author")),
                article_published_at=_optional_csv_string(row.get("article_published_at")),
            ))
        return NewsArticlesPage(
            total_articles=total_articles,
            current_page=page,
            page_size=page_size,
            total_pages=total_pages,
            sentiment_counts=sentiment_counts,
            articles=result,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not read processed news")
        raise HTTPException(status_code=503, detail="News data is currently unavailable.") from exc


@router.post(
    "/impact",
    response_model=NewsImpactResponse,
    summary="Analyze a Market News article's potential business impact",
)
def analyze_news_impact(
    request: NewsImpactRequest,
    service: Annotated[NewsImpactAnalysisService, Depends(get_news_impact_service)],
    db: Annotated[Session, Depends(get_db)],
) -> NewsImpactResponse:
    """Analyze a feed article without treating FinBERT as a price forecast."""
    try:
        context = _load_impact_context(str(request.article_url))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, KeyError) as exc:
        logger.exception("Could not load article context for impact analysis")
        raise HTTPException(
            status_code=503,
            detail="News article data is currently unavailable for impact analysis.",
        ) from exc

    try:
        result = service.analyze(context)
        try:
            persist_impact_analysis(
                db,
                context=context,
                response=result,
                analysis_timestamp=datetime.now(timezone.utc),
                model_version=os.getenv("NEWS_IMPACT_LLM_MODEL", "unknown"),
            )
        except NewsIntelligencePersistenceError:
            # The impact response remains usable if the history store is temporarily down.
            logger.exception("Could not persist news impact history for %s", context.article_url)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NewsImpactProviderError as exc:
        logger.warning("News impact provider failed for %s: %s", context.article_url, exc)
        raise HTTPException(
            status_code=503,
            detail="AI impact analysis is temporarily unavailable. Check the configured LLM service.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected impact analysis error for %s", context.article_url)
        raise HTTPException(
            status_code=503,
            detail="AI impact analysis is temporarily unavailable.",
        ) from exc
