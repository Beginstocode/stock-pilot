"""API tests for filtered Market News pagination."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from Backend.App.main import app
from Backend.routes import news as news_route


def _news_frames(article_count: int = 53) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    scored_rows = []
    for index in range(article_count):
        symbol = "TCS" if index < 30 else "INFY"
        headline = (
            f"Alpha market update {index}"
            if index % 3 == 0
            else f"Market update {index}"
        )
        article_url = f"https://news.example.com/article/{index}"
        published_at = f"2026-10-30T14:00:{index:02d}Z"
        positive = index % 2 == 0
        raw_rows.append(
            {
                "Date": published_at,
                "Symbol": symbol,
                "Headline": headline,
                "Source": "Test News",
                "URL": article_url,
            }
        )
        scored_rows.append(
            {
                "article_id": article_url,
                "published_at": published_at,
                "publication_date": "2026-10-30",
                "symbol": symbol,
                "headline": headline,
                "sentiment": "Positive" if positive else "Negative",
                "positive_probability": 0.8 if positive else 0.1,
                "negative_probability": 0.1 if positive else 0.8,
                "neutral_probability": 0.1,
                "sentiment_score": 0.7 if positive else -0.7,
                "daily_sentiment_score": 0,
                "daily_article_count": 1,
            }
        )
    return pd.DataFrame(raw_rows), pd.DataFrame(scored_rows)


def _mock_news_csvs(monkeypatch, article_count: int = 53) -> None:
    raw, scored = _news_frames(article_count)

    def read_csv(path, *args, **kwargs):
        return raw.copy() if str(path).endswith("financial_news.csv") else scored.copy()

    monkeypatch.setattr(news_route.pd, "read_csv", read_csv)


def _article_keys(response) -> list[tuple[str, str]]:
    return [
        (article["symbol"], article["article_url"])
        for article in response.json()["articles"]
    ]


def test_news_first_middle_last_pages_have_no_gaps_or_duplicates(monkeypatch):
    _mock_news_csvs(monkeypatch)
    with TestClient(app) as client:
        pages = [
            client.get("/news", params={"page": page, "page_size": 20})
            for page in (1, 2, 3)
        ]

    assert all(response.status_code == 200 for response in pages)
    payloads = [response.json() for response in pages]
    assert [payload["total_articles"] for payload in payloads] == [53, 53, 53]
    assert [payload["current_page"] for payload in payloads] == [1, 2, 3]
    assert [payload["total_pages"] for payload in payloads] == [3, 3, 3]
    assert [len(payload["articles"]) for payload in payloads] == [20, 20, 13]

    page_keys = [_article_keys(response) for response in pages]
    assert set(page_keys[0]).isdisjoint(page_keys[1])
    assert set(page_keys[0]).isdisjoint(page_keys[2])
    assert set(page_keys[1]).isdisjoint(page_keys[2])
    assert len(set(page_keys[0] + page_keys[1] + page_keys[2])) == 53


def test_news_pagination_applies_stock_sentiment_and_search_filters_first(monkeypatch):
    _mock_news_csvs(monkeypatch)
    with TestClient(app) as client:
        stock_page = client.get(
            "/news", params={"symbol": "TCS", "page": 2, "page_size": 10}
        ).json()
        sentiment_page = client.get(
            "/news",
            params={"sentiment": "Positive", "page": 2, "page_size": 20},
        ).json()
        search_page = client.get(
            "/news", params={"search": "Alpha", "page": 2, "page_size": 5}
        ).json()

    assert stock_page["total_articles"] == 30
    assert stock_page["current_page"] == 2
    assert len(stock_page["articles"]) == 10
    assert {article["symbol"] for article in stock_page["articles"]} == {"TCS"}

    assert sentiment_page["total_articles"] == 27
    assert len(sentiment_page["articles"]) == 7
    assert {article["sentiment"] for article in sentiment_page["articles"]} == {
        "Positive"
    }

    assert search_page["total_articles"] == 18
    assert len(search_page["articles"]) == 5
    assert all(
        "alpha" in article["headline"].casefold()
        for article in search_page["articles"]
    )


def test_news_pagination_validates_page_size():
    with TestClient(app) as client:
        invalid_page = client.get("/news", params={"page": 0})
        invalid_page_size = client.get("/news", params={"page_size": 51})
    assert invalid_page.status_code == 422
    assert invalid_page_size.status_code == 422
