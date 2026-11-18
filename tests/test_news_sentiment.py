import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Src"
    / "Sentiment"
    / "news_sentiment.py"
)
SPEC = importlib.util.spec_from_file_location("news_sentiment", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class NewsSentimentTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = module.FinancialNewsSentimentAnalyzer()

    def test_explicit_symbols_deduplicate_and_aggregate_by_stock_date(self):
        news = [
            {
                "article_id": "article-1",
                "published_at": "2026-10-30T09:30:00+05:30",
                "symbols": "tcs, INFY",
                "headline": "Company reports earnings",
            },
            {
                "article_id": "article-1",
                "published_at": "2026-10-30T09:30:00+05:30",
                "symbols": "TCS|SBIN",
                "headline": "Company reports earnings",
            },
            {
                "article_id": "article-2",
                "published_at": "2026-10-30T11:00:00+05:30",
                "symbols": ["TCS"],
                "text": "A separate article with its own ticker assignment.",
            },
            {
                "article_id": "future-article",
                "published_at": "2026-10-31T12:00:00Z",
                "symbols": "TCS",
                "headline": "Not yet published at cutoff",
            },
            {
                "article_id": "invalid-date",
                "published_at": "not-a-date",
                "symbols": "TCS",
                "headline": "Skipped due to invalid date",
            },
            {
                "article_id": "empty-content",
                "published_at": "2026-10-30T12:00:00Z",
                "symbols": "TCS",
                "headline": "  ",
                "text": "",
            },
            {
                "article_id": "unassociated",
                "published_at": "2026-10-30T12:00:00Z",
                "symbols": "",
                "headline": "Mentioning TCS does not assign the TCS ticker",
            },
        ]
        probabilities = [
            {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
            {"positive": 0.2, "negative": 0.4, "neutral": 0.4},
        ]
        with patch.object(self.analyzer, "_score_texts", return_value=probabilities) as score:
            results = self.analyzer.analyze(
                news,
                as_of="2026-10-30T23:59:59Z",
            )

        self.assertEqual(score.call_count, 1)
        self.assertEqual(len(score.call_args.args[0]), 2)
        self.assertEqual(len(results), 4)
        by_symbol = {}
        for result in results:
            by_symbol.setdefault(result.symbol, []).append(result)
        self.assertEqual(set(by_symbol), {"INFY", "SBIN", "TCS"})
        tcs = by_symbol["TCS"]
        self.assertEqual(len(tcs), 2)
        self.assertEqual(tcs[0].sentiment, "Positive")
        self.assertAlmostEqual(tcs[0].sentiment_score, 0.6)
        self.assertAlmostEqual(tcs[0].positive_probability, 0.7)
        self.assertAlmostEqual(tcs[0].daily_sentiment_score, 0.2)
        self.assertEqual(tcs[0].daily_article_count, 2)
        self.assertEqual(tcs[0].published_at, "2026-10-30T04:00:00Z")

    def test_article_text_alone_is_valid_and_missing_explicit_symbols_is_not(self):
        news = [
            {
                "published_at": "2026-10-30T12:00:00Z",
                "symbols": ["RELIANCE"],
                "text": "Full financial article text.",
            },
            {
                "published_at": "2026-10-30T12:00:00Z",
                "symbols": [],
                "headline": "Company name alone is not a stock assignment",
            },
        ]
        with patch.object(
            self.analyzer,
            "_score_texts",
            return_value=[{"positive": 0.1, "negative": 0.2, "neutral": 0.7}],
        ) as score:
            results = self.analyzer.analyze(news, as_of="2026-10-31T00:00:00Z")

        self.assertEqual(score.call_args.args[0], ["Full financial article text."])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].sentiment, "Neutral")
        self.assertAlmostEqual(results[0].sentiment_score, -0.1)

    def test_duplicate_article_content_is_scored_once(self):
        news = [
            {
                "article_id": "story-1",
                "published_at": "2026-10-30T12:00:00Z",
                "symbols": "TCS",
                "headline": "Results update",
                "text": "The same full publisher article content.",
            },
            {
                "article_id": "story-2",
                "published_at": "2026-10-30T12:00:00Z",
                "symbols": "INFY",
                "headline": "Results update",
                "text": "The same full publisher article content.",
            },
        ]
        with patch.object(
            self.analyzer,
            "_score_texts",
            return_value=[{"positive": 0.6, "negative": 0.1, "neutral": 0.3}],
        ) as score:
            results = self.analyzer.analyze(news, as_of="2026-10-31T00:00:00Z")
        self.assertEqual(score.call_args.args[0], ["Results update\nThe same full publisher article content."])
        self.assertEqual(len(results), 2)

    def test_invalid_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid as-of timestamp"):
            self.analyzer.analyze([], as_of="not-a-date")

    def test_loads_news_fetcher_csv_schema_for_direct_pipeline_use(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "financial_news.csv"
            with path.open("w", encoding="utf-8", newline="") as news_file:
                writer = csv.DictWriter(
                    news_file,
                    fieldnames=["Date", "Symbol", "Headline", "Source", "URL"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Date": "2026-10-30T12:00:00Z",
                        "Symbol": "TCS",
                        "Headline": "Quarterly company results",
                        "Source": "Financial source",
                        "URL": "https://example.test/article",
                    }
                )

            items = module.load_news_csv(path)
            self.assertEqual(items[0]["published_at"], "2026-10-30T12:00:00Z")
            self.assertEqual(items[0]["symbols"], "TCS")
            self.assertEqual(items[0]["article_id"], "https://example.test/article")
            with patch.object(
                self.analyzer,
                "_score_texts",
                return_value=[{"positive": 0.6, "negative": 0.1, "neutral": 0.3}],
            ):
                results = self.analyzer.analyze(
                    items,
                    as_of="2026-10-30T23:59:59Z",
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "TCS")
        self.assertEqual(results[0].sentiment, "Positive")


if __name__ == "__main__":
    unittest.main()
