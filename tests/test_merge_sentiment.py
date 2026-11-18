import json
import importlib.util
import tempfile
import unittest
from datetime import time
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Src"
    / "data processing"
    / "merge_sentiment.py"
)
SPEC = importlib.util.spec_from_file_location("merge_sentiment", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class MergeSentimentTests(unittest.TestCase):
    def test_merge_uses_cutoff_rolls_by_trading_rows_and_preserves_no_news_as_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_path = root / "stock_features.csv"
            sentiment_path = root / "news_sentiment.csv"
            output_path = root / "merged_stock_features.csv"
            pd.DataFrame(
                [
                    {"Date": "2026-10-29", "Symbol": " tcs ", "Close": 100.0, "Target": "BUY"},
                    {"Date": "2026-10-30", "Symbol": "TCS", "Close": 101.0, "Target": "HOLD"},
                    {"Date": "2026-11-02", "Symbol": "TCS", "Close": 102.0, "Target": "SELL"},
                    {"Date": "2026-10-29", "Symbol": "INFY", "Close": 200.0, "Target": "HOLD"},
                ]
            ).to_csv(features_path, index=False)
            pd.DataFrame(
                [
                    {
                        "article_id": "early",
                        "published_at": "2026-10-29T03:40:00Z",
                        "symbol": "tcs",
                        "sentiment": "Positive",
                        "sentiment_score": 0.8,
                        "positive_probability": 0.9,
                        "negative_probability": 0.05,
                        "neutral_probability": 0.05,
                    },
                    {
                        "article_id": "after-cutoff",
                        "published_at": "2026-10-29T03:46:00Z",
                        "symbol": "TCS",
                        "sentiment": "Negative",
                        "sentiment_score": -0.6,
                        "positive_probability": 0.1,
                        "negative_probability": 0.7,
                        "neutral_probability": 0.2,
                    },
                    {
                        "article_id": "duplicate",
                        "published_at": "2026-10-30T02:30:00Z",
                        "symbol": "TCS",
                        "sentiment": "Neutral",
                        "sentiment_score": 0.0,
                        "positive_probability": 0.2,
                        "negative_probability": 0.2,
                        "neutral_probability": 0.6,
                    },
                    {
                        "article_id": "duplicate",
                        "published_at": "2026-10-30T02:30:00Z",
                        "symbol": "tcs",
                        "sentiment": "Neutral",
                        "sentiment_score": 0.0,
                        "positive_probability": 0.2,
                        "negative_probability": 0.2,
                        "neutral_probability": 0.6,
                    },
                    {
                        "article_id": "after-day2-cutoff",
                        "published_at": "2026-10-30T03:46:00Z",
                        "symbol": "TCS",
                        "sentiment": "Negative",
                        "sentiment_score": -0.4,
                        "positive_probability": 0.2,
                        "negative_probability": 0.6,
                        "neutral_probability": 0.2,
                    },
                    {
                        "article_id": "bad-date",
                        "published_at": "2026-10-30 12:00:00",
                        "symbol": "TCS",
                        "sentiment": "Positive",
                        "sentiment_score": 0.5,
                        "positive_probability": 0.7,
                        "negative_probability": 0.1,
                        "neutral_probability": 0.2,
                    },
                    {
                        "article_id": "before-feature-history",
                        "published_at": "2026-10-28T15:00:00Z",
                        "symbol": "TCS",
                        "sentiment": "Positive",
                        "sentiment_score": 0.7,
                        "positive_probability": 0.8,
                        "negative_probability": 0.1,
                        "neutral_probability": 0.1,
                    },
                ]
            ).to_csv(sentiment_path, index=False)

            merged, stats = module.merge_sentiment(
                features_path,
                sentiment_path,
                output_path,
                targets_path=None,
                cutoff_time=time(9, 15),
            )
            self.assertTrue(output_path.is_file())

        self.assertEqual(len(merged), 4)
        self.assertEqual(merged["Close"].tolist(), [100.0, 101.0, 102.0, 200.0])
        self.assertEqual(merged["Target"].tolist(), ["BUY", "HOLD", "SELL", "HOLD"])
        self.assertEqual(merged["Symbol"].tolist(), ["TCS", "TCS", "TCS", "INFY"])
        tcs = merged.loc[merged["Symbol"].eq("TCS")].reset_index(drop=True)
        self.assertAlmostEqual(tcs.loc[0, "Average_Sentiment_Score"], 0.8)
        self.assertEqual(tcs.loc[0, "Total_News_Count"], 1)
        self.assertAlmostEqual(tcs.loc[1, "Average_Sentiment_Score"], -0.3)
        self.assertEqual(tcs.loc[1, "Neutral_News_Count"], 1)
        self.assertEqual(tcs.loc[1, "Total_News_Count"], 2)
        self.assertAlmostEqual(tcs.loc[1, "Sentiment_Score_5D"], 0.25)
        self.assertAlmostEqual(tcs.loc[2, "Average_Sentiment_Score"], -0.4)
        self.assertEqual(tcs.loc[2, "Total_News_Count"], 1)
        self.assertAlmostEqual(tcs.loc[2, "Sentiment_Score_5D"], 0.0333333333)
        infy = merged.loc[merged["Symbol"].eq("INFY")].iloc[0]
        self.assertEqual(infy["Total_News_Count"], 0)
        self.assertTrue(pd.isna(infy["Average_Sentiment_Score"]))
        self.assertTrue(pd.isna(infy["Sentiment_Score_5D"]))
        self.assertEqual(stats["duplicate_rows_removed"], 1)
        self.assertEqual(stats["invalid_or_timezone_naive_timestamps"], 1)
        self.assertEqual(stats["pre_history_articles_excluded"], 1)
        self.assertEqual(stats["news_timestamps_after_cutoff"], 0)
        self.assertEqual(
            merged.loc[merged["Symbol"].eq("TCS"), "News_Publication_Timestamps_UTC"]
            .map(json.loads)
            .tolist(),
            [
                ["2026-10-29T03:40:00Z"],
                ["2026-10-29T03:46:00Z", "2026-10-30T02:30:00Z"],
                ["2026-10-30T03:46:00Z"],
            ],
        )

    def test_existing_output_is_not_overwritten_without_explicit_permission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_path = root / "features.csv"
            sentiment_path = root / "sentiment.csv"
            output_path = root / "merged.csv"
            output_path.write_text("keep me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module.merge_sentiment(
                    features_path,
                    sentiment_path,
                    output_path,
                    targets_path=None,
                )
            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep me")

    def test_training_targets_are_attached_without_dropping_feature_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_path = root / "features.csv"
            sentiment_path = root / "sentiment.csv"
            targets_path = root / "training.csv"
            output_path = root / "merged.csv"
            pd.DataFrame(
                [{"Date": "2026-10-29", "Symbol": "TCS", "RSI_14": 54.0}]
            ).to_csv(features_path, index=False)
            pd.DataFrame(
                [
                    {
                        "article_id": "invalid",
                        "published_at": "bad-date",
                        "symbol": "TCS",
                        "sentiment": "Neutral",
                        "sentiment_score": 0.0,
                        "positive_probability": 0.1,
                        "negative_probability": 0.1,
                        "neutral_probability": 0.8,
                    }
                ]
            ).to_csv(sentiment_path, index=False)
            pd.DataFrame(
                [
                    {
                        "Date": "2026-10-29",
                        "Symbol": "TCS",
                        "Future_Return_5D": 0.04,
                        "Target": "BUY",
                    }
                ]
            ).to_csv(targets_path, index=False)

            merged, stats = module.merge_sentiment(
                features_path,
                sentiment_path,
                output_path,
                targets_path=targets_path,
            )

        self.assertEqual(merged.loc[0, "RSI_14"], 54.0)
        self.assertEqual(merged.loc[0, "Future_Return_5D"], 0.04)
        self.assertEqual(merged.loc[0, "Target"], "BUY")
        self.assertEqual(
            set(stats["target_columns_preserved"]), {"Target", "Future_Return_5D"}
        )


if __name__ == "__main__":
    unittest.main()
