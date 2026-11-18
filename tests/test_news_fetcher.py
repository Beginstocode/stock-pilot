import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Src"
    / "data collection"
    / "news_fetcher.py"
)
SPEC = importlib.util.spec_from_file_location("news_fetcher", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def rss_entry(
    title="Markets report",
    link="https://example.test/article",
    published="Fri, 30 Oct 2026 12:30:00 +0530",
    source="Financial News",
    summary="<p>Company <b>reports</b> quarterly results &amp; outlook.</p>",
):
    return SimpleNamespace(
        title=title,
        link=link,
        published=published,
        published_parsed=None,
        source=SimpleNamespace(title=source),
        summary=summary,
    )


class NewsFetcherTests(unittest.TestCase):
    def test_publication_date_is_normalized_and_invalid_dates_are_skipped(self):
        entry = rss_entry()
        record = module._entry_to_record(entry, "TCS")

        self.assertEqual(record["Date"], "2026-10-30T07:00:00Z")
        self.assertEqual(record["Symbol"], "TCS")
        self.assertEqual(record["Source"], "Financial News")
        self.assertEqual(record["Text"], "Company reports quarterly results & outlook.")
        self.assertIsNone(
            module._entry_to_record(rss_entry(published="not a date"), "TCS")
        )

    def test_missing_headline_or_link_is_skipped(self):
        self.assertIsNone(module._entry_to_record(rss_entry(title=" "), "TCS"))
        self.assertIsNone(module._entry_to_record(rss_entry(link=" "), "TCS"))

    def test_rss_summary_is_used_when_article_extraction_fails(self):
        with patch.object(module, "_cached_article", return_value=None):
            record = module._entry_to_record(rss_entry(), "TCS")
        self.assertEqual(record["ContentSource"], "rss_summary")
        self.assertEqual(record["ContentQuality"], "medium")
        self.assertEqual(record["Text"], "Company reports quarterly results & outlook.")

    def test_headline_is_used_when_summary_and_article_are_unavailable(self):
        with patch.object(module, "_cached_article", return_value=None):
            record = module._entry_to_record(rss_entry(summary=""), "TCS")
        self.assertEqual(record["ContentSource"], "headline_only")
        self.assertEqual(record["ContentQuality"], "low")
        self.assertEqual(record["Text"], "Markets report")

    def test_extracted_article_is_used_and_publisher_url_is_saved(self):
        extracted = module.ExtractedArticle(
            url="https://publisher.example/article", title="Publisher title",
            author="Reporter", published_at="2026-10-30T12:00:00Z",
            text="Full article content." * 20,
        )
        with patch.object(module, "_cached_article", return_value=extracted):
            record = module._entry_to_record(rss_entry(link="https://news.google.com/article"), "TCS")
        self.assertEqual(record["URL"], "https://publisher.example/article")
        self.assertEqual(record["ContentSource"], "full_article")
        self.assertEqual(record["ContentQuality"], "high")
        self.assertEqual(record["ArticleTitle"], "Publisher title")
        self.assertEqual(record["Author"], "Reporter")

    def test_article_extraction_cache_prevents_repeat_downloads(self):
        module._cached_article.cache_clear()
        extracted = module.ExtractedArticle(
            url="https://publisher.example/article", title=None, author=None,
            published_at=None, text="Full article content." * 20,
        )
        with patch.object(module, "extract_article", return_value=extracted) as extract:
            self.assertEqual(module._cached_article("https://publisher.example/rss"), extracted)
            self.assertEqual(module._cached_article("https://publisher.example/rss"), extracted)
        self.assertEqual(extract.call_count, 1)
        module._cached_article.cache_clear()

    def test_rss_summary_is_used_when_article_extraction_fails(self):
        with patch.object(module, "_cached_article", return_value=None):
            record = module._entry_to_record(rss_entry(), "TCS")
        self.assertEqual(record["ContentSource"], "rss_summary")
        self.assertEqual(record["ContentQuality"], "medium")
        self.assertEqual(record["Text"], "Company reports quarterly results & outlook.")

    def test_headline_is_used_when_summary_and_article_are_unavailable(self):
        with patch.object(module, "_cached_article", return_value=None):
            record = module._entry_to_record(rss_entry(summary=""), "TCS")
        self.assertEqual(record["ContentSource"], "headline_only")
        self.assertEqual(record["ContentQuality"], "low")
        self.assertEqual(record["Text"], "Markets report")

    def test_extracted_article_is_used_and_publisher_url_is_saved(self):
        extracted = module.ExtractedArticle(
            url="https://publisher.example/article", title="Publisher title",
            author="Reporter", published_at="2026-10-30T12:00:00Z",
            text="Full article content." * 20,
        )
        with patch.object(module, "_cached_article", return_value=extracted):
            record = module._entry_to_record(rss_entry(link="https://news.google.com/article"), "TCS")
        self.assertEqual(record["URL"], "https://publisher.example/article")
        self.assertEqual(record["ContentSource"], "full_article")
        self.assertEqual(record["ContentQuality"], "high")
        self.assertEqual(record["ArticleTitle"], "Publisher title")
        self.assertEqual(record["Author"], "Reporter")

    def test_separate_feeds_deduplicate_by_symbol_and_save_schema(self):
        feed = SimpleNamespace(
            status=200,
            bozo=False,
            entries=[
                rss_entry(),
                rss_entry(link="https://example.test/article#tracking"),
                rss_entry(
                    title="Another story",
                    link="https://example.test/another",
                    published="Fri, 30 Oct 2026 13:30:00 +0530",
                    source="Business Daily",
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "financial_news.csv"
            with (
                patch.object(module, "_fetch_feed", return_value=feed) as fetch,
                patch.object(module.time, "sleep") as sleep,
            ):
                summary = module.collect_news(
                    companies={"TCS": "Tata Consultancy Services", "INFY": "Infosys"},
                    limit_per_company=2,
                    delay_seconds=0.25,
                    output_path=output_path,
                )

            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(
                fetch.call_args_list,
                [
                    call(
                        module.build_feed_url("Tata Consultancy Services", 2),
                        20.0,
                    ),
                    call(module.build_feed_url("Infosys", 2), 20.0),
                ],
            )
            sleep.assert_called_once_with(0.25)
            self.assertEqual(summary.collected_by_symbol, {"TCS": 2, "INFY": 2})
            self.assertEqual(summary.total_articles, 4)
            with output_path.open(encoding="utf-8-sig", newline="") as result_file:
                reader = csv.DictReader(result_file)
                self.assertEqual(reader.fieldnames, list(module.CSV_COLUMNS))
                rows = list(reader)
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [row["Symbol"] for row in rows],
                ["TCS", "TCS", "INFY", "INFY"],
            )

    def test_feed_error_is_reported_but_does_not_stop_later_feeds(self):
        feed = SimpleNamespace(
            status=200,
            bozo=False,
            entries=[rss_entry()],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    module,
                    "_fetch_feed",
                    side_effect=[RuntimeError("network unavailable"), feed],
                ),
                patch.object(module.time, "sleep"),
            ):
                summary = module.collect_news(
                    companies={"TCS": "Tata Consultancy Services", "INFY": "Infosys"},
                    delay_seconds=0,
                    output_path=Path(temp_dir) / "news.csv",
                )

        self.assertEqual(summary.failed_symbols, ("TCS",))
        self.assertEqual(summary.collected_by_symbol, {"TCS": 0, "INFY": 1})

    def test_query_limit_and_invalid_configuration(self):
        url = module.build_feed_url("Reliance Industries", 7)
        self.assertIn("num=7", url)
        with self.assertRaises(ValueError):
            module.collect_news(limit_per_company=0)
        with self.assertRaises(ValueError):
            module.collect_news(delay_seconds=-1)


if __name__ == "__main__":
    unittest.main()
