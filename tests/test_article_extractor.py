"""Offline tests for safe publisher article extraction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "Src" / "data collection" / "article_extractor.py"
SPEC = importlib.util.spec_from_file_location("stockpilot_article_extractor_tests", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_successful_article_extraction_includes_metadata_and_cleans_duplicate_content():
    body = (
        "Quarterly results show sustained growth across several business lines. "
        "The company reported improved operating margins during the period. "
        "Management expects continued investment in technology and infrastructure. "
        "Analysts will monitor demand, costs, and execution risks over coming quarters."
    )
    html = f'''<html><head><title>Publisher story</title>
      <meta property="og:title" content="Quarterly growth report">
      <meta name="author" content="A. Reporter">
      <meta property="article:published_time" content="2026-11-14T12:00:00Z">
      </head><body><nav>Navigation links</nav><article><p>{body}</p><p>{body}</p>
      <div class="advertisement">Buy this now</div></article></body></html>'''
    with patch.object(module, "_download", return_value=("https://publisher.example/story", html.encode(), "text/html")):
        article = module.extract_article("https://news.google.com/story")

    assert article.url == "https://publisher.example/story"
    assert article.title == "Quarterly growth report"
    assert article.author == "A. Reporter"
    assert article.published_at == "2026-11-14T12:00:00Z"
    assert article.text.count("Quarterly results show") == 1
    assert "Navigation links" not in article.text
    assert "Buy this now" not in article.text


def test_inaccessible_publisher_is_reported():
    with patch.object(module, "_download", side_effect=OSError("connection refused")):
        with pytest.raises(OSError, match="connection refused"):
            module.extract_article("https://publisher.example/story")


def test_publisher_timeout_is_reported():
    with patch.object(module, "_download", side_effect=TimeoutError("request timed out")):
        with pytest.raises(TimeoutError, match="timed out"):
            module.extract_article("https://publisher.example/story")


@pytest.mark.parametrize("url", ["http://publisher.example/story", "https://127.0.0.1/story", "https://[::1]/story"])
def test_invalid_or_private_url_is_rejected(url):
    with pytest.raises(ValueError):
        module._public_https_url(url)


def test_empty_or_short_article_is_rejected():
    with patch.object(module, "_download", return_value=("https://publisher.example/story", b"<article><p>tiny text</p></article>", "text/html")):
        with pytest.raises(ValueError, match="too short"):
            module.extract_article("https://publisher.example/story")


def test_article_parser_removes_exact_duplicate_paragraphs():
    paragraph = (
        "Revenue increased as customers expanded deployments across regional operations. "
        "Operating costs remained stable while service adoption improved. "
        "Management expects to monitor demand, margins, and delivery risks."
    )
    html = f"<article><p>{paragraph}</p><p>{paragraph}</p></article>"
    with patch.object(module, "_download", return_value=("https://publisher.example/story", html.encode(), "text/html")):
        article = module.extract_article("https://publisher.example/story")
    assert article.text.count("Revenue increased") == 1
