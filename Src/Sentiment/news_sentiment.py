"""Financial-news sentiment scoring with ProsusAI FinBERT.

The module analyzes real articles supplied by a news provider or a prepared
CSV; it does not fetch or fabricate news and does not infer stock relevance
from company-name mentions.

CSV input columns:
    published_at, symbols, headline, text

``symbols`` must be an explicit, provider-supplied or human-verified
comma/semicolon/pipe-separated list of relevant tickers. At least one of
``headline`` or ``text`` is required per article.

Run from the project root:
    python "Src/Sentiment/news_sentiment.py" path\\to\\news.csv

Install dependencies into the selected environment with:
    python -m pip install pandas torch transformers
PyTorch's installation instructions can be used to select a CUDA-enabled
build when GPU inference is desired. The pretrained FinBERT model is
downloaded from Hugging Face on first use and cached by Transformers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "news_sentiment.csv"
DEFAULT_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_LABELS = ("positive", "negative", "neutral")
OUTPUT_COLUMNS = (
    "article_id",
    "published_at",
    "publication_date",
    "symbol",
    "headline",
    "sentiment",
    "positive_probability",
    "negative_probability",
    "neutral_probability",
    "sentiment_score",
    "daily_sentiment_score",
    "daily_article_count",
)
SYMBOL_SEPARATOR = re.compile(r"[,;|]")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsSentimentResult:
    """Sentiment and stock/date metadata for one article-symbol association."""

    article_id: str
    published_at: str
    publication_date: str
    symbol: str
    headline: str
    sentiment: str
    positive_probability: float
    negative_probability: float
    neutral_probability: float
    sentiment_score: float
    daily_sentiment_score: float
    daily_article_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record for API responses."""
        return asdict(self)


@dataclass(frozen=True)
class _PreparedArticle:
    article_id: str
    published_at: pd.Timestamp
    headline: str
    text: str
    symbols: tuple[str, ...]

    @property
    def model_text(self) -> str:
        if self.headline and self.text and self.headline.casefold() not in self.text.casefold():
            return f"{self.headline}\n{self.text}"
        return self.headline or self.text


def _article_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _parse_symbols(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = SYMBOL_SEPARATOR.split(value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = list(value)
    else:
        candidates = []
    symbols = {
        item.strip().upper()
        for item in candidates
        if isinstance(item, str) and item.strip()
    }
    return tuple(sorted(symbols))


def _parse_as_of(value: str | datetime | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Invalid as-of timestamp: {value!r}")
    return pd.Timestamp(parsed)


def _prepare_articles(
    items: Sequence[Mapping[str, Any]], as_of: pd.Timestamp
) -> list[_PreparedArticle]:
    """Validate, cutoff, and deduplicate provider-identified article inputs."""
    deduplicated: dict[tuple[str, ...], _PreparedArticle] = {}
    skipped_invalid_dates = 0
    skipped_empty_content = 0
    skipped_unassociated = 0
    skipped_after_cutoff = 0

    for row_number, item in enumerate(items, start=1):
        published_at = pd.to_datetime(item.get("published_at"), errors="coerce", utc=True)
        if pd.isna(published_at):
            skipped_invalid_dates += 1
            continue
        published_at = pd.Timestamp(published_at)
        if published_at > as_of:
            skipped_after_cutoff += 1
            continue

        headline = _article_text(item.get("headline"))
        text = _article_text(item.get("text"))
        if not headline and not text:
            skipped_empty_content += 1
            continue
        symbols = _parse_symbols(item.get("symbols"))
        if not symbols:
            skipped_unassociated += 1
            continue

        supplied_id = _article_text(item.get("article_id"))
        article_id = supplied_id or f"input-row-{row_number}"
        if supplied_id:
            dedupe_key = ("id", supplied_id.casefold(), published_at.isoformat())
        else:
            dedupe_key = (
                "content",
                f"{published_at.isoformat()}|{headline.casefold()}|{text.casefold()}",
            )

        existing = deduplicated.get(dedupe_key)
        if existing is None:
            deduplicated[dedupe_key] = _PreparedArticle(
                article_id=article_id,
                published_at=published_at,
                headline=headline,
                text=text,
                symbols=symbols,
            )
        else:
            deduplicated[dedupe_key] = _PreparedArticle(
                article_id=existing.article_id,
                published_at=existing.published_at,
                headline=existing.headline or headline,
                text=existing.text or text,
                symbols=tuple(sorted(set(existing.symbols).union(symbols))),
            )

    for reason, count in (
        ("invalid or missing publication dates", skipped_invalid_dates),
        ("missing headline and article text", skipped_empty_content),
        ("missing explicit stock-symbol associations", skipped_unassociated),
        ("articles published after the as-of timestamp", skipped_after_cutoff),
    ):
        if count:
            LOGGER.warning("Skipped %d input row(s) due to %s.", count, reason)

    return list(deduplicated.values())


class FinancialNewsSentimentAnalyzer:
    """Lazy-loaded FinBERT analyzer reusable across CLI and API requests."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.requested_device = device
        self.batch_size = batch_size
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._class_index: dict[str, int] | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT inference requires pandas, torch, and transformers. "
                "Install with: python -m pip install pandas torch transformers"
            ) from exc

        if self.requested_device is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.requested_device
            if device_name.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {device_name!r} was requested, but CUDA is unavailable."
                )
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.to(device_name)
            model.eval()
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Could not load {self.model_name!r}. Check Hugging Face access, "
                f"model availability, and network/cache configuration: {exc}"
            ) from exc

        raw_labels = model.config.id2label
        class_index: dict[str, int] = {}
        for raw_index, raw_label in raw_labels.items():
            label = str(raw_label).strip().casefold()
            if label in SENTIMENT_LABELS:
                class_index[label] = int(raw_index)
        if set(class_index) != set(SENTIMENT_LABELS):
            raise ValueError(
                f"Model {self.model_name!r} does not expose positive, negative, "
                f"and neutral labels; found {raw_labels!r}"
            )

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._device = torch.device(device_name)
        self._class_index = class_index
        LOGGER.info("Loaded %s on %s.", self.model_name, device_name)

    def _score_texts(self, texts: Sequence[str]) -> list[dict[str, float]]:
        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None
        assert self._class_index is not None

        results: list[dict[str, float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                encoded = self._tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with self._torch.inference_mode():
                    logits = self._model(**encoded).logits
                    batch_probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                raise RuntimeError(f"FinBERT inference failed: {exc}") from exc

            for probabilities in batch_probabilities:
                result = {
                    label: float(probabilities[self._class_index[label]])
                    for label in SENTIMENT_LABELS
                }
                if any(not 0.0 <= probability <= 1.0 for probability in result.values()):
                    raise ValueError("FinBERT returned invalid probability values")
                if abs(sum(result.values()) - 1.0) > 1e-5:
                    raise ValueError("FinBERT class probabilities do not sum to one")
                results.append(result)
        return results

    def analyze(
        self,
        items: Sequence[Mapping[str, Any]],
        as_of: str | datetime | pd.Timestamp | None = None,
    ) -> list[NewsSentimentResult]:
        """Analyze articles and return one scored record per article-symbol pair.

        ``symbols`` must be set by the source provider or verified upstream;
        this function never assigns symbols based on headline/body mentions.
        Records published later than ``as_of`` are excluded. With no cutoff,
        the current UTC time is used.
        """
        cutoff = _parse_as_of(as_of)
        articles = _prepare_articles(items, cutoff)
        if not articles:
            return []

        model_texts = [article.model_text for article in articles]
        unique_texts = list(dict.fromkeys(model_texts))
        unique_scores = self._score_texts(unique_texts)
        score_by_text = dict(zip(unique_texts, unique_scores, strict=True))
        article_scores = [score_by_text[text] for text in model_texts]
        provisional: list[dict[str, Any]] = []
        for article, probabilities in zip(articles, article_scores, strict=True):
            score = probabilities["positive"] - probabilities["negative"]
            sentiment = max(SENTIMENT_LABELS, key=probabilities.__getitem__).title()
            for symbol in article.symbols:
                provisional.append(
                    {
                        "article_id": article.article_id,
                        "published_at": article.published_at.isoformat().replace("+00:00", "Z"),
                        "publication_date": article.published_at.date().isoformat(),
                        "symbol": symbol,
                        "headline": article.headline,
                        "sentiment": sentiment,
                        "positive_probability": probabilities["positive"],
                        "negative_probability": probabilities["negative"],
                        "neutral_probability": probabilities["neutral"],
                        "sentiment_score": score,
                    }
                )

        frame = pd.DataFrame(provisional)
        aggregates = frame.groupby(
            ["symbol", "publication_date"], sort=False
        )["sentiment_score"].agg(["mean", "count"])
        results: list[NewsSentimentResult] = []
        for row in provisional:
            daily = aggregates.loc[(row["symbol"], row["publication_date"])]
            results.append(
                NewsSentimentResult(
                    **row,
                    daily_sentiment_score=float(daily["mean"]),
                    daily_article_count=int(daily["count"]),
                )
            )
        return results


def load_news_csv(path: Path) -> list[dict[str, str]]:
    """Load real news from canonical headers or the news_fetcher CSV schema."""
    if not path.is_file():
        raise FileNotFoundError(f"News input CSV was not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            headers = [column.strip() for column in (reader.fieldnames or [])]
            normalized_headers = [header.casefold() for header in headers]
            if len(set(normalized_headers)) != len(normalized_headers):
                raise ValueError("News CSV contains duplicate column names")
            columns = dict(zip(normalized_headers, headers, strict=True))
            published_column = columns.get("published_at") or columns.get("date")
            symbols_column = columns.get("symbols") or columns.get("symbol")
            headline_column = columns.get("headline")
            text_column = columns.get("text") or columns.get("article_text")
            if published_column is None or symbols_column is None:
                raise ValueError(
                    "News CSV requires published_at and symbols columns, or "
                    "Date and Symbol columns from news_fetcher.py"
                )
            if headline_column is None and text_column is None:
                raise ValueError("News CSV must include a headline and/or text column")
            article_id_column = columns.get("article_id") or columns.get("url")
            return [
                {
                    "article_id": (
                        (row.get(article_id_column, "") or "").strip()
                        if article_id_column
                        else ""
                    ),
                    "published_at": (row.get(published_column, "") or "").strip(),
                    "symbols": (row.get(symbols_column, "") or "").strip(),
                    "headline": (
                        (row.get(headline_column, "") or "").strip()
                        if headline_column
                        else ""
                    ),
                    "text": (
                        (row.get(text_column, "") or "").strip() if text_column else ""
                    ),
                }
                for row in reader
            ]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"Could not read news CSV {path}: {exc}") from exc


def save_results(results: Sequence[NewsSentimentResult], path: Path) -> None:
    """Write article and daily aggregate scores in a stable CSV schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(
            [result.to_dict() for result in results], columns=OUTPUT_COLUMNS
        ).to_csv(path, index=False)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Could not save sentiment results to {path}: {exc}") from exc


def analyze_news_csv(
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    analyzer: FinancialNewsSentimentAnalyzer | None = None,
    as_of: str | datetime | pd.Timestamp | None = None,
) -> list[NewsSentimentResult]:
    """Load, score, aggregate, and persist the supplied real news articles."""
    news_items = load_news_csv(input_path)
    sentiment_analyzer = analyzer or FinancialNewsSentimentAnalyzer()
    results = sentiment_analyzer.analyze(news_items, as_of=as_of)
    save_results(results, output_path)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV of real articles with published_at, symbols, headline and/or text",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--as-of",
        help="UTC cutoff timestamp (ISO 8601); later-published articles are excluded",
    )
    parser.add_argument("--device", help="Inference device, such as cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        analyzer = FinancialNewsSentimentAnalyzer(
            device=args.device,
            batch_size=args.batch_size,
        )
        results = analyze_news_csv(
            args.input_csv,
            output_path=args.output,
            analyzer=analyzer,
            as_of=args.as_of,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except (OSError, pd.errors.ParserError) as exc:
        LOGGER.error("Could not process news input: %s", exc)
        return 1

    print(f"Analyzed {len(results)} article-symbol record(s).")
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
