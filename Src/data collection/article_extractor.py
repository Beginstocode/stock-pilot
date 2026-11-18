"""Safely download and extract readable text from public HTTPS news pages."""

from __future__ import annotations

import http.client
import ipaddress
import logging
import socket
import ssl
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

LOGGER = logging.getLogger(__name__)
REQUEST_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_BYTES = 2_000_000
MAX_REDIRECTS = 5
MIN_ARTICLE_CHARS = 200
_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "button", "svg", "noscript", "iframe"}
_SKIP_HINTS = ("advert", "cookie", "social", "share", "related", "recommend", "promo", "navigation", "subscribe", "paywall")


@dataclass(frozen=True)
class ExtractedArticle:
    url: str
    title: str | None
    author: str | None
    published_at: str | None
    text: str


def _public_https_url(value: str) -> tuple[str, int, str, str]:
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").rstrip(".").casefold()
        port = parts.port or 443
    except ValueError as exc:
        raise ValueError("Invalid article URL") from exc
    if (parts.scheme.casefold() != "https" or not host or parts.username or parts.password
            or parts.fragment or port != 443):
        raise ValueError("Article URL must be a public HTTPS URL on port 443")
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        try:
            addresses = sorted({item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
        except OSError as exc:
            raise ValueError("Article host could not be resolved") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Article URL resolved to a non-public address")
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return host, port, path, addresses[0]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to an already validated public IP while retaining TLS host checks."""

    def __init__(self, host: str, port: int, pinned_ip: str, timeout: float):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _download(url: str) -> tuple[str, bytes, str]:
    current = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        host, port, path, pinned_ip = _public_https_url(current)
        connection = _PinnedHTTPSConnection(host, port, pinned_ip, REQUEST_TIMEOUT_SECONDS)
        try:
            connection.request("GET", path, headers={"User-Agent": "StockPilotNewsReader/1.0", "Accept": "text/html"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location or redirect_count == MAX_REDIRECTS:
                    raise ValueError("Article redirect limit exceeded or Location missing")
                current = urljoin(current, location)
                continue
            if response.status < 200 or response.status >= 300:
                raise ValueError(f"Publisher returned HTTP {response.status}")
            content_type = (response.getheader("Content-Type") or "").casefold()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                raise ValueError("Publisher response is not HTML")
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("Publisher page exceeded the response size limit")
            return current, body, response.getheader("Content-Type") or "text/html"
        finally:
            connection.close()
    raise ValueError("Article redirect limit exceeded")


class _ArticleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.all_parts: list[str] = []
        self.main_parts: list[str] = []
        self.article_parts: list[str] = []
        self.skip_stack: list[str] = []
        self.main_depth = 0
        self.main_stack: list[str] = []
        self.article_depth = 0
        self.title_depth = 0
        self.title_parts: list[str] = []
        self.metadata: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        values = {key.casefold(): value or "" for key, value in attrs}
        role = values.get("role", "").casefold()
        identity = " ".join((values.get("id", ""), values.get("class", ""), values.get("aria-label", ""))).casefold()
        should_skip = tag in _SKIP_TAGS or any(hint in identity for hint in _SKIP_HINTS)
        if self.skip_stack:
            if should_skip:
                self.skip_stack.append(tag)
            return
        if should_skip:
            self.skip_stack.append(tag)
            return
        if tag == "title":
            self.title_depth += 1
        if tag in {"main", "article"} or role == "main" or any(hint in identity for hint in ("article-body", "article-content", "story-body", "post-content")):
            self.main_depth += 1
            self.main_stack.append(tag)
        if tag == "article":
            self.article_depth += 1
        meta_key = (values.get("property") or values.get("name") or values.get("itemprop", "")).casefold()
        meta_value = values.get("content", "").strip()
        if meta_key and meta_value:
            self.metadata.setdefault(meta_key, meta_value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self.skip_stack:
            if tag == self.skip_stack[-1]:
                self.skip_stack.pop()
            return
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if tag == "article" and self.article_depth:
            self.article_depth -= 1
        if self.main_stack and tag == self.main_stack[-1]:
            self.main_depth -= 1
            self.main_stack.pop()

    def handle_data(self, data: str) -> None:
        if self.skip_stack:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self.title_depth:
            self.title_parts.append(text)
            return
        self.all_parts.append(text)
        if self.main_depth:
            self.main_parts.append(text)
        if self.article_depth:
            self.article_parts.append(text)


def _unique_text(parts: list[str]) -> str:
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        key = " ".join(part.split()).casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(" ".join(part.split()))
    return "\n".join(unique)


def extract_article(url: str) -> ExtractedArticle:
    """Fetch a public HTTPS page and extract its main article text and metadata."""
    final_url, payload, _ = _download(url)
    parser = _ArticleHTMLParser()
    try:
        parser.feed(payload.decode("utf-8", errors="replace"))
        parser.close()
    except (AssertionError, ValueError) as exc:
        raise ValueError("Publisher HTML could not be parsed") from exc
    text = _unique_text(parser.article_parts or parser.main_parts or parser.all_parts)
    if len(text) < MIN_ARTICLE_CHARS:
        raise ValueError("Publisher article body is empty or too short")
    return ExtractedArticle(
        url=final_url,
        title=(parser.metadata.get("og:title") or parser.metadata.get("twitter:title") or _unique_text(parser.title_parts) or None),
        author=(parser.metadata.get("author") or parser.metadata.get("article:author") or None),
        published_at=(parser.metadata.get("article:published_time") or parser.metadata.get("datepublished") or parser.metadata.get("date") or None),
        text=text,
    )


__all__ = ["ExtractedArticle", "MIN_ARTICLE_CHARS", "extract_article"]
