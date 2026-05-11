import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import NHS_CONDITIONS_URL

HEADERS = {
    "User-Agent": "Mozilla/5.0 nhs-conditions-vector-demo/1.0; educational local indexing demo"
}


@dataclass
class ConditionLink:
    title: str
    url: str
    category: str


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def get_condition_links(limit: int | None = None) -> list[ConditionLink]:
    soup = BeautifulSoup(fetch_html(NHS_CONDITIONS_URL), "html.parser")
    links: list[ConditionLink] = []
    seen = set()

    current_category = "unknown"
    for node in soup.select("main h2, main h3, main a[href]"):
        if node.name in {"h2", "h3"}:
            text = node.get_text(" ", strip=True)
            if len(text) == 1 and text.isalpha():
                current_category = text.upper()
            continue

        href = node.get("href", "")
        if not href:
            continue

        url = urljoin("https://www.nhs.uk", href.split("#")[0])
        parsed = urlparse(url)
        if parsed.netloc not in {"www.nhs.uk", "nhs.uk"}:
            continue

        # Keep article-like condition pages while skipping index pages and anchors.
        path = parsed.path.rstrip("/")
        if "/conditions/" not in path:
            continue
        if path.endswith("/conditions"):
            continue

        title = node.get_text(" ", strip=True)
        if not title or title.lower().startswith("view all"):
            continue

        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        links.append(ConditionLink(title=title, url=url, category=current_category))

        if limit and len(links) >= limit:
            break

    return links


def extract_condition_page(link: ConditionLink) -> dict:
    soup = BeautifulSoup(fetch_html(link.url), "html.parser")
    title_el = soup.select_one("main h1") or soup.select_one("h1")
    title = title_el.get_text(" ", strip=True) if title_el else link.title

    chunks = []
    current_section = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_section
        text = " ".join(buffer).strip()
        if text:
            chunks.append({"section": current_section, "text": text})
        buffer = []

    for tag in soup.select("main h2, main h3, main p, main li"):
        if tag.name in {"h2", "h3"}:
            flush()
            current_section = tag.get_text(" ", strip=True)
        else:
            text = tag.get_text(" ", strip=True)
            if text and not text.lower().startswith("page last reviewed"):
                buffer.append(text)
    flush()

    return {
        "condition_id": slugify(title),
        "title": title,
        "category": link.category,
        "url": link.url,
        "chunks": chunks,
    }


def build_chunk_documents(condition: dict) -> Iterable[dict]:
    for i, chunk in enumerate(condition["chunks"]):
        chunk_text = (
            f"Condition: {condition['title']}\n"
            f"Section: {chunk['section']}\n"
            f"Text: {chunk['text']}"
        )
        yield {
            "doc_id": f"{condition['condition_id']}__{i:03d}",
            "condition_id": condition["condition_id"],
            "title": condition["title"],
            "category": condition["category"],
            "section": chunk["section"],
            "url": condition["url"],
            "chunk_text": chunk_text,
        }


def crawl_condition_documents(limit: int | None = 20, sleep_seconds: float = 0.2) -> list[dict]:
    docs: list[dict] = []
    for link in get_condition_links(limit=limit):
        condition = extract_condition_page(link)
        docs.extend(build_chunk_documents(condition))
        time.sleep(sleep_seconds)
    return docs
