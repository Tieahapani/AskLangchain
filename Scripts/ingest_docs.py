"""
One-time script: Scrape LangChain docs, chunk, embed, and save FAISS index.

Usage:
    python3 Scripts/ingest_docs.py
"""

import os
import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document


load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
SEED_URLS = [
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langchain/quickstart",
    "https://docs.langchain.com/oss/python/langchain/agents",
    "https://docs.langchain.com/oss/python/langchain/models",
    "https://docs.langchain.com/oss/python/langchain/tools",
    "https://docs.langchain.com/oss/python/langchain/retrieval",
    "https://docs.langchain.com/oss/python/integrations/vectorstores",
    "https://docs.langchain.com/oss/python/integrations/text_embedding",
    "https://docs.langchain.com/oss/python/integrations/document_loaders",
    "https://docs.langchain.com/oss/python/integrations/retrievers",
    "https://docs.langchain.com/oss/python/deepagents/overview",
]
MAX_PAGES = 100
RATE_LIMIT_DELAY = 1  # seconds between requests


def discover_doc_urls(seed_urls: list[str], max_pages: int = MAX_PAGES) -> list[str]:
    """
    Start from seed pages and crawl to discover linked doc pages.
    Only follows links within docs.langchain.com/oss/python/
    """
    visited = set()
    to_visit = list(seed_urls)
    doc_urls = []

    print(f"🔍 Discovering doc pages from {len(seed_urls)} seed URLs...")

    while to_visit and len(doc_urls) < max_pages:
        url = to_visit.pop(0)

        # Normalize: remove fragment (#section) and trailing slash
        url = url.split("#")[0].rstrip("/")

        if url in visited:
            continue
        visited.add(url)

        # Only process LangChain Python doc pages
        if not url.startswith("https://docs.langchain.com/oss/python"):
            continue

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ✗ Failed: {url} — {e}")
            continue

        doc_urls.append(url)
        print(f"  ✓ [{len(doc_urls)}/{max_pages}] {url}")

        # Find more doc links on this page
        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(url, href).split("#")[0].rstrip("/")

            if (
                full_url.startswith("https://docs.langchain.com/oss/python")
                and full_url not in visited
                and "/api_reference" not in full_url
            ):
                to_visit.append(full_url)

        time.sleep(RATE_LIMIT_DELAY)

    print(f"\n✅ Discovered {len(doc_urls)} doc pages")
    return doc_urls


def scrape_page(url: str) -> dict | None:
    """Scrape a single doc page and return its text + metadata."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ Failed: {url} — {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try multiple selectors for main content
    content = (
        soup.find("div", id="content")
        or soup.find("div", id="content-area")
        or soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="markdown")
        or soup.find("div", {"role": "main"})
    )

    if not content:
        print(f"  ✗ No content found: {url}")
        return None

    # Remove nav, sidebar, footer, scripts that might be nested inside
    for tag in content.find_all(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()

    # Get clean text
    text = content.get_text(separator="\n", strip=True)

    # Skip pages with very little content
    if len(text) < 100:
        print(f"  ✗ Too short ({len(text)} chars): {url}")
        return None

    # Extract page title
    title = soup.find("title")
    title_text = title.get_text(strip=True) if title else url.split("/")[-1]

    return {
        "text": text,
        "source": url,
        "title": title_text,
    }


def scrape_all_pages(urls: list[str]) -> list[dict]:
    """Scrape content from all discovered URLs."""
    print(f"\n📄 Scraping content from {len(urls)} pages...")
    pages = []

    for i, url in enumerate(urls, 1):
        page = scrape_page(url)
        if page:
            pages.append(page)
            print(f"  [{i}/{len(urls)}] ✓ {page['title'][:60]}")
        else:
            print(f"  [{i}/{len(urls)}] ✗ skipped")

    print(f"\n✅ Successfully scraped {len(pages)}/{len(urls)} pages")
    return pages

def chunk_pages(pages: list[dict]) -> list[Document]:
    """Convert scraped pages into chunked Documents with metadata."""
    docs = [
        Document(
            page_content=page["text"],
            metadata={"source": page["source"], "title": page["title"]},
        )
        for page in pages
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print(f"\n✂️ Split {len(docs)} pages into {len(chunks)} chunks")
    return chunks

def embed_and_save(chunks: list[Document], save_path: str = "vectorstore"): 
    """Embed chunks with Gemini and save FAISS index to disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"\n💾 Saved FAISS index with {len(chunks)} chunks to {save_path}/")



# ── Test it ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Discover URLs (small test with 5 pages)
    urls = discover_doc_urls(SEED_URLS)

    # Step 2: Scrape content
    pages = scrape_all_pages(urls)
    chunks = chunk_pages(pages)
    embed_and_save(chunks)
