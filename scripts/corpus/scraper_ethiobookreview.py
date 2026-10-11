"""Scraper for free books on ethiobookreview.com.
Fetches book metadata and downloads PDF files.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

BASE_URL = "https://www.ethiobookreview.com"
FREE_BOOKS_URL = f"{BASE_URL}/free_books"
ASSET_PDF_PREFIX = f"{BASE_URL}/assets/free_books"


def fetch_page(url: str, timeout: int = 20) -> str:
    """Fetch web page using curl."""
    cmd = [
        "curl", "-s", "-L", "-m", str(timeout),
        "-A", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        url
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return res.stdout.decode("utf-8", errors="replace")


def parse_free_books_page(html_content: str) -> List[Dict[str, any]]:
    """Parse book cards from ethiobookreview.com/free_books HTML."""
    # Pattern to match book cards
    # Structure contains: image, title link, page count, and download link
    book_cards = re.findall(
        r'<div class="col-md-12 readers">(.*?)</div>\s*<div class="col-md-3',
        html_content,
        re.DOTALL
    )

    books = []
    seen_slugs = set()

    for card in book_cards:
        # Title and slug
        title_match = re.search(
            r'<h2[^>]*>\s*<a\s+href=[\"\']https://www\.ethiobookreview\.com/free_books/([^\"\']+)[\"\'][^>]*>(.*?)</a>',
            card,
            re.DOTALL
        )
        if not title_match:
            continue

        slug = title_match.group(1).strip()
        raw_title = title_match.group(2).strip()
        clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()

        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        # Cover image
        img_match = re.search(r'<img[^>]+src=[\"\']([^\"\']+)[\"\']', card)
        img_url = img_match.group(1) if img_match else None

        # Pages & Downloads info
        pages_match = re.search(r'(\d+)\s+Pages', card, re.IGNORECASE)
        pages = int(pages_match.group(1)) if pages_match else None

        # Author extraction from title (often "Title: Author" or "Title: Author Translated By ...")
        author = None
        if ":" in clean_title:
            parts = clean_title.split(":", 1)
            book_name = parts[0].strip()
            author = parts[1].strip()
        else:
            book_name = clean_title

        pdf_url = f"{ASSET_PDF_PREFIX}/{slug}.pdf"

        books.append({
            "id": f"ebr_{slug}",
            "slug": slug,
            "title": clean_title,
            "book_name": book_name,
            "author": author,
            "pages": pages,
            "source_url": f"{FREE_BOOKS_URL}/{slug}",
            "pdf_url": pdf_url,
            "cover_image_url": img_url,
            "source": "ethiobookreview.com",
        })

    return books


def download_book(pdf_url: str, dest_path: Path, timeout: int = 25) -> bool:
    """Download PDF file if not already present."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True

    temp_path = dest_path.with_suffix(".tmp")
    cmd = [
        "curl", "-s", "-L", "-m", str(timeout),
        "-A", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-o", str(temp_path),
        pdf_url
    ]
    try:
        subprocess.run(cmd, check=True)
        if temp_path.exists() and temp_path.stat().st_size > 1000:
            temp_path.rename(dest_path)
            return True
        elif temp_path.exists():
            temp_path.unlink()
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"Error downloading {pdf_url}: {e}")
    return False


def run_ethiobookreview_scraper(
    output_dir: Path,
    download_pdfs: bool = True,
    max_books: Optional[int] = None,
) -> List[Dict[str, any]]:
    """Crawl metadata and download books."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EthioBookReview] Fetching books index from {FREE_BOOKS_URL}...")
    try:
        html = fetch_page(FREE_BOOKS_URL)
    except Exception as e:
        print(f"[EthioBookReview] Live fetch failed: {e}. Checking cached HTML...")
        cache_file = Path("/home/codeknight/.gemini/antigravity/brain/e75ede59-a25e-4eb0-9f7a-81fef1258c77/.system_generated/steps/4/content.md")
        if cache_file.exists():
            html = cache_file.read_text(encoding="utf-8")
        else:
            raise

    books = parse_free_books_page(html)
    print(f"[EthioBookReview] Found {len(books)} books.")

    if max_books:
        books = books[:max_books]

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print(f"[EthioBookReview] Saved manifest to {manifest_path}")

    if download_pdfs:
        print(f"[EthioBookReview] Downloading PDFs to {pdf_dir}...")
        for i, b in enumerate(books, 1):
            dest = pdf_dir / f"{b['slug']}.pdf"
            print(f"[{i}/{len(books)}] Downloading {b['slug']} ({b.get('pages', '?')} pages)...", end="", flush=True)
            success = download_book(b["pdf_url"], dest)
            if success:
                b["local_pdf"] = str(dest.resolve())
                print(f" done ({dest.stat().st_size // 1024} KB)")
            else:
                b["local_pdf"] = None
                print(" failed (unavailable or timeout)")

        # Update manifest with local paths
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(books, f, ensure_ascii=False, indent=2)

    return books


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape EthioBookReview books")
    parser.add_argument("--output-dir", type=str, default="data/books_raw/ethiobookreview")
    parser.add_argument("--no-download", action="store_true", help="Only build manifest without downloading PDFs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of books to download")
    args = parser.parse_args()

    run_ethiobookreview_scraper(
        output_dir=Path(args.output_dir),
        download_pdfs=not args.no_download,
        max_books=args.limit
    )
