"""Scraper for free Amharic books on allaboutethio.com.
Crawls category pages, library pages, extracts book metadata, and downloads PDFs.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

BASE_URL = "https://allaboutethio.com"

CATEGORIES = [
    {"name": "Fiction", "url": f"{BASE_URL}/amharic-books-fiction.html"},
    {"name": "Biography", "url": f"{BASE_URL}/amharic-books-biography.html"},
    {"name": "History", "url": f"{BASE_URL}/amharic-books-history.html"},
    {"name": "Children", "url": f"{BASE_URL}/amharic-books-children-books.html"},
    {"name": "Reference", "url": f"{BASE_URL}/amharic-books-reference.html"},
    {"name": "Family & Marriage", "url": f"{BASE_URL}/amharic-books-family-and-marriage.html"},
    {"name": "Islam", "url": f"{BASE_URL}/amharic-books-islam.html"},
    {"name": "Scripture", "url": f"{BASE_URL}/amharic-books-scripture.html"},
    {"name": "Teachers", "url": f"{BASE_URL}/amharic-books-teachers.html"},
    {"name": "Articles", "url": f"{BASE_URL}/amharic-books-articles.html"},
]


def fetch_url(url: str, timeout: int = 20) -> str:
    """Fetch content using curl with standard browser headers."""
    cmd = [
        "curl", "-s", "-L", "-m", str(timeout),
        "-A", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        url
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    try:
        return res.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return res.stdout.decode("latin-1", errors="replace")


def extract_library_links(category_html: str) -> List[str]:
    """Find all unique library detail page URLs in a category page."""
    matches = re.findall(r'href=[\"\'](library[a-zA-Z0-9_\-\.]+\.html)[\"\']', category_html)
    return list(dict.fromkeys(matches))


def parse_library_page(library_url: str, category_name: str) -> Optional[Dict[str, any]]:
    """Parse a single library page to extract title, parts, and PDF download URLs."""
    try:
        html = fetch_url(library_url, timeout=15)
    except Exception as e:
        print(f"Failed to fetch {library_url}: {e}")
        return None

    # Title extraction
    title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""
    title = re.sub(r"&mdash;.*$", "", title).replace("FREE Amharic Books:", "").strip()

    # Find heading or book name
    h_match = re.search(r'<h[1-3][^>]*>(.*?)</h[1-3]>', html, re.DOTALL | re.IGNORECASE)
    heading = re.sub(r"<[^>]+>", "", h_match.group(1)).strip() if h_match else ""

    # Find PDF links
    pdf_matches = re.findall(r'href=[\"\'](books/[a-zA-Z0-9_\-\.]+\.pdf)[\"\']', html, re.IGNORECASE)
    pdf_links = list(dict.fromkeys(pdf_matches))

    # Also check if direct absolute PDF links exist
    abs_pdf_matches = re.findall(r'href=[\"\'](https?://allaboutethio\.com/books/[a-zA-Z0-9_\-\.]+\.pdf)[\"\']', html, re.IGNORECASE)
    for p in abs_pdf_matches:
        rel = p.replace(f"{BASE_URL}/", "")
        if rel not in pdf_links:
            pdf_links.append(rel)

    if not pdf_links:
        return None

    # Try extracting human title from analytics trackPageview names (e.g. 'theodore1.pdf' -> 'Theodore')
    tracks = re.findall(r"_trackPageview',\s*'([^']+)'", html)
    track_title = ""
    if tracks:
        clean_track = re.sub(r"[\-_]?(?:cr|part|toc|\d+)?\.pdf$", "", tracks[0], flags=re.IGNORECASE)
        track_title = clean_track.replace("-", " ").replace("_", " ").title()

    slug = re.sub(r"\.html$", "", Path(library_url).name)
    book_title = heading if heading and "FREE Amharic" not in heading else title
    if not book_title or "allaboutethio" in book_title.lower():
        book_title = track_title or f"Amharic Book ({slug})"

    return {
        "id": f"aae_{slug}",
        "slug": slug,
        "title": book_title,
        "category": category_name,
        "library_url": library_url,
        "pdf_urls": [f"{BASE_URL}/{p}" if not p.startswith("http") else p for p in pdf_links],
        "pdf_filenames": [Path(p).name for p in pdf_links],
        "source": "allaboutethio.com",
    }


def download_file(url: str, dest_path: Path, timeout: int = 50) -> bool:
    """Download a file with curl."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True

    temp_path = dest_path.with_suffix(".tmp")
    cmd = [
        "curl", "-s", "-f", "-L", "-m", str(timeout),
        "-A", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-o", str(temp_path),
        url
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
        print(f"Error downloading {url}: {e}")
    return False


def run_allaboutethio_scraper(
    output_dir: Path,
    download_pdfs: bool = False,
    max_books: Optional[int] = None,
) -> List[Dict[str, any]]:
    """Crawl categories and book entries from allaboutethio.com."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    existing_by_url = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    existing_by_url[entry.get("library_url")] = entry
        except Exception:
            pass

    books: List[Dict[str, any]] = []
    seen_library_urls: Set[str] = set()

    print("[AllAboutEthio] Crawling categories...")
    for cat in CATEGORIES:
        print(f"  Fetching category: {cat['name']} ({cat['url']})...")
        try:
            html = fetch_url(cat["url"], timeout=15)
            lib_files = extract_library_links(html)
            print(f"    Found {len(lib_files)} library pages.")
        except Exception as e:
            print(f"    Error crawling category {cat['name']}: {e}")
            continue

        for lib_file in lib_files:
            lib_url = f"{BASE_URL}/{lib_file}"
            if lib_url in seen_library_urls:
                continue
            seen_library_urls.add(lib_url)

            if lib_url in existing_by_url:
                entry = existing_by_url[lib_url]
            else:
                entry = parse_library_page(lib_url, cat["name"])

            if entry:
                books.append(entry)
                if max_books and len(books) >= max_books:
                    break

        if max_books and len(books) >= max_books:
            break

    print(f"[AllAboutEthio] Total books/collections found: {len(books)}")

    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print(f"[AllAboutEthio] Saved manifest to {manifest_path}")

    if download_pdfs:
        print(f"[AllAboutEthio] Downloading PDFs to {pdf_dir} using parallel threads...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Collect all download tasks
        download_tasks = []
        for b in books:
            b_paths = []
            for pdf_url in b["pdf_urls"]:
                filename = Path(pdf_url).name
                dest = pdf_dir / filename
                download_tasks.append((pdf_url, dest, b))

        def _do_download(task):
            url, dest, book_entry = task
            ok = download_file(url, dest)
            return url, dest, ok, book_entry

        completed = 0
        total = len(download_tasks)
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_task = {executor.submit(_do_download, task): task for task in download_tasks}
            for future in as_completed(future_to_task):
                completed += 1
                url, dest, ok, book_entry = future.result()
                if ok:
                    if "local_pdfs" not in book_entry:
                        book_entry["local_pdfs"] = []
                    local_str = str(dest.resolve())
                    if local_str not in book_entry["local_pdfs"]:
                        book_entry["local_pdfs"].append(local_str)
                    print(f"[{completed}/{total}] Downloaded: {dest.name} ({dest.stat().st_size // 1024} KB)")
                else:
                    print(f"[{completed}/{total}] Failed: {url}")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(books, f, ensure_ascii=False, indent=2)

    return books


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape AllAboutEthio books")
    parser.add_argument("--output-dir", type=str, default="data/books_raw/allaboutethio")
    parser.add_argument("--download", action="store_true", help="Download PDF files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of books to process")
    args = parser.parse_args()

    run_allaboutethio_scraper(
        output_dir=Path(args.output_dir),
        download_pdfs=args.download,
        max_books=args.limit
    )
