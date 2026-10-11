"""Scraper for free Amharic books on ethiopianorthodox.org.
Crawls the open directory at https://www.ethiopianorthodox.org/amharic/holybooks/,
catalogs books, and downloads PDFs.
"""

import argparse
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List

BASE_URL = "https://www.ethiopianorthodox.org/amharic/holybooks/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def fetch_directory_listing() -> str:
    """Fetch the HTML index from the holybooks directory."""
    req = urllib.request.Request(BASE_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=25) as resp:
        return resp.read().decode("latin-1", errors="replace")


def parse_pdf_links(html: str) -> List[Dict]:
    """Parse table rows to extract PDF filenames, dates, and sizes."""
    # Find all href="*.pdf"
    pattern = r'href=[\"\']([^\"\']+\.pdf)[\"\']'
    matches = re.findall(pattern, html, re.IGNORECASE)

    unique_files = list(dict.fromkeys(matches))
    books = []

    for rel_path in unique_files:
        decoded_name = urllib.parse.unquote(rel_path).strip()
        slug = re.sub(r"\.pdf$", "", decoded_name, flags=re.IGNORECASE).strip().replace(" ", "-").lower()
        title = re.sub(r"[\-_]", " ", Path(decoded_name).stem).title()
        encoded_rel = urllib.parse.quote(decoded_name)
        pdf_url = urllib.parse.urljoin(BASE_URL, encoded_rel)

        books.append({
            "id": f"eotc_{slug}",
            "slug": slug,
            "title": title,
            "filename": Path(decoded_name).name,
            "pdf_url": pdf_url,
            "source": "ethiopianorthodox.org",
            "source_url": BASE_URL,
        })

    return books


def download_curl(url: str, dest_path: Path, timeout: int = 60) -> bool:
    """Download a file using curl."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True

    temp_path = dest_path.with_suffix(".tmp")
    cmd = [
        "curl", "-s", "-f", "-L", "-m", str(timeout),
        "-A", HEADERS["User-Agent"],
        "-o", str(temp_path),
        url
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if temp_path.exists() and temp_path.stat().st_size > 1000:
            with open(temp_path, "rb") as f:
                if not f.read(10).startswith(b"%PDF"):
                    temp_path.unlink(missing_ok=True)
                    return False
            temp_path.rename(dest_path)
            return True
        else:
            if temp_path.exists():
                temp_path.unlink()
            return False
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        return False


def run_eotc_scraper(
    dest_dir: Path,
    max_books: int = 50,
    download_pdfs: bool = True,
    max_size_mb: int = 40
) -> List[Dict]:
    """Catalog and download books from ethiopianorthodox.org."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir = dest_dir / "pdfs"
    pdfs_dir.mkdir(exist_ok=True)

    manifest_file = dest_dir / "manifest.json"

    print("[EOTC] Fetching directory index...")
    html = fetch_directory_listing()
    all_books = parse_pdf_links(html)
    print(f"[EOTC] Discovered {len(all_books)} book PDFs in archive.")

    # Load existing manifest if present to preserve downloaded books
    existing_by_slug = {}
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                for b in json.load(f):
                    existing_by_slug[b["slug"]] = b
        except Exception:
            pass

    cataloged_books = []
    downloaded_count = 0

    for book in all_books:
        if downloaded_count >= max_books:
            break

        filename = book["filename"]
        target_path = pdfs_dir / filename

        # If already downloaded
        if target_path.exists() and target_path.stat().st_size > 1000:
            book["local_pdf"] = str(target_path.resolve())
            cataloged_books.append(book)
            downloaded_count += 1
            continue

        print(f"\n[EOTC] Processing '{book['title']}' ({filename})...")

        if download_pdfs:
            # Check size via curl -I first
            cmd_size = ["curl", "-s", "-I", "-m", "10", book["pdf_url"]]
            try:
                res = subprocess.run(cmd_size, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                hdr = res.stdout.decode("latin-1", errors="ignore")
                match = re.search(r"Content-Length:\s*(\d+)", hdr, re.IGNORECASE)
                if match:
                    size_mb = int(match.group(1)) / (1024 * 1024)
                    if size_mb > max_size_mb:
                        print(f"  Skipping (size {size_mb:.1f} MB exceeds limit {max_size_mb} MB)")
                        continue
            except Exception:
                pass

            if download_curl(book["pdf_url"], target_path, timeout=60):
                book["local_pdf"] = str(target_path.resolve())
                downloaded_count += 1
                cataloged_books.append(book)
                print(f"  -> Downloaded ({target_path.stat().st_size:,} bytes)")
            else:
                print("  -> Download failed or timed out.")
        else:
            cataloged_books.append(book)
            downloaded_count += 1

        time.sleep(0.3)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(cataloged_books, f, ensure_ascii=False, indent=2)

    print(f"\n[EOTC] Finished! {len(cataloged_books)} books saved to {manifest_file}.")
    return cataloged_books


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Ethiopian Orthodox Holy Books Archive")
    parser.add_argument("--dest-dir", type=str, default="data/books_raw/eotc")
    parser.add_argument("--limit", type=int, default=30, help="Max number of books to download")
    parser.add_argument("--no-download", action="store_true", help="Only catalog manifest without downloading")
    parser.add_argument("--max-mb", type=int, default=35, help="Max file size in MB")
    args = parser.parse_args()

    run_eotc_scraper(
        dest_dir=Path(args.dest_dir),
        max_books=args.limit,
        download_pdfs=not args.no_download,
        max_size_mb=args.max_mb
    )
