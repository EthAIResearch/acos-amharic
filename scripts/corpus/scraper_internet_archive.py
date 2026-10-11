"""Scraper for Amharic books and texts on Internet Archive (archive.org).
Queries Archive.org search API and downloads Amharic book PDFs and direct texts.
"""

import argparse
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set

SEARCH_API = "https://archive.org/advancedsearch.php"
METADATA_API = "https://archive.org/metadata"
DOWNLOAD_BASE = "https://archive.org/download"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

# Curated list of verified Amharic books & periodicals on archive.org
CURATED_ITEMS = [
    {"identifier": "20251108_20251108_2117", "title": "አልቦ ዘመድ - ብርሃኑ ድንቄ", "creator": "ብርሃኑ ድንቄ"},
    {"identifier": "20220323_20220323_2327", "title": "አደፍርስ", "creator": "ዳኛቸው ወርቁ"},
    {"identifier": "20240120_20240120_0927", "title": "ከሰውነት ውጪ የሆነ ልምድ", "creator": "ጳጉሜን"},
    {"identifier": "20240120_20240120_0948", "title": "የጳጉሜን ጥቅሶች", "creator": "ጳጉሜን"},
    {"identifier": "guramaile-1", "title": "ጉራማይሌ", "creator": "Unknown"},
    {"identifier": "20211011_20211011_0653", "title": "ኢትኦጵ ያዕቆብ ኃይለማሪያም ቅንጅት", "creator": "ያዕቆብ ኃይለማሪያም"},
    {"identifier": "tessemaeshete", "title": "ተሰማ እሸቴ (Tessema Eshete)", "creator": "ተሰማ እሸቴ"},
    {"identifier": "yekatit-magazine-1970-ec", "title": "የካቲት መጽሔት (Yekatit Magazine)", "creator": "የካቲት"},
    {"identifier": "AtseYohannesIV", "title": "አፄ ዮሐንስ አራተኛ (Atse Yohannes IV)", "creator": "Unknown"},
    {"identifier": "Gadla-kidus-kirkos-ena-kidst-eyeluta", "title": "ገድለ እየሉጣ እና ቂርቆስ", "creator": "Ethiopian Orthodox Church"},
    {"identifier": "AmharicBibleDictionary", "title": "Amharic Bible Dictionary", "creator": "EOTC"},
]


def contains_ethiopic(text: str) -> bool:
    """Check if string contains any Ethiopic script characters."""
    return any(0x1200 <= ord(c) <= 0x137F for c in text)


def count_ethiopic_chars(text: str) -> int:
    """Count number of Ethiopic characters."""
    return sum(1 for c in text if 0x1200 <= ord(c) <= 0x137F)


def search_amharic_items(rows: int = 100) -> List[Dict]:
    """Search Archive.org for Amharic texts using multiple search queries."""
    queries = [
        "subject:amharic AND mediatype:texts",
        "title:amharic AND mediatype:texts",
        "title:(አማርኛ OR መጽሐፍ OR ታሪክ OR ልብወለድ OR ገድለ OR ድርሳን OR ስብከት OR ጳጉሜን OR አደፍርስ OR ቄደር OR ድንቄ) AND mediatype:texts",
    ]

    all_docs = []
    seen_ids = set()

    for q in queries:
        params = {
            "q": q,
            "fl[]": ["identifier", "title", "creator", "year", "downloads", "description", "item_size"],
            "sort[]": ["downloads desc"],
            "rows": rows,
            "page": 1,
            "output": "json",
        }
        url = f"{SEARCH_API}?{urllib.parse.urlencode(params, doseq=True)}"
        print(f"[InternetArchive] Querying: {q[:60]}...")
        req = urllib.request.Request(url, headers=HEADERS)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                docs = data.get("response", {}).get("docs", [])
                print(f"  -> Found {len(docs)} documents.")
                for d in docs:
                    ident = d.get("identifier")
                    if ident and ident not in seen_ids:
                        seen_ids.add(ident)
                        all_docs.append(d)
        except Exception as e:
            print(f"  -> Search query failed: {e}")

    return all_docs


def get_item_files(identifier: str) -> List[Dict]:
    """Fetch files list for a specific archive.org item."""
    url = f"{METADATA_API}/{identifier}/files"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("result", [])
    except Exception as e:
        print(f"  Error fetching file list for {identifier}: {e}")
        return []


def download_ia_file(url: str, dest_path: Path, timeout: int = 120, retries: int = 2) -> bool:
    """Download a file with timeout and retry using urllib."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True

    temp_path = dest_path.with_suffix(".tmp")
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(temp_path, "wb") as f:
                shutil.copyfileobj(resp, f)

            if temp_path.exists() and temp_path.stat().st_size > 500:
                temp_path.rename(dest_path)
                return True
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                time.sleep(2)

    if temp_path.exists():
        temp_path.unlink()
    return False


def run_ia_scraper(
    dest_dir: Path,
    max_items: int = 40,
    max_pdf_size_mb: int = 45,
    download_pdfs: bool = True,
) -> List[Dict]:
    """Main scraper workflow for Internet Archive Amharic texts."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = dest_dir / "texts"
    pdfs_dir = dest_dir / "pdfs"
    texts_dir.mkdir(exist_ok=True)
    pdfs_dir.mkdir(exist_ok=True)

    manifest_file = dest_dir / "manifest.json"
    existing_manifest: Dict[str, Dict] = {}
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    existing_manifest[entry["identifier"]] = entry
        except Exception:
            pass

    # Start with curated items
    all_targets: List[Dict] = []
    seen_ids: Set[str] = set()

    for item in CURATED_ITEMS:
        ident = item["identifier"]
        seen_ids.add(ident)
        all_targets.append(item)

    # Search for additional items
    search_docs = search_amharic_items(rows=60)
    for doc in search_docs:
        ident = doc.get("identifier")
        title = doc.get("title", "")
        if not ident or ident in seen_ids:
            continue
        # Filter for genuine Amharic items
        if contains_ethiopic(title) or "amharic" in title.lower() or "amharic" in ident.lower():
            seen_ids.add(ident)
            all_targets.append(doc)

    print(f"\n[InternetArchive] Processing {len(all_targets)} candidate items (limit={max_items})...")
    cataloged_books = []
    processed_count = 0

    for doc in all_targets:
        if processed_count >= max_items:
            break

        identifier = doc.get("identifier")
        title = doc.get("title", identifier)
        creator = doc.get("creator", "Unknown")
        year = doc.get("year", "")
        downloads = doc.get("downloads", 0)

        print(f"\n[{processed_count+1}] Inspecting '{title}' ({identifier})...")

        # Check existing manifest
        if identifier in existing_manifest:
            manifest_entry = existing_manifest[identifier]
            if (manifest_entry.get("local_txt") and Path(manifest_entry["local_txt"]).exists()) or \
               (manifest_entry.get("local_pdfs") and any(Path(p).exists() for p in manifest_entry["local_pdfs"])):
                cataloged_books.append(manifest_entry)
                processed_count += 1
                print(f"  -> Already in manifest ({title})")
                continue

        files = get_item_files(identifier)
        if not files:
            continue

        # Find direct text files
        txt_files = [f for f in files if f.get("name", "").endswith(".txt") and not f.get("name", "").endswith("_meta.txt")]
        # Find PDF files (filter under max size)
        pdf_files = [
            f for f in files
            if f.get("name", "").lower().endswith(".pdf")
            and (int(f.get("size", 0)) / (1024 * 1024)) <= max_pdf_size_mb
            and int(f.get("size", 0)) > 1000
        ]

        local_txt_path = None
        local_pdf_paths = []

        # 1. Try any direct text file
        for tf in txt_files:
            tname = tf["name"]
            size = int(tf.get("size", 0))
            if size < 1000:
                continue
            txt_url = f"{DOWNLOAD_BASE}/{identifier}/{urllib.parse.quote(tname)}"
            target_txt = texts_dir / f"{identifier}_{Path(tname).stem}.txt"
            print(f"  Testing direct text: {tname} ({size // 1024} KB)...")
            if download_ia_file(txt_url, target_txt, timeout=60):
                try:
                    content = target_txt.read_text(encoding="utf-8", errors="ignore")
                    eth_count = count_ethiopic_chars(content)
                    if eth_count >= 500:
                        local_txt_path = str(target_txt.resolve())
                        print(f"  -> Verified genuine Amharic text ({eth_count:,} Ethiopic chars)!")
                        break
                    else:
                        target_txt.unlink(missing_ok=True)
                except Exception:
                    pass

        # 2. Download PDF files for OCR if download_pdfs is enabled
        if download_pdfs and pdf_files:
            # Sort preferred: text.pdf or smaller first
            pdf_files.sort(key=lambda x: int(x.get("size", 0)))
            for pf in pdf_files[:3]:  # Up to 3 volumes per collection
                pname = pf["name"]
                psize_mb = int(pf.get("size", 0)) / (1024 * 1024)
                safe_stem = re.sub(r"[^\w\-.]", "_", Path(pname).stem)
                target_pdf = pdfs_dir / f"{identifier}_{safe_stem}.pdf"
                pdf_url = f"{DOWNLOAD_BASE}/{identifier}/{urllib.parse.quote(pname)}"
                print(f"  Downloading PDF: {pname} ({psize_mb:.1f} MB)...", end="", flush=True)
                if download_ia_file(pdf_url, target_pdf, timeout=120):
                    local_pdf_paths.append(str(target_pdf.resolve()))
                    print(f" done ({target_pdf.stat().st_size // 1024} KB)")
                else:
                    print(" failed")

        if local_txt_path or local_pdf_paths:
            entry = {
                "id": f"ia_{identifier}",
                "identifier": identifier,
                "title": title,
                "creator": creator,
                "year": year,
                "downloads": downloads,
                "source": "archive.org",
                "source_url": f"https://archive.org/details/{identifier}",
                "local_txt": local_txt_path,
                "local_pdfs": local_pdf_paths,
            }
            cataloged_books.append(entry)
            existing_manifest[identifier] = entry
            processed_count += 1

            # Update manifest incrementally
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(cataloged_books, f, ensure_ascii=False, indent=2)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(cataloged_books, f, ensure_ascii=False, indent=2)

    print(f"\n[InternetArchive] Finished! {len(cataloged_books)} items cataloged in {manifest_file}.")
    return cataloged_books


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Amharic books from Internet Archive")
    parser.add_argument("--dest-dir", type=str, default="data/books_raw/ia")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of books to download")
    parser.add_argument("--max-mb", type=int, default=45, help="Max PDF size in MB")
    args = parser.parse_args()

    run_ia_scraper(
        dest_dir=Path(args.dest_dir),
        max_items=args.limit,
        max_pdf_size_mb=args.max_mb,
        download_pdfs=True,
    )
