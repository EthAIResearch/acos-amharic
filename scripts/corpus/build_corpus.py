"""End-to-End Amharic Corpus Builder.
Scrapes and integrates Amharic books from:
- ethiobookreview.com
- allaboutethio.com (all 10 categories)
- archive.org (Internet Archive Amharic texts)
- ethiopianorthodox.org (Holy books archive)

Extracts text via digital poppler, direct digitized text, or parallel Tesseract OCR,
normalizes and cleans Amharic text, and outputs standardized mC4 and OSCAR corpora.
"""

import argparse
import datetime
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from amharic_cleaner import clean_document
from amharic_ocr import extract_pdf_with_ocr
from scraper_ethiobookreview import run_ethiobookreview_scraper
from scraper_allaboutethio import run_allaboutethio_scraper
from scraper_internet_archive import run_ia_scraper
from scraper_eotc import run_eotc_scraper

CORPUS_DIR = Path("data/corpus")
BOOKS_RAW_DIR = Path("data/books_raw")
OCR_CACHE_DIR = BOOKS_RAW_DIR / "ocr_cache"


def extract_text_from_pdf(
    pdf_path: Path,
    enable_ocr: bool = True,
    ocr_workers: int = 4,
    ocr_max_pages: Optional[int] = None,
) -> Tuple[Optional[str], str]:
    """Extract text from a PDF file using pdftotext (digital) or Tesseract (OCR).

    Returns:
        (text, extraction_method) where method is 'digital', 'ocr', 'cached_ocr', or 'none'.
    """
    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        return None, "none"

    # 1. Try fast digital extraction first
    cmd = ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            text = res.stdout.decode("utf-8", errors="replace")
            ethiopic_chars = sum(1 for c in text if "\u1200" <= c <= "\u137F")
            if ethiopic_chars > 200:
                return text, "digital"
    except Exception:
        pass

    # 2. Check OCR cache if this book was already OCRed
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OCR_CACHE_DIR / f"{pdf_path.stem}.txt"
    if cache_file.exists() and cache_file.stat().st_size > 100:
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read(), "cached_ocr"

    # 3. Fall back to parallel OCR if enabled
    if enable_ocr:
        print(" [Running OCR]...", end="", flush=True)
        ocr_text = extract_pdf_with_ocr(
            pdf_path=pdf_path,
            workers=ocr_workers,
            max_pages=ocr_max_pages,
        )
        if ocr_text:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            return ocr_text, "ocr"

    return None, "none"


def process_book(
    book_meta: Dict[str, any],
    pdf_path: Optional[Path] = None,
    txt_path: Optional[Path] = None,
    doc_index: int = 0,
    min_doc_chars: int = 100,
    min_ethiopic_ratio: float = 0.50,
    enable_ocr: bool = True,
    ocr_workers: int = 4,
    ocr_max_pages: Optional[int] = None,
) -> Optional[Tuple[Dict[str, any], Dict[str, any]]]:
    """Extract text from book assets and prepare OSCAR and mC4 records."""
    raw_text = None
    method = "none"

    # 1. Direct text file (e.g., from archive.org pre-extracted OCR)
    if txt_path and txt_path.exists():
        try:
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
            method = "direct_text"
        except Exception:
            raw_text = None

    # 2. PDF extraction (digital poppler or local OCR)
    if not raw_text and pdf_path and pdf_path.exists():
        raw_text, method = extract_text_from_pdf(
            pdf_path,
            enable_ocr=enable_ocr,
            ocr_workers=ocr_workers,
            ocr_max_pages=ocr_max_pages,
        )

    if not raw_text:
        return None

    cleaned_text, quality = clean_document(
        raw_text=raw_text,
        min_doc_chars=min_doc_chars,
        min_ethiopic_ratio=min_ethiopic_ratio,
    )

    if not cleaned_text:
        return None

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    oscar_record = {
        "id": doc_index,
        "text": cleaned_text,
        "meta": {
            "language": "am",
            "title": book_meta.get("title", "Unknown"),
            "author": book_meta.get("author", book_meta.get("creator", None)),
            "source": book_meta.get("source", "Unknown"),
            "source_url": book_meta.get("source_url", book_meta.get("library_url", "")),
            "pdf_url": book_meta.get("pdf_url", (book_meta.get("pdf_urls", [""])[0] if book_meta.get("pdf_urls") else "")),
            "extraction_method": method,
            "timestamp": now_iso,
            "char_count": quality["char_count"],
            "word_count": quality["word_count"],
            "ethiopic_char_count": quality["ethiopic_char_count"],
            "ethiopic_ratio": quality["ethiopic_ratio"],
            "quality_warnings": quality["quality_warnings"],
        }
    }

    mc4_record = {
        "text": cleaned_text,
        "url": book_meta.get("source_url", book_meta.get("library_url", "")),
        "timestamp": now_iso,
    }

    return oscar_record, mc4_record


def build_corpus(
    output_dir: Path = CORPUS_DIR,
    scrape_ethiobookreview: bool = False,
    scrape_allaboutethio: bool = False,
    scrape_ia: bool = False,
    scrape_eotc: bool = False,
    allaboutethio_limit: Optional[int] = None,
    ia_limit: int = 30,
    eotc_limit: int = 30,
    min_doc_chars: int = 100,
    min_ethiopic_ratio: float = 0.50,
    enable_ocr: bool = True,
    ocr_workers: int = 4,
    ocr_max_pages: Optional[int] = 30,
) -> Dict[str, any]:
    """Execute the multi-source corpus generation workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ebr_dir = BOOKS_RAW_DIR / "ethiobookreview"
    aae_dir = BOOKS_RAW_DIR / "allaboutethio"
    ia_dir = BOOKS_RAW_DIR / "ia"
    eotc_dir = BOOKS_RAW_DIR / "eotc"

    all_items_to_process: List[Tuple[Dict[str, any], Optional[Path], Optional[Path]]] = []

    # 1. EthioBookReview
    manifest_ebr = ebr_dir / "manifest.json"
    if scrape_ethiobookreview:
        print("\n[Step] Scraping EthioBookReview...")
        ebr_books = run_ethiobookreview_scraper(ebr_dir, download_pdfs=True)
    elif manifest_ebr.exists():
        with open(manifest_ebr, "r", encoding="utf-8") as f:
            ebr_books = json.load(f)
    else:
        ebr_books = []

    for b in ebr_books:
        local_pdf = b.get("local_pdf")
        if not local_pdf or not Path(local_pdf).exists():
            candidate = ebr_dir / "pdfs" / f"{b['slug']}.pdf"
            if candidate.exists():
                local_pdf = str(candidate.resolve())
        if local_pdf and Path(local_pdf).exists():
            all_items_to_process.append((b, Path(local_pdf), None))

    # 2. AllAboutEthio
    manifest_aae = aae_dir / "manifest.json"
    if scrape_allaboutethio:
        print("\n[Step] Scraping AllAboutEthio...")
        aae_books = run_allaboutethio_scraper(aae_dir, download_pdfs=True, max_books=allaboutethio_limit)
    elif manifest_aae.exists():
        with open(manifest_aae, "r", encoding="utf-8") as f:
            aae_books = json.load(f)
    else:
        aae_books = []

    seen_aae_files: Set[str] = set()
    for b in aae_books:
        for url in b.get("pdf_urls", []):
            cand = aae_dir / "pdfs" / Path(url).name
            if cand.exists() and cand.stat().st_size > 1000 and cand.name not in seen_aae_files:
                all_items_to_process.append((b, cand, None))
                seen_aae_files.add(cand.name)
        for p in b.get("local_pdfs", []):
            path_obj = Path(p)
            if path_obj.exists() and path_obj.stat().st_size > 1000 and path_obj.name not in seen_aae_files:
                all_items_to_process.append((b, path_obj, None))
                seen_aae_files.add(path_obj.name)

    if (aae_dir / "pdfs").exists():
        for pdf_file in (aae_dir / "pdfs").glob("*.pdf"):
            if pdf_file.name not in seen_aae_files and pdf_file.stat().st_size > 1000:
                synthetic_meta = {
                    "title": pdf_file.stem,
                    "source": "allaboutethio.com",
                    "source_url": "https://allaboutethio.com/amharic-books.html"
                }
                all_items_to_process.append((synthetic_meta, pdf_file, None))
                seen_aae_files.add(pdf_file.name)

    # 3. Internet Archive
    manifest_ia = ia_dir / "manifest.json"
    if scrape_ia:
        print("\n[Step] Scraping Internet Archive...")
        ia_books = run_ia_scraper(ia_dir, max_items=ia_limit)
    elif manifest_ia.exists():
        with open(manifest_ia, "r", encoding="utf-8") as f:
            ia_books = json.load(f)
    else:
        ia_books = []

    seen_ia_items: Set[str] = set()
    for b in ia_books:
        txt_path = Path(b["local_txt"]) if b.get("local_txt") and Path(b["local_txt"]).exists() else None
        if txt_path:
            all_items_to_process.append((b, None, txt_path))
            seen_ia_items.add(txt_path.name)

        # Single PDF field
        if b.get("local_pdf") and Path(b["local_pdf"]).exists():
            p = Path(b["local_pdf"])
            if p.name not in seen_ia_items:
                all_items_to_process.append((b, p, None))
                seen_ia_items.add(p.name)

        # List of PDFs field
        if b.get("local_pdfs"):
            for lp in b["local_pdfs"]:
                p = Path(lp)
                if p.exists() and p.name not in seen_ia_items:
                    all_items_to_process.append((b, p, None))
                    seen_ia_items.add(p.name)

    if (ia_dir / "texts").exists():
        for txt_file in (ia_dir / "texts").glob("*.txt"):
            if txt_file.name not in seen_ia_items and txt_file.stat().st_size > 1000:
                synthetic_meta = {
                    "title": txt_file.stem.replace("_", " ").title(),
                    "source": "archive.org",
                    "source_url": f"https://archive.org/details/{txt_file.stem}"
                }
                all_items_to_process.append((synthetic_meta, None, txt_file))
                seen_ia_items.add(txt_file.name)

    if (ia_dir / "pdfs").exists():
        for pdf_file in (ia_dir / "pdfs").glob("*.pdf"):
            if pdf_file.name not in seen_ia_items and pdf_file.stat().st_size > 1000:
                synthetic_meta = {
                    "title": pdf_file.stem.replace("_", " ").title(),
                    "source": "archive.org",
                    "source_url": f"https://archive.org/details/{pdf_file.stem}"
                }
                all_items_to_process.append((synthetic_meta, pdf_file, None))
                seen_ia_items.add(pdf_file.name)

    # 4. Ethiopian Orthodox (EOTC)
    manifest_eotc = eotc_dir / "manifest.json"
    if scrape_eotc:
        print("\n[Step] Scraping Ethiopian Orthodox Archive...")
        eotc_books = run_eotc_scraper(eotc_dir, max_books=eotc_limit)
    elif manifest_eotc.exists():
        with open(manifest_eotc, "r", encoding="utf-8") as f:
            eotc_books = json.load(f)
    else:
        eotc_books = []

    seen_eotc_files: Set[str] = set()
    for b in eotc_books:
        cand = eotc_dir / "pdfs" / b.get("filename", "")
        if cand.exists() and cand.stat().st_size > 1000 and cand.name not in seen_eotc_files:
            all_items_to_process.append((b, cand, None))
            seen_eotc_files.add(cand.name)
        elif b.get("local_pdf") and Path(b["local_pdf"]).exists() and Path(b["local_pdf"]).name not in seen_eotc_files:
            path_obj = Path(b["local_pdf"])
            all_items_to_process.append((b, path_obj, None))
            seen_eotc_files.add(path_obj.name)

    if (eotc_dir / "pdfs").exists():
        for pdf_file in list((eotc_dir / "pdfs").glob("*.pdf")) + list((eotc_dir / "pdfs").glob("*.PDF")):
            if pdf_file.name not in seen_eotc_files and pdf_file.stat().st_size > 1000:
                synthetic_meta = {
                    "title": pdf_file.stem.replace("_", " ").title(),
                    "source": "ethiopianorthodox.org",
                    "source_url": "https://www.ethiopianorthodox.org/amharic/holybooks/"
                }
                all_items_to_process.append((synthetic_meta, pdf_file, None))
                seen_eotc_files.add(pdf_file.name)

    # 5. Process all gathered books
    print("\n==========================================")
    print(f"PROCESSING {len(all_items_to_process)} BOOK ASSETS...")
    print("==========================================")

    oscar_records = []
    mc4_records = []
    seen_hashes: Set[str] = set()

    doc_counter = 0
    scanned_or_skipped = 0

    for book_meta, pdf_path, txt_path in all_items_to_process:
        title = book_meta.get("title", pdf_path.stem if pdf_path else "book")
        fname = pdf_path.name if pdf_path else (txt_path.name if txt_path else "asset")
        print(f"Processing '{title}' ({fname})...", end="", flush=True)

        res = process_book(
            book_meta=book_meta,
            pdf_path=pdf_path,
            txt_path=txt_path,
            doc_index=doc_counter,
            min_doc_chars=min_doc_chars,
            min_ethiopic_ratio=min_ethiopic_ratio,
            enable_ocr=enable_ocr,
            ocr_workers=ocr_workers,
            ocr_max_pages=ocr_max_pages,
        )

        if res:
            oscar_rec, mc4_rec = res

            # Deduplication by content hash
            text_hash = hashlib.sha256(oscar_rec["text"][:2000].encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                print(" SKIPPED (duplicate content)")
                continue
            seen_hashes.add(text_hash)

            oscar_records.append(oscar_rec)
            mc4_records.append(mc4_rec)
            doc_counter += 1
            words = oscar_rec["meta"]["word_count"]
            chars = oscar_rec["meta"]["char_count"]
            ratio = oscar_rec["meta"]["ethiopic_ratio"]
            method = oscar_rec["meta"]["extraction_method"]
            print(f" ACCEPTED [{method}] ({words:,} words, {chars:,} chars, Ethiopic: {ratio:.1%})")
        else:
            scanned_or_skipped += 1
            print(" SKIPPED (insufficient Amharic text)")

    # 6. Export Corpora
    print("\n==========================================")
    print("EXPORTING CORPUS FILES...")
    print("==========================================")

    oscar_gz_path = output_dir / "amharic_corpus_oscar.jsonl.gz"
    oscar_jsonl_path = output_dir / "amharic_corpus_oscar.jsonl"
    mc4_gz_path = output_dir / "amharic_corpus_mc4.jsonl.gz"
    mc4_jsonl_path = output_dir / "amharic_corpus_mc4.jsonl"

    # Write OSCAR
    with open(oscar_jsonl_path, "w", encoding="utf-8") as f_out, \
         gzip.open(oscar_gz_path, "wt", encoding="utf-8") as f_gz:
        for rec in oscar_records:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            f_out.write(line)
            f_gz.write(line)

    # Write mC4
    with open(mc4_jsonl_path, "w", encoding="utf-8") as f_out, \
         gzip.open(mc4_gz_path, "wt", encoding="utf-8") as f_gz:
        for rec in mc4_records:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            f_out.write(line)
            f_gz.write(line)

    # 7. Summary & Stats
    total_docs = len(oscar_records)
    total_words = sum(r["meta"]["word_count"] for r in oscar_records)
    total_chars = sum(r["meta"]["char_count"] for r in oscar_records)
    total_ethiopic_chars = sum(r["meta"]["ethiopic_char_count"] for r in oscar_records)

    stats = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_documents": total_docs,
        "skipped_or_failed": scanned_or_skipped,
        "total_words": total_words,
        "total_characters": total_chars,
        "total_ethiopic_characters": total_ethiopic_chars,
        "average_words_per_doc": round(total_words / total_docs, 1) if total_docs else 0,
        "overall_ethiopic_char_ratio": round(total_ethiopic_chars / total_chars, 4) if total_chars else 0,
        "output_files": {
            "oscar_compressed": str(oscar_gz_path.resolve()),
            "oscar_jsonl": str(oscar_jsonl_path.resolve()),
            "mc4_compressed": str(mc4_gz_path.resolve()),
            "mc4_jsonl": str(mc4_jsonl_path.resolve()),
        },
        "included_books": [
            {
                "id": r["id"],
                "title": r["meta"]["title"],
                "author": r["meta"]["author"],
                "extraction_method": r["meta"]["extraction_method"],
                "words": r["meta"]["word_count"],
                "characters": r["meta"]["char_count"],
                "source": r["meta"]["source"],
                "url": r["meta"]["source_url"],
            }
            for r in oscar_records
        ]
    }

    stats_file = output_dir / "corpus_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n==========================================")
    print("CORPUS GENERATION COMPLETE")
    print("==========================================")
    print(f"Accepted Documents : {total_docs}")
    print(f"Skipped / Failed   : {scanned_or_skipped}")
    print(f"Total Words        : {total_words:,}")
    print(f"Total Characters   : {total_chars:,}")
    print(f"Ethiopic Characters: {total_ethiopic_chars:,}")
    print(f"OSCAR Corpus Path  : {oscar_gz_path}")
    print(f"mC4 Corpus Path    : {mc4_gz_path}")
    print(f"Detailed Stats     : {stats_file}")
    print("==========================================\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Amharic Corpus from Multiple Sources")
    parser.add_argument("--output-dir", type=str, default="data/corpus")
    parser.add_argument("--scrape-aae", action="store_true", help="Scrape AllAboutEthio")
    parser.add_argument("--aae-limit", type=int, default=None, help="Limit AllAboutEthio books")
    parser.add_argument("--scrape-ebr", action="store_true", help="Scrape EthioBookReview")
    parser.add_argument("--scrape-ia", action="store_true", help="Scrape Internet Archive")
    parser.add_argument("--ia-limit", type=int, default=30, help="Internet Archive book limit")
    parser.add_argument("--scrape-eotc", action="store_true", help="Scrape Ethiopian Orthodox Archive")
    parser.add_argument("--eotc-limit", type=int, default=30, help="EOTC book limit")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    parser.add_argument("--ocr-workers", type=int, default=4, help="Number of parallel OCR worker processes")
    parser.add_argument("--ocr-max-pages", type=int, default=30, help="Cap OCR pages per book")
    args = parser.parse_args()

    build_corpus(
        output_dir=Path(args.output_dir),
        scrape_ethiobookreview=args.scrape_ebr,
        scrape_allaboutethio=args.scrape_aae,
        allaboutethio_limit=args.aae_limit,
        scrape_ia=args.scrape_ia,
        ia_limit=args.ia_limit,
        scrape_eotc=args.scrape_eotc,
        eotc_limit=args.eotc_limit,
        enable_ocr=not args.no_ocr,
        ocr_workers=args.ocr_workers,
        ocr_max_pages=args.ocr_max_pages,
    )
