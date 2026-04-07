#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
digi_fetcher.py
===============
Headless command-line tool to search and download newspaper texts
(ALTO XML or plain TXT) from digi.kansalliskirjasto.fi.

Reverse-engineered from the official NatLibFi digidownloadtool
(https://github.com/NatLibFi/digidownloadtool), adapted for
scriptable, GUI-free bulk extraction.

QUICK START
-----------
  pip install requests tqdm

  # Search for river ice keywords, Finnish newspapers, 1880-1920:
  python digi_fetcher.py --query "jää" --start 1880-01-01 --end 1920-12-31

  # Download full ALTO XML files too:
  python digi_fetcher.py --query "jäätyminen" --start 1860-01-01 --end 1900-12-31 --download --format alto

  # Multiple keywords (OR logic — repeat for each):
  python digi_fetcher.py --query "jää joki" --start 1850-01-01 --end 1939-12-31 --download

  # Limit to a specific newspaper by ISSN (no keyword needed):
  python digi_fetcher.py --issn 0356-1429 --start 1880-01-01 --end 1920-12-31 --download

  # Download everything in a date range from one newspaper, no keyword filter:
  python digi_fetcher.py --issn 0355-6913 --start 1870-01-01 --end 1900-12-31 --download --all-pages

  # Browse all available issues in a period (no keyword, no ISSN):
  python digi_fetcher.py --start 1850-01-01 --end 1860-12-31 --manifest-only

  # Only show the search manifest (no download):
  python digi_fetcher.py --query "jäänlähtö" --start 1870-01-01 --end 1939-12-31 --manifest-only

PARAMETERS
----------
  --query       Search keyword(s), e.g. "jää" or "jää joki"
  --start       Start date  YYYY-MM-DD  (default: 1820-01-01)
  --end         End date    YYYY-MM-DD  (default: 1939-12-31)
  --issn        Filter to a specific newspaper ISSN (optional)
  --language    Language filter: fi / sv / de / all  (default: fi)
  --format      What to download: alto | txt | both  (default: alto)
  --download    Actually download files (without this flag, only lists results)
  --all-pages   Download every page of each issue (slow). Default: only hit page.
  --out-dir     Output directory  (default: ./digi_output)
  --workers     Parallel download threads  (default: 5)
  --manifest-only  Save a CSV manifest of all matching issues, no download
  --max-results    Stop after this many bindings (0 = no limit, default: 0)
  --resume      Resume a previous interrupted download (reads manifest CSV)

OUTPUT STRUCTURE
----------------
  digi_output/
    manifest.csv              ← all found bindings + metadata
    {issn}/
      {year}/
        {bindingId}_page-00001.xml    ← ALTO XML (OCR text, structured)
        {bindingId}_page-00001.txt    ← Plain text version

ALTO XML NOTE
-------------
  ALTO XML files contain the full OCR text of a newspaper page in a structured
  XML format. To extract just the text from an ALTO file, use:

    from xml.etree import ElementTree as ET
    tree = ET.parse("file.xml")
    words = [el.get("CONTENT","") for el in tree.iter() if el.get("CONTENT")]
    text = " ".join(words)
"""

import argparse
import concurrent.futures
import csv
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from socket import timeout as SocketTimeout

try:
    import requests
    USE_REQUESTS = True
except ImportError:
    USE_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL          = "https://digi.kansalliskirjasto.fi"
BINDING_SEARCH    = f"{BASE_URL}/api/dam/binding-search"
TITLES_API        = f"{BASE_URL}/api/newspaper/titles"
REQUEST_HEADERS   = {
    "User-Agent":  "DigiResearchFetcher/1.0 (academic; river ice thesis)",
    "Referer":     "https://digi.kansalliskirjasto.fi/",
    "Connection":  "keep-alive",
}
RETRY_LIMIT       = 4
SLEEP_BETWEEN     = 0.2   # seconds between requests (polite rate limiting)

# ── Utility ───────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def http_get_json(url: str, retries: int = RETRY_LIMIT) -> dict | None:
    """GET a URL and parse JSON response. Returns None on failure."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=REQUEST_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                import json
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, SocketTimeout, ConnectionError) as e:
            if attempt == retries - 1:
                log(f"Failed after {retries} attempts: {url} — {e}", "ERROR")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def http_download_file(url: str, dest_path: str, retries: int = RETRY_LIMIT) -> bool:
    """Download a URL to a local file. Returns True on success."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=REQUEST_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(resp.read())
            return True
        except Exception as e:
            if attempt == retries - 1:
                log(f"Download failed: {url} — {e}", "WARN")
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


def alto_xml_to_text(xml_path: str) -> str:
    """Extract plain text from an ALTO XML file. Returns joined word content."""
    try:
        from xml.etree import ElementTree as ET
        tree = ET.parse(xml_path)
        words = [
            el.get("CONTENT", "")
            for el in tree.iter()
            if el.get("CONTENT") is not None
        ]
        return " ".join(w for w in words if w.strip())
    except Exception as e:
        log(f"Could not parse ALTO XML {xml_path}: {e}", "WARN")
        return ""

# ── Search / binding discovery ────────────────────────────────────────────────

def build_search_url(query, start: str, end: str,
                     issn: str = None, language: str = "fi") -> str:
    """
    Build the digi.kansalliskirjasto.fi search URL that the binding-search
    API expects as its query parameters.

    query is optional — omitting it returns all issues matching the other
    filters (date range, ISSN, language), useful for downloading a specific
    newspaper in full or browsing available coverage without a keyword filter.
    """
    params = {
        "startDate":     start,
        "endDate":       end,
        "formats":       "NEWSPAPER",
        "requireOnline": "true",
    }
    if query:
        params["query"] = query
    if issn:
        params["publicationId"] = issn
    if language and language != "all":
        params["language"] = language

    return "?" + urllib.parse.urlencode(params)


def fetch_all_bindings(query_string: str,
                       max_results: int = 0,
                       verbose: bool = True) -> list[dict]:
    """
    Paginate through the binding-search API and collect all matching bindings.

    The API uses scroll-based pagination:
      - First call:  /api/dam/binding-search?<params>
      - Next calls:  /api/dam/binding-search/<scrollId>

    Each response contains:
      rows[]            — list of binding metadata dicts
      scrollId          — cursor for next page
      totalResults      — total count
      pageImageTemplate — URL template for page JPGs,  e.g. "/image/{{page}}"
      altoXmlTemplate   — URL template for ALTO XML,   e.g. "/page-{{page}}.xml"
      altoTxtTemplate   — URL template for plain TXT,  e.g. "/page-{{page}}.txt"
      bindingPageCounts — {bindingId: pageCount}
    """
    url = BINDING_SEARCH + query_string
    all_rows = []
    page_counts = {}
    format_templates = {}
    scroll_id = None
    total = None
    is_first = True

    while True:
        if not is_first and scroll_id:
            url = f"{BINDING_SEARCH}/{scroll_id}"

        result = http_get_json(url)
        if result is None:
            log("binding-search API returned no result — stopping pagination.", "ERROR")
            break

        # Capture URL templates from first response
        if is_first:
            format_templates = {
                "pageImage": result.get("pageImageTemplate", "/image/{{page}}"),
                "altoXml":   result.get("altoXmlTemplate",  "/page-{{page}}.xml"),
                "altoTxt":   result.get("altoTxtTemplate",  "/page-{{page}}.txt"),
            }
            scroll_id = result.get("scrollId")
            total     = result.get("totalResults", 0)
            if verbose:
                log(f"Total matching bindings found: {total}")
            is_first = False

        rows = result.get("rows", [])
        if not rows:
            break   # No more pages

        # Merge page counts
        page_counts.update(result.get("bindingPageCounts", {}))
        all_rows.extend(rows)

        if verbose:
            log(f"  Fetched {len(all_rows)}/{total} binding metadata records...")

        if max_results > 0 and len(all_rows) >= max_results:
            all_rows = all_rows[:max_results]
            log(f"  Stopping at --max-results={max_results}")
            break

        time.sleep(SLEEP_BETWEEN)

    # Attach page count to each row for convenience
    for row in all_rows:
        bid = str(row.get("bindingId", ""))
        row["_pageCount"] = page_counts.get(bid, 1)

    return all_rows, format_templates


# ── Manifest / CSV ─────────────────────────────────────────────────────────────

MANIFEST_FIELDS = [
    "bindingId", "bindingTitle", "publicationId", "date", "year",
    "issue", "generalType", "publisher", "pageNumber", "pageCount",
    "baseUrl", "pdfUrl", "url", "copyrightWarnings",
    "textHighlights", "terms", "searchQuery",
]

def rows_to_manifest(rows: list[dict], query: str) -> list[dict]:
    """Normalise raw API rows into flat manifest dicts."""
    records = []
    for row in rows:
        date = row.get("date", "")
        year = date[:4] if date else ""
        highlights = row.get("textHighlights", {})
        if isinstance(highlights, dict):
            highlights = " ### ".join(highlights.get("text", []))
        records.append({
            "bindingId":       str(row.get("bindingId", "")),
            "bindingTitle":    row.get("bindingTitle", ""),
            "publicationId":   row.get("publicationId", ""),
            "date":            date,
            "year":            year,
            "issue":           row.get("issue", ""),
            "generalType":     row.get("generalType", ""),
            "publisher":       row.get("publisher", ""),
            "pageNumber":      str(row.get("pageNumber", "1")),
            "pageCount":       str(row.get("_pageCount", "1")),
            "baseUrl":         row.get("baseUrl", ""),
            "pdfUrl":          row.get("pdfUrl", ""),
            "url":             row.get("url", ""),
            "copyrightWarnings": str(row.get("copyrightWarnings", "")),
            "textHighlights":  highlights if isinstance(highlights, str) else "",
            "terms":           ",".join(row.get("terms", []) or []),
            "searchQuery":     query,
        })
    return records


def save_manifest(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    log(f"Manifest saved → {path}  ({len(records)} bindings)")


def load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Download logic ─────────────────────────────────────────────────────────────

def build_file_url(base_url: str, template: str, page: int) -> str:
    """Fill the {{page}} placeholder in a URL template."""
    page_str = str(page).zfill(5)   # e.g. 00001
    return base_url + template.replace("{{page}}", page_str)


def download_binding(record: dict,
                     format_templates: dict,
                     out_dir: str,
                     fmt: str = "alto",       # alto | txt | both
                     all_pages: bool = False) -> dict:
    """
    Download one binding (newspaper issue).
    Returns a status dict with 'success', 'files', 'binding_id'.
    """
    binding_id  = record["bindingId"]
    base_url    = record["baseUrl"]
    issn        = record["publicationId"]
    year        = record["year"] or "unknown"
    page_count  = int(record["pageCount"] or 1)
    hit_page    = int(record["pageNumber"] or 1)

    # Determine which pages to download
    pages_to_get = range(1, page_count + 1) if all_pages else [hit_page]

    downloaded = []
    failed     = []

    for page in pages_to_get:
        page_str = str(page).zfill(5)

        # ALTO XML
        if fmt in ("alto", "both"):
            url   = build_file_url(base_url, format_templates["altoXml"], page)
            dest  = os.path.join(out_dir, issn, year,
                                 f"{binding_id}_page-{page_str}.xml")
            if not os.path.exists(dest):   # skip if already downloaded
                ok = http_download_file(url, dest)
                (downloaded if ok else failed).append(dest)
            else:
                downloaded.append(dest)   # count as success (already present)

        # Plain TXT
        if fmt in ("txt", "both"):
            url   = build_file_url(base_url, format_templates["altoTxt"], page)
            dest  = os.path.join(out_dir, issn, year,
                                 f"{binding_id}_page-{page_str}.txt")
            if not os.path.exists(dest):
                ok = http_download_file(url, dest)
                (downloaded if ok else failed).append(dest)
            else:
                downloaded.append(dest)

        time.sleep(SLEEP_BETWEEN)

    return {
        "binding_id": binding_id,
        "success":    len(failed) == 0,
        "downloaded": len(downloaded),
        "failed":     len(failed),
    }


def download_all(records: list[dict],
                 format_templates: dict,
                 out_dir: str,
                 fmt: str = "alto",
                 all_pages: bool = False,
                 workers: int = 5):
    """Download all bindings in parallel with a thread pool."""
    log(f"Starting download of {len(records)} bindings "
        f"(format={fmt}, all_pages={all_pages}, workers={workers})")

    total_ok   = 0
    total_fail = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_binding, rec, format_templates, out_dir, fmt, all_pages
            ): rec
            for rec in records
        }

        iterator = concurrent.futures.as_completed(futures)
        if HAS_TQDM:
            iterator = tqdm(iterator, total=len(futures), desc="Downloading")

        for future in iterator:
            try:
                result = future.result()
                if result["success"]:
                    total_ok += 1
                else:
                    total_fail += 1
                    log(f"  ✗ binding {result['binding_id']} had "
                        f"{result['failed']} failed pages", "WARN")
            except Exception as e:
                total_fail += 1
                log(f"  ✗ unexpected error: {e}", "WARN")

    log(f"Download complete — ✓ {total_ok} bindings ok, ✗ {total_fail} failed")


# ── Pretty summary ─────────────────────────────────────────────────────────────

def print_summary(records: list[dict], args):
    print()
    print("=" * 72)
    print(f"  SEARCH RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Query       : {args.query or '(none — all issues in range)'}")
    print(f"  Date range  : {args.start}  →  {args.end}")
    if args.issn:
        print(f"  ISSN filter : {args.issn}")
    print(f"  Total found : {len(records)} binding(s)")
    print()

    # Group by newspaper title
    from collections import Counter
    by_title = Counter(r["bindingTitle"] for r in records)
    print(f"  {'Newspaper title':<45} {'Issues':>6}")
    print(f"  {'-'*45} {'------':>6}")
    for title, count in by_title.most_common(20):
        print(f"  {title[:45]:<45} {count:>6}")
    if len(by_title) > 20:
        print(f"  ... and {len(by_title)-20} more newspapers (see manifest CSV)")

    # Year distribution (bucketed by decade)
    years = [int(r["year"]) for r in records if r["year"].isdigit()]
    if years:
        print()
        print(f"  Issues per decade:")
        from collections import defaultdict
        by_decade = defaultdict(int)
        for y in years:
            by_decade[(y // 10) * 10] += 1
        for decade in sorted(by_decade):
            bar = "█" * min(40, by_decade[decade] // max(1, max(by_decade.values()) // 40))
            print(f"  {decade}s  {bar}  {by_decade[decade]}")

    print("=" * 72)
    print()


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch Finnish newspaper texts from digi.kansalliskirjasto.fi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--query",        default=None,
                   help='Search keyword(s), e.g. "jää" or "jäätyminen joki". '
                        'Optional — omit to fetch all issues matching the date/ISSN filters.')
    p.add_argument("--start",        default="1820-01-01",
                   help="Start date YYYY-MM-DD (default: 1820-01-01)")
    p.add_argument("--end",          default="1939-12-31",
                   help="End date YYYY-MM-DD (default: 1939-12-31)")
    p.add_argument("--issn",         default=None,
                   help="Limit to a single newspaper by ISSN (optional)")
    p.add_argument("--language",     default="fi",
                   choices=["fi", "sv", "de", "all"],
                   help="Language filter (default: fi)")
    p.add_argument("--format",       default="alto",
                   choices=["alto", "txt", "both"],
                   help="File format to download: alto (XML), txt, or both (default: alto)")
    p.add_argument("--download",     action="store_true",
                   help="Actually download files (default: list only)")
    p.add_argument("--all-pages",    action="store_true",
                   help="Download every page of each issue (default: hit page only)")
    p.add_argument("--out-dir",      default="./digi_output",
                   help="Output directory (default: ./digi_output)")
    p.add_argument("--workers",      type=int, default=5,
                   help="Parallel download threads (default: 5)")
    p.add_argument("--manifest-only",action="store_true",
                   help="Save manifest CSV only, skip downloading")
    p.add_argument("--max-results",  type=int, default=0,
                   help="Stop after N bindings (0 = no limit)")
    p.add_argument("--resume",       default=None,
                   help="Path to a previous manifest.csv to resume an interrupted download")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir      = args.out_dir
    manifest_path = os.path.join(out_dir, "manifest.csv")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Get bindings (resume or fresh search) ─────────────────────────
    if args.resume:
        log(f"Resuming from manifest: {args.resume}")
        records = load_manifest(args.resume)
        # We need format templates — re-run one search call to get them
        query_string    = build_search_url(
            args.query, args.start, args.end, args.issn, args.language)
        _, format_templates = fetch_all_bindings(
            query_string, max_results=1, verbose=False)
        log(f"Loaded {len(records)} records from manifest.")
    else:
        query_label = f'"{args.query}"' if args.query else "(no keyword — all issues)"
        log(f'Searching: query={query_label} | {args.start} → {args.end} '
            f'| lang={args.language}' + (f' | issn={args.issn}' if args.issn else ''))

        query_string = build_search_url(
            args.query, args.start, args.end, args.issn, args.language)

        rows, format_templates = fetch_all_bindings(
            query_string,
            max_results=args.max_results,
            verbose=True
        )

        if not rows:
            log("No results found. Try different keywords or date range.")
            sys.exit(0)

        records = rows_to_manifest(rows, args.query)
        save_manifest(records, manifest_path)

    # ── Step 2: Show summary ──────────────────────────────────────────────────
    print_summary(records, args)

    # ── Step 3: Download (if requested) ──────────────────────────────────────
    if args.manifest_only:
        log("--manifest-only set. Skipping download.")
        log(f"Manifest at: {manifest_path}")
        return

    if not args.download and not args.resume:
        log("Tip: add --download to fetch the actual ALTO XML / text files.")
        log(f"     Manifest already saved to: {manifest_path}")
        return

    download_all(
        records         = records,
        format_templates= format_templates,
        out_dir         = out_dir,
        fmt             = args.format,
        all_pages       = args.all_pages,
        workers         = args.workers,
    )

    log(f"All done. Files in: {out_dir}")
    log(f"To extract plain text from an ALTO XML file:")
    log(f"  from digi_fetcher import alto_xml_to_text")
    log(f"  text = alto_xml_to_text('digi_output/issn/year/binding_page-00001.xml')")


if __name__ == "__main__":
    main()