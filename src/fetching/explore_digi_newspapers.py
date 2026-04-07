# explore_digi_newspapers.py
# Place the full explore_digi_newspapers.py script here.
#
# Usage:
#   python explore_digi_newspapers.py --start 1850 --end 1939 --no-probe
"""
explore_digi_newspapers.py
==========================
Explore the Finnish National Library's digitized newspaper collection
at digi.kansalliskirjasto.fi and check which newspapers have accessible
ALTO XML (full text) files for a given time range.

HOW IT WORKS:
  1. Fetches all newspaper titles from the Digi API
  2. Filters by your desired date range
  3. For each newspaper, probes a sample issue to confirm ALTO XML access
  4. Saves a summary CSV + prints a human-readable table

USAGE:
  pip install requests tqdm pandas
  python explore_digi_newspapers.py

  Optional flags:
    --start  Start year (default: 1820)
    --end    End year  (default: 1940)
    --probe  How many issues to probe per newspaper for ALTO check (default: 1)
    --out    Output CSV filename (default: newspaper_coverage.csv)

Author: generated for Pooya's river-ice thesis project
"""

import requests
import argparse
import csv
import time
import sys
from datetime import datetime

# ── Try importing optional dependencies ──────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] tqdm not found. Install with: pip install tqdm  (progress bars disabled)")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL   = "https://digi.kansalliskirjasto.fi"
TITLES_API = f"{BASE_URL}/api/newspaper/titles"
SEARCH_API = f"{BASE_URL}/api/search"
OAI_API    = f"{BASE_URL}/interfaces/OAI-PMH"

HEADERS = {
    "User-Agent": "RiverIceResearch/1.0 (academic thesis; contact via university)"
}

# ── Helper functions ──────────────────────────────────────────────────────────

def fetch_all_titles(language: str = "fi") -> list[dict]:
    """
    Fetch the complete list of newspaper titles from the Digi API.
    Returns a list of dicts with keys: issn, title, startYear, endYear, language, etc.
    """
    print(f"\n[1/3] Fetching all newspaper titles from Digi API (language={language})...")
    try:
        resp = requests.get(
            TITLES_API,
            params={"language": language},
            headers=HEADERS,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        # The API can return a list directly or a dict with a 'titles' key
        if isinstance(data, list):
            titles = data
        elif isinstance(data, dict):
            titles = data.get("titles") or data.get("results") or list(data.values())[0]
        else:
            titles = []

        print(f"    ✓ Found {len(titles)} newspaper titles total.")
        return titles

    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to digi.kansalliskirjasto.fi")
        print("        Make sure you have internet access and the domain is reachable.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to fetch titles: {e}")
        sys.exit(1)


def filter_by_year_range(titles: list[dict], start: int, end: int) -> list[dict]:
    """
    Keep only titles that overlap with the requested [start, end] year range.
    Handles missing/None start or end years gracefully.
    """
    filtered = []
    for t in titles:
        t_start = t.get("startYear") or t.get("start_year") or t.get("firstIssueDate", "")
        t_end   = t.get("endYear")   or t.get("end_year")   or t.get("lastIssueDate", "")

        # Extract year from date strings like "1880-03-12"
        if isinstance(t_start, str) and len(t_start) >= 4:
            t_start = int(t_start[:4])
        if isinstance(t_end, str) and len(t_end) >= 4:
            t_end = int(t_end[:4])

        # If we can't determine dates, keep the title (don't exclude by default)
        if not isinstance(t_start, int):
            t_start = 1700
        if not isinstance(t_end, int):
            t_end = datetime.now().year

        # Overlap check: title range intersects with [start, end]
        if t_start <= end and t_end >= start:
            t["_filtered_start"] = max(t_start, start)
            t["_filtered_end"]   = min(t_end, end)
            filtered.append(t)

    return filtered


def find_sample_binding(issn: str, year: int) -> int | None:
    """
    Search for a binding ID for a given newspaper (by ISSN) around the given year.
    Returns the binding ID (int) or None if not found.
    """
    try:
        # Try the search endpoint to find an issue
        resp = requests.get(
            SEARCH_API,
            params={
                "query": "",
                "issn":  issn,
                "startDate": f"{year}-01-01",
                "endDate":   f"{year}-12-31",
                "requireOnline": "true",
                "maxResults": 1,
                "type": "NEWSPAPER"
            },
            headers=HEADERS,
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            # Look for bindingId in the response
            results = (
                data.get("results") or
                data.get("bindings") or
                data.get("items") or
                []
            )
            if results:
                r = results[0]
                bid = r.get("bindingId") or r.get("id") or r.get("binding_id")
                if bid:
                    return int(bid)
    except Exception:
        pass
    return None


def check_alto_access(binding_id: int) -> dict:
    """
    Given a binding ID, try to access:
      - METS file (to find page count and ALTO filenames)
      - Page 1 ALTO XML

    Returns a dict with 'mets_ok', 'alto_ok', 'page_count', 'sample_alto_url'.
    """
    result = {
        "mets_ok": False,
        "alto_ok": False,
        "page_count": None,
        "sample_alto_url": None
    }

    # 1. Try METS file
    mets_url = f"{BASE_URL}/sanomalehti/binding/{binding_id}/mets"
    try:
        r = requests.get(mets_url, headers=HEADERS, timeout=15)
        if r.status_code == 200 and "<mets" in r.text.lower():
            result["mets_ok"] = True
            # Count ALTO file references
            alto_count = r.text.lower().count("alto")
            result["page_count"] = max(1, alto_count // 2)  # rough estimate
    except Exception:
        pass

    # 2. Try fetching ALTO XML for page 1
    alto_url = f"{BASE_URL}/sanomalehti/binding/{binding_id}/page-00001.xml"
    result["sample_alto_url"] = alto_url
    try:
        r = requests.get(alto_url, headers=HEADERS, timeout=15)
        if r.status_code == 200 and ("<alto" in r.text.lower() or "<layout" in r.text.lower()):
            result["alto_ok"] = True
    except Exception:
        pass

    return result


def probe_newspaper(title: dict, probe_year: int) -> dict:
    """
    Full probe of a single newspaper: find a binding, check ALTO access.
    Returns enriched info dict.
    """
    issn  = title.get("issn", "")
    name  = title.get("title") or title.get("name") or "Unknown"

    info = {
        "title":          name,
        "issn":           issn,
        "language":       title.get("language", ""),
        "available_from": title.get("_filtered_start", ""),
        "available_to":   title.get("_filtered_end", ""),
        "binding_found":  False,
        "sample_binding": None,
        "alto_accessible": False,
        "mets_accessible": False,
        "est_pages":       None,
        "sample_alto_url": None,
    }

    binding_id = find_sample_binding(issn, probe_year)
    if binding_id:
        info["binding_found"]  = True
        info["sample_binding"] = binding_id

        alto_info = check_alto_access(binding_id)
        info["alto_accessible"] = alto_info["alto_ok"]
        info["mets_accessible"] = alto_info["mets_ok"]
        info["est_pages"]       = alto_info["page_count"]
        info["sample_alto_url"] = alto_info["sample_alto_url"]

    time.sleep(0.3)  # polite rate limiting
    return info


def print_summary_table(results: list[dict]) -> None:
    """Print a readable ASCII summary table."""
    alto_ok  = [r for r in results if r["alto_accessible"]]
    mets_ok  = [r for r in results if r["mets_accessible"]]
    no_bind  = [r for r in results if not r["binding_found"]]

    print("\n" + "="*80)
    print(f"  NEWSPAPER COVERAGE SUMMARY")
    print("="*80)
    print(f"  Total newspapers in date range : {len(results)}")
    print(f"  With ALTO XML accessible       : {len(alto_ok)}")
    print(f"  With METS accessible           : {len(mets_ok)}")
    print(f"  No binding found (date range?) : {len(no_bind)}")
    print("="*80)

    if alto_ok:
        print(f"\n  ✅ NEWSPAPERS WITH CONFIRMED ALTO TEXT ACCESS:\n")
        print(f"  {'Title':<40} {'ISSN':<12} {'Years':<15} {'Binding':<12}")
        print(f"  {'-'*40} {'-'*12} {'-'*15} {'-'*12}")
        for r in sorted(alto_ok, key=lambda x: x["available_from"]):
            years = f"{r['available_from']}–{r['available_to']}"
            print(f"  {r['title'][:40]:<40} {r['issn']:<12} {years:<15} {r['sample_binding']}")

    if no_bind:
        print(f"\n  ⚠️  NEWSPAPERS WHERE NO BINDING WAS FOUND (may need manual check):\n")
        for r in no_bind[:10]:
            print(f"    - {r['title']} ({r['issn']})  {r['available_from']}–{r['available_to']}")
        if len(no_bind) > 10:
            print(f"    ... and {len(no_bind)-10} more (see CSV)")

    print()


def save_csv(results: list[dict], filename: str) -> None:
    """Save full results to a CSV file."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  💾 Full results saved to: {filename}")


def show_example_queries() -> None:
    """Print example direct URLs for ALTO access."""
    print("\n" + "="*80)
    print("  EXAMPLE ALTO XML ACCESS URLS (manual test in browser or curl)")
    print("="*80)
    print("""
  # Fetch ALTO XML for page 1 of binding 1426186:
  GET https://digi.kansalliskirjasto.fi/sanomalehti/binding/1426186/page-00001.xml

  # Fetch METS manifest (lists all pages/files in an issue):
  GET https://digi.kansalliskirjasto.fi/sanomalehti/binding/1426186/mets

  # Fetch page image (JPG):
  GET https://digi.kansalliskirjasto.fi/sanomalehti/binding/1426186/image/1

  # Fetch full issue as PDF:
  GET https://digi.kansalliskirjasto.fi/sanomalehti/binding/1426186/pdf

  # List all newspaper titles (JSON):
  GET https://digi.kansalliskirjasto.fi/api/newspaper/titles?language=fi

  # OAI-PMH harvest metadata for a specific binding:
  GET https://digi.kansalliskirjasto.fi/interfaces/OAI-PMH?verb=GetRecord
      &metadataPrefix=marc21&identifier=oai:digi.kansalliskirjasto.fi:1426186

  # Search issues by keyword + date range (used internally by this script):
  GET https://digi.kansalliskirjasto.fi/api/search
      ?query=jää&startDate=1880-01-01&endDate=1880-12-31&type=NEWSPAPER
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore accessible Finnish newspaper ALTO XML files on digi.kansalliskirjasto.fi"
    )
    parser.add_argument("--start",  type=int, default=1820, help="Start year (default: 1820)")
    parser.add_argument("--end",    type=int, default=1940, help="End year   (default: 1940)")
    parser.add_argument("--probe",  type=int, default=1,    help="Issues to probe per newspaper (default: 1)")
    parser.add_argument("--lang",   type=str, default="fi", help="Language code: fi / sv / all (default: fi)")
    parser.add_argument("--out",    type=str, default="newspaper_coverage.csv", help="Output CSV filename")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip ALTO probing, just list titles in date range (fast mode)")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("  Finnish Newspaper ALTO Coverage Explorer")
    print(f"  Date range: {args.start} – {args.end}   |   Language: {args.lang}")
    print("="*80)

    # Step 1: Fetch all titles
    all_titles = fetch_all_titles(language=args.lang)

    # Step 2: Filter by year range
    print(f"\n[2/3] Filtering newspapers overlapping with {args.start}–{args.end}...")
    in_range = filter_by_year_range(all_titles, args.start, args.end)
    print(f"    ✓ {len(in_range)} newspapers overlap with your date range.")

    if not in_range:
        print("\n[!] No newspapers found in this range. Try broadening your years.")
        show_example_queries()
        return

    # Step 3: Probe for ALTO access (or fast-list mode)
    if args.no_probe:
        print(f"\n[3/3] Fast mode — skipping ALTO probing. Listing titles only.")
        results = []
        for t in in_range:
            results.append({
                "title":           t.get("title") or t.get("name") or "Unknown",
                "issn":            t.get("issn", ""),
                "language":        t.get("language", ""),
                "available_from":  t.get("_filtered_start", ""),
                "available_to":    t.get("_filtered_end", ""),
                "binding_found":   "not checked",
                "alto_accessible": "not checked",
            })
    else:
        print(f"\n[3/3] Probing {len(in_range)} newspapers for ALTO accessibility...")
        print("      (This may take a few minutes — rate-limited to be polite)")
        results = []
        iterator = tqdm(in_range) if HAS_TQDM else in_range

        for title in iterator:
            probe_year = title.get("_filtered_start", args.start)
            info = probe_newspaper(title, probe_year)
            results.append(info)

    # Output
    print_summary_table(results)
    save_csv(results, args.out)
    show_example_queries()

    print("  Done! ✅\n")


if __name__ == "__main__":
    main()