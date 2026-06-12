"""
extract_ice_files.py — batch ice-event extractor with river + date classification
Scans all .txt files in an input directory, keeps only those that contain at
least one ice-term match, and writes structured results to an output directory.

Usage:
    python extract_ice_files.py --dir prelim_cleaned/1900_1930 --out ice_hits
    python extract_ice_files.py --dir prelim_cleaned --out ice_hits
    python extract_ice_files.py --dir prelim_cleaned/1900_1930 --out ice_hits --quiet
    python extract_ice_files.py --dir prelim_cleaned/1900_1930 --out ice_hits --no-copy-txt

Output per matched file (in --out):
    <stem>.json   — structured matches with river + date fields
    <stem>.txt    — original text (unless --no-copy-txt)

Always written:
    _summary.json — aggregate stats for the whole run
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Import river/date classifier (must be in same directory)
try:
    from classify_river_date import classify_sentence_full, year_from_filename
except ImportError:
    raise SystemExit(
        "ERROR: classify_river_date.py not found.\n"
        "Place it in the same directory as this script."
    )

# ── ICE_RULES ─────────────────────────────────────────────────────────────────
ICE_RULES = {
    "freeze": [
        r"jokien jäätyminen",
        r"jäätyi",
        r"jäässä",
        r"jääkansi",
        r"joenjää",
        r"jokijää",
        r"kohvajää",
        r"suppojää",
        r"ahtojää",
        r"hyhmä",
        r"hyyde",
    ],
    "breakup": [
        r"jäänlähtö",
        r"jäidenlähtö",
        r"jään murtuminen",
        r"jokien avautuminen",
        r"avautui",
        r"jäät lähtivät",
        r"lähti jäistä",
    ],
    "ice_jam": [
        r"jääpato",
        r"hyydepato",
        r"jää ?tukos",
        r"jäälautta",
        r"jääpadon purkautuminen",
        r"jääpatojen purku",
    ],
    "flood": [
        r"tulva",
        r"kevättulva",
        r"talvitulva",
    ],
    "ice_conditions": [
        r"jäänpaksuus",
        r"jäätilanne",
        r"jään vahingot",
    ],
}

# ── Historical normaliser ─────────────────────────────────────────────────────
HISTORICAL_REPLACEMENTS = [
    (re.compile(r"w"),  "v"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"ck"), "kk"),
    (re.compile(r"â"),  "ä"),
    (re.compile(r"ô"),  "ö"),
    (re.compile(r"û"),  "u"),
]

def normalize_historical(text: str) -> str:
    result = text.lower()
    for pattern, replacement in HISTORICAL_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result

# ── Sentence splitting ────────────────────────────────────────────────────────
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]

# ── Ice matching ──────────────────────────────────────────────────────────────
def classify_sentence_ice(sentence: str) -> list[dict]:
    normalized = normalize_historical(sentence)
    matches = []
    for category, patterns in ICE_RULES.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                matches.append({"category": category, "pattern": pattern})
    return matches

# ── Per-file processor ────────────────────────────────────────────────────────
def process_file(file_path: Path) -> dict | None:
    text      = file_path.read_text(encoding="utf-8", errors="ignore")
    sentences = split_sentences(text)
    filename  = file_path.name

    # Year from filename as fallback for sentences with no date
    file_year = year_from_filename(filename)

    matched_sentences = []
    category_counts   = defaultdict(int)
    river_counts      = defaultdict(int)

    for i, sentence in enumerate(sentences):
        ice_matches = classify_sentence_ice(sentence)
        if not ice_matches:
            continue

        # River + date classification on matched sentence
        rd = classify_sentence_full(sentence, filename=filename)

        # If no year found in sentence, use file-level fallback
        year = rd["year"] or file_year

        cats = list({m["category"] for m in ice_matches})
        for cat in cats:
            category_counts[cat] += 1
        for river in rd["rivers"]:
            river_counts[river] += 1

        matched_sentences.append({
            "sentence_index": i,
            "sentence":       sentence,
            "normalized":     normalize_historical(sentence),
            "ice_categories": cats,
            "ice_patterns":   [m["pattern"] for m in ice_matches],
            "rivers":         rd["rivers"],
            "publishing_date": rd["publishing_date"],
            "year":           year,
            })

    if not matched_sentences:
        return None

    return {
        "source_file":       filename,
        "source_path":       str(file_path),
        "file_year":         file_year,
        "sentences_total":   len(sentences),
        "sentences_matched": len(matched_sentences),
        "categories_found":  dict(category_counts),
        "rivers_found":      dict(river_counts),
        "matches":           matched_sentences,
    }

# ── Batch runner ──────────────────────────────────────────────────────────────
def run_batch(
    input_dir:  Path,
    output_dir: Path,
    recursive:  bool = True,
    copy_txt:   bool = True,
    verbose:    bool = True,
) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)

    glob_pattern = "**/*.txt" if recursive else "*.txt"
    all_files    = sorted(input_dir.glob(glob_pattern))

    if not all_files:
        print(f"No .txt files found in {input_dir}")
        return {}

    total_files   = len(all_files)
    matched_files = 0
    skipped_files = 0
    total_sentences_scanned = 0
    total_sentences_matched = 0
    category_totals = defaultdict(int)
    river_totals    = defaultdict(int)
    file_results    = []

    print(f"Scanning {total_files} files in {input_dir} ...")
    print(f"Output  → {output_dir}\n")

    for i, file_path in enumerate(all_files, 1):
        if verbose:
            print(f"  [{i:>4}/{total_files}] {file_path.name}", end=" ... ")

        result = process_file(file_path)

        if result is None:
            skipped_files += 1
            if verbose:
                print("no match")
            continue

        matched_files += 1
        total_sentences_scanned += result["sentences_total"]
        total_sentences_matched += result["sentences_matched"]
        for cat, count in result["categories_found"].items():
            category_totals[cat] += count
        for river, count in result["rivers_found"].items():
            river_totals[river] += count

        # Save JSON
        json_out = output_dir / f"{file_path.stem}.json"
        json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Copy original .txt
        if copy_txt:
            txt_out = output_dir / file_path.name
            txt_out.write_text(
                file_path.read_text(encoding="utf-8", errors="ignore"),
                encoding="utf-8",
            )

        filename = file_path.name
        file_results.append({
        "file":              filename,
        "file_year":         result["file_year"],
        "sentences_matched": result["sentences_matched"],
        "sentences_total":   result["sentences_total"],
        "categories_found":  result["categories_found"],
        "rivers_found":      result["rivers_found"],
        })


        if verbose:
            cats   = ", ".join(result["categories_found"].keys())
            rivers = ", ".join(result["rivers_found"].keys()) or "river unknown"
            print(f"{result['sentences_matched']} match(es)  [{cats}]  {rivers}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "run_timestamp":      datetime.now().isoformat(timespec="seconds"),
        "input_directory":    str(input_dir),
        "output_directory":   str(output_dir),
        "files_scanned":      total_files,
        "files_with_matches": matched_files,
        "files_skipped":      skipped_files,
        "match_rate_pct":     round(matched_files / max(total_files, 1) * 100, 1),
        "sentences_scanned":  total_sentences_scanned,
        "sentences_matched":  total_sentences_matched,
        "category_totals":    dict(category_totals),
        "river_totals":       dict(river_totals),
        "matched_files":      file_results,
    }

    summary_path = output_dir / "_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    W = 72
    print("\n" + "=" * W)
    print("BATCH COMPLETE")
    print("=" * W)
    print(f"  Files scanned       : {total_files}")
    print(f"  Files with matches  : {matched_files}  ({summary['match_rate_pct']}%)")
    print(f"  Files skipped       : {skipped_files}")
    print(f"  Sentences scanned   : {total_sentences_scanned}")
    print(f"  Sentences matched   : {total_sentences_matched}")

    print(f"\n  Ice event categories:")
    for cat, count in sorted(category_totals.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 35)
        print(f"    {cat:<18} {count:>4}  {bar}")

    print(f"\n  Rivers identified:")
    for river, count in sorted(river_totals.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 35)
        print(f"    {river:<20} {count:>4}  {bar}")

    print(f"\n  Summary saved → {summary_path}")
    print("=" * W)

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Batch ice-event extractor with river + date classification"
    )
    parser.add_argument("--dir",  required=True, type=str, help="Input directory")
    parser.add_argument("--out",  required=True, type=str, help="Output directory")
    parser.add_argument(
        "--recursive", action="store_true", default=True,
        help="Search subdirectories (default: True)"
    )
    parser.add_argument(
        "--no-recursive", action="store_false", dest="recursive",
        help="Top-level directory only"
    )
    parser.add_argument(
        "--no-copy-txt", action="store_false", dest="copy_txt",
        help="Do not copy original .txt files to output"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file output"
    )
    args = parser.parse_args()

    run_batch(
        input_dir  = Path(args.dir),
        output_dir = Path(args.out),
        recursive  = args.recursive,
        copy_txt   = args.copy_txt,
        verbose    = not args.quiet,
    )


if __name__ == "__main__":
    main()
