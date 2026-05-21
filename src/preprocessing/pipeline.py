#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Run the full text extraction pipeline on one or more ALTO XML files:

  alto_parser  →  article_segmenter  →  text_normalizer  →  clean text

Each step builds on the previous:
  1. alto_parser       : extracts raw text from ALTO XML (handles namespace,
                         hyphenation, column layout)
  2. article_segmenter : groups TextBlocks into coherent articles using
                         column sorting + TextTiling boundary detection
  3. text_normalizer   : converts old Finnish spelling to modern (w→v etc.)
                         and rejoins hyphenated line-breaks

USAGE
-----
  # Process a single ALTO XML file, print results:
  python pipeline.py path/to/binding_page-00001.xml

  # Save output to a text file:
  python pipeline.py page.xml --out cleaned.txt

  # Process a whole directory of ALTO XML files:
  python pipeline.py data/raw/issn/1880/ --out-dir data/processed/1880/

  # Only return ice-relevant articles:
  python pipeline.py page.xml --ice-only

  # Show all articles with headers:
  python pipeline.py page.xml --verbose

AS A MODULE
-----------
  from pipeline import run_pipeline, run_pipeline_dir

  # Single file — returns list of ProcessedArticle
  articles = run_pipeline("binding_page-00001.xml")
  for art in articles:
      print(art.header)
      print(art.clean_text)

  # Directory — returns dict {filename: [ProcessedArticle]}
  results = run_pipeline_dir("data/raw/issn/1880/", ice_only=True)
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field


# ── Output data structure ─────────────────────────────────────────────────────

@dataclass
class ProcessedArticle:
    """
    One article after the full pipeline.

    Attributes
    ----------
    header           : detected article title (empty string if none)
    clean_text       : normalized text ready for NLP
    raw_text         : text before normalization (for comparison/debugging)
    has_ice_keywords : True if ice-event keywords found in the text
    n_blocks         : number of TextBlocks that make up this article
    boundary_type    : how this segment boundary was detected
    source_file      : filename of the source ALTO XML
    """
    header:           str
    clean_text:       str
    raw_text:         str
    has_ice_keywords: bool
    n_blocks:         int
    boundary_type:    str
    source_file:      str = ""


# ── Core pipeline function ────────────────────────────────────────────────────

def run_pipeline(xml_path: str,
                 ice_only:        bool  = False,
                 n_columns:       int   = 0,
                 depth_threshold: float = 0.10,
                 window:          int   = 3,
                 min_block_words: int   = 2) -> list[ProcessedArticle]:
    """
    Run the full pipeline on a single ALTO XML file.

    Parameters
    ----------
    xml_path         : path to the ALTO XML file
    ice_only         : if True, return only ice-event relevant articles
    n_columns        : number of newspaper columns (0 = auto-detect)
    depth_threshold  : TextTiling sensitivity (lower = more boundaries)
    window           : TextTiling window size in blocks
    min_block_words  : minimum words per block to include

    Returns
    -------
    List of ProcessedArticle objects, one per detected article segment.
    """
    src = Path(xml_path)

    # ── Step 1: Parse ALTO XML ────────────────────────────────────────────────
    try:
        from alto_parser import parse_alto_file
    except ImportError:
        _die("alto_parser.py not found. Make sure it is in the same directory.")

    try:
        page = parse_alto_file(str(xml_path))
    except FileNotFoundError:
        _die(f"File not found: {xml_path}")
    except Exception as e:
        _die(f"Failed to parse {xml_path}: {e}")

    if not page.blocks:
        return []

    # ── Step 2: Segment into articles ─────────────────────────────────────────
    try:
        from article_segmenter import ArticleSegmenter
    except ImportError:
        _die("article_segmenter.py not found. Make sure it is in the same directory.")

    segmenter = ArticleSegmenter(
        k               = window,
        depth_threshold = depth_threshold,
        n_columns       = n_columns,
        min_block_words = min_block_words,
    )
    segments = segmenter.segment(page)

    # ── Step 3: Normalize each segment ────────────────────────────────────────
    try:
        from text_normalizer import normalize_text
    except ImportError:
        _die("text_normalizer.py not found. Make sure it is in the same directory.")

    results = []
    for seg in segments:
        if ice_only and not seg.has_ice_keywords:
            continue

        norm = normalize_text(seg.full_text)

        # Re-extract header from normalized text so it has modern spelling
        # (seg.header comes from raw block text before normalization)
        from article_segmenter import is_header as _is_header
        clean_header = ""
        for line in norm.normalized_text.splitlines()[:3]:
            if _is_header(line):
                clean_header = line.strip().rstrip('.')
                break

        results.append(ProcessedArticle(
            header           = clean_header,
            clean_text       = norm.normalized_text,
            raw_text         = seg.full_text,
            has_ice_keywords = seg.has_ice_keywords,
            n_blocks         = len(seg.blocks),
            boundary_type    = seg.boundary_type,
            source_file      = src.name,
        ))

    return results


def run_pipeline_dir(input_dir:  str,
                     output_dir: str  = None,
                     ice_only:   bool = False,
                     **kwargs) -> dict[str, list[ProcessedArticle]]:
    """
    Run the pipeline on every ALTO XML file in a directory.

    Parameters
    ----------
    input_dir  : directory containing .xml files
    output_dir : if given, saves one .txt per file here
    ice_only   : only keep ice-relevant articles
    **kwargs   : passed through to run_pipeline()

    Returns
    -------
    Dict mapping filename → list of ProcessedArticle.
    """
    xml_files = sorted(Path(input_dir).glob("*.xml"))
    if not xml_files:
        print(f"[INFO] No .xml files found in {input_dir}")
        return {}

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    for xml_path in xml_files:
        articles = run_pipeline(str(xml_path), ice_only=ice_only, **kwargs)
        all_results[xml_path.name] = articles

        n_ice = sum(1 for a in articles if a.has_ice_keywords)
        print(f"  {xml_path.name:<40} {len(articles):>3} articles"
              f"  {n_ice:>2} ice-relevant")

        if output_dir and articles:
            _save_articles(articles, xml_path.name, output_dir)

    return all_results


# ── Save helpers ──────────────────────────────────────────────────────────────

def _save_articles(articles:   list[ProcessedArticle],
                   source_name: str,
                   output_dir:  str):
    """
    Save processed articles to a .txt file.

    Format: articles separated by a blank line.
    The header is already the first line of clean_text — no separate # line needed.
    """
    lines = []
    for art in articles:
        lines.append(art.clean_text.strip())
        lines.append("")          # blank line between articles

    out_path = Path(output_dir) / source_name.replace(".xml", ".txt")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def articles_to_plain_text(articles: list[ProcessedArticle],
                            separator: str = "\n\n---\n\n") -> str:
    """
    Join a list of ProcessedArticle objects into a single plain text string.

    Parameters
    ----------
    articles  : output of run_pipeline()
    separator : string placed between articles

    Returns
    -------
    Single string with all article texts joined.
    """
    parts = []
    for art in articles:
        parts.append(art.clean_text.strip())
    return separator.join(parts)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _die(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def _print_report(articles: list[ProcessedArticle],
                  source:   str,
                  verbose:  bool = False):
    """Print a human-readable pipeline report."""
    n_ice = sum(1 for a in articles if a.has_ice_keywords)

    print(f"\n{'='*64}")
    print(f"  {source}")
    print(f"  {len(articles)} article(s)  |  {n_ice} ice-relevant")
    print(f"{'='*64}\n")

    for i, art in enumerate(articles, 1):
        ice_flag  = "  ❄ ICE" if art.has_ice_keywords else ""
        hdr_str   = f'"{art.header}"' if art.header else "(no header)"
        bnd_str   = f"[{art.boundary_type}]" if art.boundary_type else ""

        print(f"  Article {i:>3}  {hdr_str:<38} "
              f"{art.n_blocks} blocks  {bnd_str}{ice_flag}")

        if verbose:
            for line in art.clean_text.splitlines():
                print(f"    {line}")
        else:
            preview = art.clean_text[:200].replace("\n", " ")
            print(f"    {preview}")
            if len(art.clean_text) > 200:
                print(f"    ... ({len(art.clean_text)} chars)")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Full text extraction pipeline: ALTO XML → clean normalized text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("xml_file", nargs="?",  help="Single ALTO XML file")
    src.add_argument("--dir",                help="Directory of ALTO XML files")

    p.add_argument("--out",          default=None,
                   help="Output .txt file (for single file mode)")
    p.add_argument("--out-dir",      default=None,
                   help="Output directory (for --dir mode)")
    p.add_argument("--ice-only",     action="store_true",
                   help="Return only ice-event relevant articles")
    p.add_argument("--verbose",      action="store_true",
                   help="Print full text of each article")
    p.add_argument("--n-columns",    type=int,   default=0,
                   help="Number of newspaper columns (0 = auto-detect)")
    p.add_argument("--threshold",    type=float, default=0.10,
                   help="TextTiling depth threshold (default: 0.10)")
    p.add_argument("--window",       type=int,   default=3,
                   help="TextTiling window size in blocks (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()

    kwargs = dict(
        ice_only        = args.ice_only,
        n_columns       = args.n_columns,
        depth_threshold = args.threshold,
        window          = args.window,
    )

    if args.dir:
        out_dir = args.out_dir or str(Path(args.dir).parent / "processed")
        print(f"Processing directory: {args.dir}")
        print(f"Output directory:     {out_dir}\n")
        results = run_pipeline_dir(args.dir, output_dir=out_dir, **kwargs)

        total_arts = sum(len(v) for v in results.values())
        total_ice  = sum(sum(1 for a in v if a.has_ice_keywords)
                         for v in results.values())
        print(f"\nTotal: {len(results)} files  |  "
              f"{total_arts} articles  |  {total_ice} ice-relevant")

    else:
        articles = run_pipeline(args.xml_file, **kwargs)
        _print_report(articles, Path(args.xml_file).name, verbose=args.verbose)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for art in articles:
                lines.append(art.clean_text.strip())
                lines.append("")
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Saved → {args.out}")
        elif not args.verbose:
            print("  Tip: add --verbose to see full text, "
                  "--out to save to file.")


if __name__ == "__main__":
    main()