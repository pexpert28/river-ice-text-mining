#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alto_parser.py
==============
Parse ALTO XML files from digi.kansalliskirjasto.fi into clean text
ready for NLP processing.

Handles the specifics of the real Digi ALTO v4 format:
  - ALTO v4 namespace (http://www.loc.gov/standards/alto/ns-v4#)
  - Hyphenated words across lines (SUBS_TYPE / SUBS_CONTENT)
  - Column layout reading order (sorted by VPOS/HPOS)
  - OCR noise cleaning for 19th/early 20th century Finnish/Swedish newspapers
  - Preserves TextBlock boundaries (useful for article segmentation)

USAGE
-----
  # As a script — parse and print:
  python alto_parser.py path/to/file.xml

  # Save cleaned text to file:
  python alto_parser.py path/to/file.xml --out cleaned.txt

  # Show per-block breakdown:
  python alto_parser.py path/to/file.xml --blocks

  # Parse a whole directory:
  python alto_parser.py path/to/dir/ --out-dir cleaned/

  # As a module:
  from alto_parser import parse_alto_file
  result = parse_alto_file("binding_page-00001.xml")
  print(result["full_text"])          # cleaned full page text
  for block in result["blocks"]:
      print(block["text"])            # per-column/article segment
"""

import argparse
import os
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from dataclasses import dataclass, field


# ── ALTO v4 namespace ─────────────────────────────────────────────────────────
# The real Digi files use this exact namespace.
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
NS = {"alto": ALTO_NS}

# Also handle files without namespace (older versions)
ALTO_NS_NONE = ""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    block_id:   str
    vpos:       int    # vertical position (top) — used for reading order
    hpos:       int    # horizontal position (left) — used for column order
    text:       str    # reconstructed clean text of this block
    word_count: int    = 0
    has_ocr_issues: bool = False


@dataclass
class ParsedPage:
    filename:    str
    full_text:   str           # entire page as one string
    blocks:      list          # list of TextBlock objects
    word_count:  int  = 0
    page_width:  int  = 0
    page_height: int  = 0
    ocr_software: str = ""


# ── Namespace-aware element tag helper ───────────────────────────────────────

def _tag(name: str, ns: str = ALTO_NS) -> str:
    """Build a namespace-qualified tag string."""
    if ns:
        return f"{{{ns}}}{name}"
    return name


def _detect_namespace(root) -> str:
    """
    Detect which namespace the file uses.
    Returns the namespace string or "" if none.
    """
    tag = root.tag
    if tag.startswith("{"):
        return tag[1:tag.index("}")]
    return ""


# ── Core word extraction ──────────────────────────────────────────────────────

def _extract_words_from_line(line_el, ns: str) -> list[str]:
    """
    Extract words from a TextLine element.

    Key rules:
      - String elements with SUBS_TYPE="HypPart1": use SUBS_CONTENT (full word)
        instead of CONTENT (which is just the pre-hyphen fragment)
      - String elements with SUBS_TYPE="HypPart2": skip (already captured above)
      - SP elements: add a space marker
      - Regular String elements: use CONTENT as-is
    """
    words = []
    for child in line_el:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if local == "String":
            subs_type    = child.get("SUBS_TYPE", "")
            subs_content = child.get("SUBS_CONTENT", "")
            content      = child.get("CONTENT", "")

            if subs_type == "HypPart1":
                # Use the full reconstructed word, not the fragment
                words.append(subs_content if subs_content else content)
            elif subs_type == "HypPart2":
                # Already captured via HypPart1 — skip
                pass
            else:
                if content.strip():
                    words.append(content)

        elif local == "SP":
            # Space element — marks word boundary, we handle via join
            pass

    return words


def _extract_text_from_block(block_el, ns: str) -> tuple[str, bool]:
    """
    Extract all text from a TextBlock, reconstructing lines and paragraphs.

    Returns (text, has_ocr_issues).
    has_ocr_issues is True if the block contains obvious OCR artifacts.
    """
    lines = []
    for line_el in block_el.iter(_tag("TextLine", ns)):
        words = _extract_words_from_line(line_el, ns)
        if words:
            lines.append(" ".join(words))

    text = "\n".join(lines)
    has_issues = _has_ocr_artifacts(text)
    return text, has_issues


# ── OCR cleaning ──────────────────────────────────────────────────────────────

# Patterns that indicate OCR noise rather than real text
_OCR_NOISE_PATTERNS = [
    r'^[^a-zA-ZåäöÅÄÖ]*$',      # No alphabetic characters at all
    r'[A-Z]{8,}',                 # Suspiciously long all-caps string
    r'(.)\1{4,}',                 # Character repeated 5+ times
    r'[|}{\\]{2,}',               # Multiple pipe/brace characters
]
_NOISE_RE = [re.compile(p) for p in _OCR_NOISE_PATTERNS]

# Separator lines (column rules, decorative lines)
_SEPARATOR_RE = re.compile(r'^[-=_*~+]{3,}$')

# Very short all-caps tokens that are likely OCR artifacts (but keep real abbrevs)
_GARBAGE_TOKEN_RE = re.compile(r'^[A-Z]{5,}[^a-zA-ZåäöÅÄÖ\s]')


def _has_ocr_artifacts(text: str) -> bool:
    """Heuristically detect if a text block has significant OCR problems."""
    if not text.strip():
        return False
    tokens = text.split()
    if not tokens:
        return False
    # Check ratio of garbage tokens
    garbage = sum(1 for t in tokens if any(p.search(t) for p in _NOISE_RE))
    return garbage / len(tokens) > 0.3


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR text from 19th/early 20th century Finnish/Swedish newspapers.

    What this does:
      1. Remove separator lines (----, ====)
      2. Remove lines that are pure punctuation/symbols with no letters
      3. Normalize whitespace
      4. Remove zero-width and control characters
      5. Normalize em-dashes and special quotes to ASCII equivalents
      6. Collapse multiple spaces

    What this does NOT do (intentionally):
      - Does not fix spelling (afled -> avled etc) — keep original for NLP
      - Does not remove lines with OCR errors — NLP models handle noise
      - Does not strip Finnish/Swedish special chars (å ä ö) — essential
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Skip pure separator lines
        if _SEPARATOR_RE.match(line.strip()):
            continue
        # Skip lines with no alphabetic content
        if line.strip() and not re.search(r'[a-zA-ZåäöÅÄÖ]', line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Normalize special typography
    text = text.replace("\u2014", " -- ")   # em dash
    text = text.replace("\u2013", " - ")    # en dash
    text = text.replace("\u201c", '"')       # left double quote
    text = text.replace("\u201d", '"')       # right double quote
    text = text.replace("\u201e", '"')       # low double quote
    text = text.replace("\u2018", "'")       # left single quote
    text = text.replace("\u2019", "'")       # right single quote
    text = text.replace("\u201a", "'")       # low single quote

    # Remove control characters and zero-width chars
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]', '', text)

    # Normalize whitespace within lines (but preserve line breaks)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


# ── Reading order sort ────────────────────────────────────────────────────────

def _sort_blocks_reading_order(blocks: list[TextBlock],
                                page_width: int) -> list[TextBlock]:
    """
    Sort TextBlocks into approximate reading order for a multi-column newspaper.

    Strategy: group blocks into column bands by horizontal position,
    then sort within each band by vertical position.
    This handles 2, 3, 4+ column layouts typical in 19th century Finnish papers.

    Falls back to simple top-to-bottom sort if page_width is unknown.
    """
    if not blocks or page_width == 0:
        return sorted(blocks, key=lambda b: (b.vpos, b.hpos))

    # Estimate number of columns based on block horizontal spread
    # Divide page into bands and sort vertically within each
    NUM_BANDS = 4   # assume up to 4 columns; works for 2-4 column layouts
    band_width = page_width / NUM_BANDS

    def sort_key(b: TextBlock):
        col_band = int(b.hpos / band_width)
        return (col_band, b.vpos)

    return sorted(blocks, key=sort_key)


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_alto_file(xml_path: str,
                    clean: bool = True,
                    sort_reading_order: bool = True) -> ParsedPage:
    """
    Parse an ALTO XML file and return a ParsedPage object.

    Parameters
    ----------
    xml_path            : path to the .xml file
    clean               : apply OCR text cleaning (default True)
    sort_reading_order  : sort TextBlocks into reading order (default True)

    Returns
    -------
    ParsedPage with:
      .full_text   : entire page as one string
      .blocks      : list of TextBlock objects (one per column/article segment)
      .word_count  : total word count
      .page_width/height : page dimensions in ALTO units (mm×10)
      .ocr_software : software used (e.g. "Transkribus")
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise ValueError(f"Could not parse XML: {xml_path} — {e}")

    root = tree.getroot()
    ns   = _detect_namespace(root)

    # ── Page dimensions ───────────────────────────────────────────────────────
    page_width  = 0
    page_height = 0
    page_el = root.find(f".//{_tag('Page', ns)}")
    if page_el is not None:
        page_width  = round(float(page_el.get("WIDTH",  0)))
        page_height = round(float(page_el.get("HEIGHT", 0)))

    # ── OCR software ──────────────────────────────────────────────────────────
    ocr_software = ""
    sw_el = root.find(f".//{_tag('softwareName', ns)}")
    if sw_el is not None and sw_el.text:
        ocr_software = sw_el.text.strip()

    # ── Extract TextBlocks ────────────────────────────────────────────────────
    blocks = []
    for block_el in root.iter(_tag("TextBlock", ns)):
        block_id = block_el.get("ID", "")
        vpos     = round(float(block_el.get("VPOS", 0)))
        hpos     = round(float(block_el.get("HPOS", 0)))

        raw_text, has_issues = _extract_text_from_block(block_el, ns)

        if clean:
            text = clean_ocr_text(raw_text)
        else:
            text = raw_text

        if not text.strip():
            continue   # skip empty blocks

        word_count = len(text.split())
        blocks.append(TextBlock(
            block_id       = block_id,
            vpos           = vpos,
            hpos           = hpos,
            text           = text,
            word_count     = word_count,
            has_ocr_issues = has_issues,
        ))

    # ── Sort into reading order ────────────────────────────────────────────────
    if sort_reading_order:
        blocks = _sort_blocks_reading_order(blocks, page_width)

    # ── Assemble full page text ───────────────────────────────────────────────
    full_text  = "\n\n".join(b.text for b in blocks)
    word_count = sum(b.word_count for b in blocks)

    return ParsedPage(
        filename     = path.name,
        full_text    = full_text,
        blocks       = blocks,
        word_count   = word_count,
        page_width   = page_width,
        page_height  = page_height,
        ocr_software = ocr_software,
    )


def parse_alto_directory(
    directory: str,
    clean: bool = True,
    sort_reading_order: bool = True,
    recursive: bool = True,
) -> dict[str, "ParsedPage"]:
    """
    Parse ALTO XML files under a directory.

    With recursive=True (default), descends into all subdirectories —
    matching the nested output of digi_fetcher (issn/year/binding/page.xml).

    Keys in the returned dict are relative paths from `directory`, e.g.:
      "0355-0842/1902/binding-001234/1902-03-11_binding-001234_page-00001.xml"

    This avoids key collisions when two bindings have same-named page files.
    """
    results: dict[str, "ParsedPage"] = {}
    root_path = Path(directory)
    pattern = "**/*.xml" if recursive else "*.xml"
    xml_files = sorted(root_path.glob(pattern))

    if not xml_files:
        print(f"[INFO] No .xml files found in {directory} (recursive={recursive})")
        return results

    for xml_path in xml_files:
        rel_key = str(xml_path.relative_to(root_path))
        try:
            page = parse_alto_file(str(xml_path), clean, sort_reading_order)
            results[rel_key] = page
        except Exception as e:
            print(f"[WARN] Skipping {rel_key}: {e}")

    return results


# ── NLP-ready output ──────────────────────────────────────────────────────────

def to_sentences(page: ParsedPage) -> list[str]:
    """
    Split a ParsedPage into sentences using simple punctuation rules.
    Works for Finnish and Swedish without needing a language model.
    Returns a flat list of sentence strings.
    """
    # Join all text, normalize newlines
    text = page.full_text.replace("\n", " ")
    # Split on sentence-ending punctuation followed by space + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÅÄÖ])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def to_keyword_windows(page: ParsedPage,
                        keywords: list[str],
                        window: int = 200) -> list[dict]:
    """
    Find all occurrences of keywords in the page text and return
    surrounding context windows. Used for ice event extraction.

    Parameters
    ----------
    keywords : list of Finnish/Swedish ice-related keywords
               e.g. ["jää", "jäätyi", "jäänlähtö", "isläggning", "islossning"]
    window   : characters of context around each match

    Returns list of dicts with: keyword, match_start, snippet, block_id
    """
    hits = []
    text = page.full_text.lower()
    original = page.full_text

    for kw in keywords:
        kw_lower = kw.lower()
        pos = 0
        while True:
            idx = text.find(kw_lower, pos)
            if idx == -1:
                break
            start   = max(0, idx - window)
            end     = min(len(original), idx + len(kw) + window)
            snippet = original[start:end].replace("\n", " ").strip()

            # Find which block this hit belongs to
            block_id = ""
            char_count = 0
            for block in page.blocks:
                char_count += len(block.text) + 2   # +2 for \n\n
                if char_count >= idx:
                    block_id = block.block_id
                    break

            hits.append({
                "keyword":     kw,
                "position":    idx,
                "snippet":     snippet,
                "block_id":    block_id,
                "filename":    page.filename,
            })
            pos = idx + 1

    return hits


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parse ALTO XML from digi.kansalliskirjasto.fi into clean NLP-ready text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("path",
                   help="Path to .xml file or directory of .xml files")
    p.add_argument("--out",      default=None,
                   help="Save full page text to this file")
    p.add_argument("--out-dir",  default=None,
                   help="Save one .txt per XML file into this directory")
    p.add_argument("--blocks",   action="store_true",
                   help="Print per-TextBlock breakdown")
    p.add_argument("--no-clean", action="store_true",
                   help="Skip OCR cleaning (output raw extracted text)")
    p.add_argument("--keywords", default=None,
                   help='Comma-separated keywords to search for, e.g. "jää,jäätyi,jäänlähtö"')
    p.add_argument("--window",   type=int, default=200,
                   help="Context window size in chars for keyword search (default: 200)")
    return p.parse_args()


def main():
    args = parse_args()
    clean = not args.no_clean
    target = Path(args.path)

    if target.is_dir():
        pages = parse_alto_directory(str(target), clean=clean)
        if not pages:
            print("No XML files found.")
            return
        for fname, page in pages.items():
            print(f"\n{'='*60}")
            print(f"  {fname}  |  {page.word_count} words  |  {len(page.blocks)} blocks  |  OCR: {page.ocr_software}")
            print(f"{'='*60}")
            print(page.full_text[:500] + ("..." if len(page.full_text) > 500 else ""))
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname, page in pages.items():
                out_path = out_dir / fname.replace(".xml", ".txt")
                out_path.write_text(page.full_text, encoding="utf-8")
            print(f"\nSaved {len(pages)} text files to: {args.out_dir}")
        return

    # Single file
    page = parse_alto_file(str(target), clean=clean)

    print(f"\n{'='*60}")
    print(f"  File     : {page.filename}")
    print(f"  Size     : {page.page_width} × {page.page_height} units")
    print(f"  OCR by   : {page.ocr_software or 'unknown'}")
    print(f"  Words    : {page.word_count}")
    print(f"  Blocks   : {len(page.blocks)}")
    print(f"{'='*60}\n")

    if args.blocks:
        for i, block in enumerate(page.blocks, 1):
            flag = " ⚠ OCR issues" if block.has_ocr_issues else ""
            print(f"  Block {i:>3}  [{block.block_id}]  pos=({block.hpos},{block.vpos})  {block.word_count} words{flag}")
            print(f"           {block.text[:120].replace(chr(10), ' ')}...")
            print()
    else:
        preview = page.full_text[:1000]
        print(preview)
        if len(page.full_text) > 1000:
            print(f"\n... ({page.word_count} words total, showing first 1000 chars)")

    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",")]
        hits = to_keyword_windows(page, keywords, window=args.window)
        if hits:
            print(f"\n{'='*60}")
            print(f"  Keyword hits ({len(hits)} found):")
            print(f"{'='*60}")
            for h in hits:
                print(f"\n  [{h['keyword']}]  pos={h['position']}  block={h['block_id']}")
                print(f"  ...{h['snippet']}...")
        else:
            print(f"\n  No hits found for: {keywords}")

    if args.out:
        Path(args.out).write_text(page.full_text, encoding="utf-8")
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
