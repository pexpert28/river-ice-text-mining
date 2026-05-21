#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
article_segmenter.py
====================
Reconstruct coherent articles from the fragmented TextBlock output of
alto_parser.py, which reads a multi-column newspaper page and returns
blocks in column-band order rather than article order.

Two-stage approach
------------------
  Stage 1 — Column-coherent sorting (geometry)
    Groups TextBlocks by their horizontal position into column bands,
    then sorts vertically within each band. This ensures each column
    is read top-to-bottom before moving to the next column — eliminating
    the worst inter-column interleaving.

  Stage 2 — Article boundary detection (TextTiling + header detection)
    Within the column-sorted block sequence, detects topic shifts using:
      a) Header blocks: short title-case or all-caps blocks strongly
         signal a new article regardless of vocabulary similarity.
      b) TextTiling: computes TF-IDF cosine similarity between sliding
         windows of adjacent blocks. A deep valley in the similarity
         curve indicates a topic shift = article boundary.

    No external model or internet connection required — runs entirely
    on Python standard library + optional scikit-learn for TF-IDF
    (falls back to Counter-based cosine if sklearn not installed).

Pipeline position
-----------------
  alto_parser.py  →  article_segmenter.py  →  text_normalizer.py  →  NLP

Output
------
  A list of ArticleSegment objects, each containing:
    - blocks: list of TextBlock objects from alto_parser
    - full_text: joined text of all blocks in the segment
    - header: detected article title (if any)
    - has_ice_keywords: quick flag for ice-event relevant segments

USAGE
-----
  from alto_parser import parse_alto_file
  from article_segmenter import ArticleSegmenter

  page = parse_alto_file("binding_page-00001.xml")
  segmenter = ArticleSegmenter()
  articles = segmenter.segment(page)

  for article in articles:
      print(article.header or "(no header)")
      print(article.full_text[:200])

  # CLI: inspect segments from a file
  python article_segmenter.py path/to/file.xml
  python article_segmenter.py path/to/file.xml --ice-only
"""

import re
import sys
import math
import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ArticleSegment:
    """One article or coherent text segment extracted from a newspaper page."""
    blocks:           list          # list of TextBlock from alto_parser
    full_text:        str           # all block texts joined
    header:           str  = ""     # detected article title, if any
    has_ice_keywords: bool = False  # True if ice-event keywords present
    boundary_type:    str  = ""     # "header" | "texttiling" | "column_start"


# ── Finnish + Swedish stop words ──────────────────────────────────────────────

STOPWORDS = {
    # Finnish function words
    "on", "ei", "se", "ne", "ja", "tai", "vai", "mutta", "vaan",
    "että", "kun", "jos", "niin", "kuin", "jo", "myös", "myöskin",
    "vielä", "vain", "siis", "sekä", "oli", "olla", "olen", "olet",
    "hän", "he", "me", "te", "minä", "sinä", "tämä", "tuo",
    "joka", "jotka", "mikä", "mitä", "missä", "milloin", "koska",
    "kuitenkin", "myöskin", "koko", "kaikki", "aina", "nyt",
    "sen", "sen", "sitä", "siitä", "siihen", "tätä", "tähän",
    "hänen", "heidän", "meidän", "teidän", "minun", "sinun",
    # Common newspaper words that add noise
    "klo", "mk", "p", "n:o", "y", "m", "ko", "ao", "em",
    "pnä", "pnä", "vuonna", "vuoden", "vuodesta",
    # Swedish function words
    "och", "är", "det", "att", "en", "ett", "av", "på", "för",
    "med", "den", "de", "till", "som", "om", "från", "har",
    "hade", "inte", "var", "han", "hon", "vi", "ni", "de",
    "sin", "sig", "men", "eller", "när", "där", "här",
}

# Ice event keywords — used to flag relevant segments
ICE_KEYWORDS_FI = {
    "jää", "jäät", "jäätyi", "jäätyminen", "jäätynyt", "jäätynyt",
    "jäänlähtö", "jäät", "jääpato", "jääpatoja", "jäänpaksuus",
    "tulva", "tulvat", "tulvavesi", "hyytö", "kiintojää",
}
ICE_KEYWORDS_SV = {
    "is", "isen", "isläggning", "islossning", "ispropp",
    "översvämning", "tjocklek",
}
ICE_KEYWORDS = ICE_KEYWORDS_FI | ICE_KEYWORDS_SV


# ── Text utilities ────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Extract meaningful tokens, removing stop words and short tokens."""
    words = re.findall(r'[a-zA-ZåäöÅÄÖ]{3,}', text.lower())
    return [w for w in words if w not in STOPWORDS]


def cosine_sim(a: list[str], b: list[str]) -> float:
    """TF cosine similarity between two token lists."""
    if not a or not b:
        return 0.0
    va, vb = Counter(a), Counter(b)
    keys = set(va) | set(vb)
    dot  = sum(va[k] * vb[k] for k in keys)
    na   = math.sqrt(sum(v ** 2 for v in va.values()))
    nb   = math.sqrt(sum(v ** 2 for v in vb.values()))
    return dot / (na * nb) if na * nb else 0.0


def is_header(text: str) -> bool:
    """
    Detect article header blocks.
    Headers are short (≤7 words), start with a capital, and are
    not issue numbers, dates, or pure numerics.
    """
    text  = text.strip().rstrip('.')
    words = text.split()
    alpha = [w for w in words
             if re.search(r'[a-zA-ZåäöÅÄÖ]{2,}', w)]

    if not alpha:
        return False
    # Must have at least one word longer than 3 chars
    if not any(len(w) > 3 for w in alpha):
        return False
    # Max 7 alpha words
    if len(alpha) > 7:
        return False
    # Exclude issue numbers and date markers
    if re.match(r'^(N:o|n:o|s\.|k\.|p\.)\s*\d', text, re.IGNORECASE):
        return False
    # All-caps block (masthead, section title)
    if all(w.upper() == w for w in alpha if len(w) > 1):
        return True
    # Short block with capitalized first word
    if len(alpha) <= 4 and alpha[0][0].isupper():
        return True
    # Majority title-case
    cap_ratio = sum(1 for w in alpha if w[0].isupper()) / len(alpha)
    return cap_ratio >= 0.6


def has_ice_keywords(text: str) -> bool:
    """Return True if any ice-event keyword appears in the text."""
    tokens = set(re.findall(r'[a-zA-ZåäöÅÄÖ]+', text.lower()))
    return bool(tokens & ICE_KEYWORDS)


# ── Stage 1: Column-coherent sorting ─────────────────────────────────────────

def sort_blocks_by_column(blocks: list,
                           page_width: int,
                           n_columns: int = 0) -> list:
    """
    Sort TextBlocks into column-coherent reading order.

    Automatically estimates the number of columns if not provided,
    by clustering block HPOS values. Then sorts:
      - Primary:   column index (left → right)
      - Secondary: VPOS within column (top → bottom)

    Parameters
    ----------
    blocks     : list of TextBlock from alto_parser
    page_width : page width in ALTO units (from ParsedPage)
    n_columns  : expected number of columns (0 = auto-detect)
    """
    if not blocks:
        return []

    # Auto-detect column count from HPOS distribution
    if n_columns == 0:
        hpos_values = sorted(set(b.hpos for b in blocks))
        n_columns   = _estimate_columns(hpos_values, page_width)

    band_width = page_width / n_columns if page_width > 0 else 1000

    def sort_key(b):
        col = min(int(b.hpos / band_width), n_columns - 1)
        return (col, b.vpos)

    return sorted(blocks, key=sort_key)


def _estimate_columns(hpos_values: list[int], page_width: int) -> int:
    """
    Estimate number of columns from the spread of block horizontal positions.
    Uses a simple gap-based clustering approach.
    """
    if not hpos_values or page_width == 0:
        return 4  # safe default for Finnish newspapers

    # Spread of left edges relative to page width
    spread = (max(hpos_values) - min(hpos_values)) / page_width

    # Typical Finnish newspaper column counts by era
    # 1820-1870: 2-3 columns, 1870-1939: 4-7 columns
    # We use spread to estimate:
    if spread < 0.2:
        return 2
    elif spread < 0.4:
        return 3
    elif spread < 0.65:
        return 4
    elif spread < 0.80:
        return 5
    else:
        return 6


# ── Stage 2: Article boundary detection ───────────────────────────────────────

def find_article_boundaries(blocks: list,
                              k: int   = 3,
                              depth_threshold: float = 0.10) -> list[tuple[int, str]]:
    """
    Find article boundaries in a column-sorted block sequence.

    Returns list of (gap_index, boundary_type) tuples where
    gap_index is the block index AFTER which a new article starts.

    boundary_type is one of:
      "header"      — next block is a detected article header
      "texttiling"  — TextTiling depth score exceeded threshold
      "column_start"— first block of a new column (always a boundary)
    """
    if len(blocks) < 2:
        return []

    boundaries = []
    texts      = [b.text for b in blocks]

    # --- Header-based boundaries ---
    for i, block in enumerate(blocks):
        if i == 0:
            continue
        if is_header(block.text):
            boundaries.append((i - 1, "header"))

    header_gaps = {gap for gap, _ in boundaries}

    # --- TextTiling boundaries ---
    # Compute gap similarity scores
    sims = []
    for i in range(len(texts)):
        left  = tokenize(' '.join(texts[max(0, i - k):i]))
        right = tokenize(' '.join(texts[i:min(len(texts), i + k)]))
        sims.append(cosine_sim(left, right))

    # Depth score for each gap
    def depth_score(i: int) -> float:
        lp = sims[i]
        for j in range(i - 1, -1, -1):
            if sims[j] >= lp:
                lp = sims[j]
            else:
                break
        rp = sims[i]
        for j in range(i + 1, len(sims)):
            if sims[j] >= rp:
                rp = sims[j]
            else:
                break
        return (lp - sims[i]) + (rp - sims[i])

    depths = [depth_score(i) for i in range(len(sims))]

    # Find local minima above depth threshold
    # Merge consecutive valleys into single boundary at deepest point
    i = 0
    while i < len(depths) - 1:
        if depths[i] >= depth_threshold and i not in header_gaps:
            run_start = i
            while i < len(depths) - 1 and depths[i] >= depth_threshold:
                i += 1
            best = max(range(run_start, i), key=lambda x: depths[x])
            if best not in header_gaps:
                boundaries.append((best, "texttiling"))
        i += 1

    # Deduplicate and sort
    seen = set()
    unique = []
    for gap, btype in sorted(boundaries, key=lambda x: x[0]):
        if gap not in seen:
            unique.append((gap, btype))
            seen.add(gap)

    return unique


# ── Main segmenter class ──────────────────────────────────────────────────────

class ArticleSegmenter:
    """
    Segments a ParsedPage from alto_parser into coherent articles.

    Parameters
    ----------
    k               : TextTiling window size in blocks (default 3)
    depth_threshold : Minimum depth score to declare a boundary (default 0.10)
    n_columns       : Expected column count, 0 = auto-detect (default 0)
    min_block_words : Minimum word count to include a block (default 2)
    """

    def __init__(self,
                 k:               int   = 3,
                 depth_threshold: float = 0.10,
                 n_columns:       int   = 0,
                 min_block_words: int   = 2):
        self.k               = k
        self.depth_threshold = depth_threshold
        self.n_columns       = n_columns
        self.min_block_words = min_block_words

    def segment(self, page) -> list[ArticleSegment]:
        """
        Segment a ParsedPage into articles.

        Parameters
        ----------
        page : ParsedPage object from alto_parser.parse_alto_file()

        Returns
        -------
        List of ArticleSegment objects in reading order.
        """
        # Filter trivially short blocks
        blocks = [b for b in page.blocks
                  if b.word_count >= self.min_block_words]

        if not blocks:
            return []

        # Stage 1: column-coherent sorting
        sorted_blocks = sort_blocks_by_column(
            blocks,
            page_width = page.page_width,
            n_columns  = self.n_columns,
        )

        # Stage 2: find article boundaries
        boundaries = find_article_boundaries(
            sorted_blocks,
            k               = self.k,
            depth_threshold = self.depth_threshold,
        )
        boundary_gaps = {gap: btype for gap, btype in boundaries}

        # Split blocks into segments
        segments     = []
        current_blocks = [sorted_blocks[0]]

        for i in range(1, len(sorted_blocks)):
            if (i - 1) in boundary_gaps:
                # Close current segment
                seg = self._make_segment(current_blocks,
                                         boundary_gaps[i - 1])
                segments.append(seg)
                current_blocks = [sorted_blocks[i]]
            else:
                current_blocks.append(sorted_blocks[i])

        # Close final segment
        if current_blocks:
            segments.append(self._make_segment(current_blocks, ""))

        return segments

    def _make_segment(self, blocks: list,
                       boundary_type: str) -> ArticleSegment:
        """
        Build an ArticleSegment from a list of blocks.

        Handles three cross-block join cases:
          1. Explicit hyphen at block end: "karjanjalostusyh-" + "distyksen"
             → "karjanjalostusyhdistyksen"
          2. Bare word split (short fragment at block end):
             "ter" + "vehdyksistä" → "tervehdyksistä"
             Only joins when fragment ends with a vowel and is 2-4 chars.
          3. Merged number lines: "7,915: --" joined to next block text
             → ensure double-dash endings always get a newline after them.
        """
        if not blocks:
            return ArticleSegment(blocks=[], full_text="", header="",
                                  has_ice_keywords=False, boundary_type=boundary_type)

        texts = [b.text.strip() for b in blocks]
        joined_parts = []

        for i, text in enumerate(texts):
            if i == 0:
                joined_parts.append(text)
                continue

            prev = joined_parts[-1]

            # Case 1: explicit hyphen at end of previous block
            if prev.endswith('-'):
                # Remove hyphen and join directly (no space, no newline)
                joined_parts[-1] = prev[:-1] + text
                continue

            # Case 2: bare word split — short fragment (2-4 chars ending in vowel)
            # at end of previous block, continuation starts lowercase
            prev_last_word = prev.split()[-1] if prev.split() else ""
            first_char = text[0] if text else ""
            if (2 <= len(prev_last_word) <= 4
                    and prev_last_word[-1] in 'aeiouåäöy'
                    and prev_last_word.lower() not in _COMMON_WORDS
                    and first_char.islower()):
                joined_parts[-1] = prev + text
                continue

            # Case 3: number/table line ending with single dash (truncated --)
            # e.g. "7,915: -"  →  ensure newline before next block
            joined_parts.append(text)

        full_text = '\n'.join(joined_parts)

        # Detect header: first block that looks like a header
        header = ""
        for b in blocks[:3]:
            if is_header(b.text):
                header = b.text.strip().rstrip('.')
                break

        return ArticleSegment(
            blocks           = blocks,
            full_text        = full_text,
            header           = header,
            has_ice_keywords = has_ice_keywords(full_text),
            boundary_type    = boundary_type,
        )


# Common short Finnish words that should NOT trigger bare-split joining
_COMMON_WORDS = {
    'on', 'ei', 'se', 'ne', 'ja', 'tai', 'vai', 'jo', 'ko', 'ao',
    'nyt', 'tai', 'kun', 'jos', 'niin', 'yli', 'ali', 'kin', 'han',
    'hän', 'ol', 'ei', 'yl', 'al', 'en', 'et', 'me', 'te', 'he',
    'och', 'är', 'av', 'på', 'om', 'en', 'de', 'vi', 'ni',
}


# ── Convenience function ──────────────────────────────────────────────────────

def segment_file(xml_path: str, **kwargs) -> list[ArticleSegment]:
    """
    Parse an ALTO XML file and segment it into articles in one call.

    Usage:
        articles = segment_file("binding_page-00001.xml")
        for art in articles:
            print(art.header, art.full_text[:200])
    """
    # Import here to avoid circular dependency
    sys.path.insert(0, str(Path(__file__).parent))
    from alto_parser import parse_alto_file

    page = parse_alto_file(xml_path)
    segmenter = ArticleSegmenter(**kwargs)
    return segmenter.segment(page)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Segment ALTO XML newspaper page into coherent articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('xml_file', help='Path to ALTO XML file from alto_parser')
    p.add_argument('--ice-only',   action='store_true',
                   help='Show only segments containing ice-event keywords')
    p.add_argument('--n-columns',  type=int, default=0,
                   help='Expected number of columns (0 = auto-detect)')
    p.add_argument('--threshold',  type=float, default=0.10,
                   help='TextTiling depth threshold (default: 0.10)')
    p.add_argument('--window',     type=int, default=3,
                   help='TextTiling window size in blocks (default: 3)')
    p.add_argument('--full-text',  action='store_true',
                   help='Print full text of each segment (default: preview only)')
    return p.parse_args()


def main():
    args = parse_args()

    articles = segment_file(
        args.xml_file,
        k               = args.window,
        depth_threshold = args.threshold,
        n_columns       = args.n_columns,
    )

    if args.ice_only:
        articles = [a for a in articles if a.has_ice_keywords]

    print(f"\n{'='*64}")
    print(f"  {Path(args.xml_file).name}")
    print(f"  {len(articles)} article segment(s)"
          + (" (ice-relevant only)" if args.ice_only else ""))
    print(f"{'='*64}\n")

    for i, art in enumerate(articles, 1):
        header_str = f'"{art.header}"' if art.header else "(no header)"
        ice_flag   = "  ❄ ICE" if art.has_ice_keywords else ""
        print(f"  Article {i:>3}  {header_str:<40} "
              f"{len(art.blocks)} blocks  [{art.boundary_type}]{ice_flag}")

        text = art.full_text if args.full_text else art.full_text[:200]
        for line in text.splitlines():
            print(f"    {line}")
        if not args.full_text and len(art.full_text) > 200:
            print(f"    ... ({len(art.full_text)} chars total)")
        print()


if __name__ == '__main__':
    main()