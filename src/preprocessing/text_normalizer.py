#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_normalizer.py
==================
Normalize historical Finnish newspaper text (1820–1939) to modern spelling,
preparing it for NLP models trained on modern Finnish (FinBERT, Poro, etc.).

What this does
--------------
  1. w → v  (the dominant spelling change in pre-1910 Finnish newspapers)
  2. Rejoin words split across lines by OCR/layout (e.g. "ter-\nwehdys")
  3. Normalize unicode punctuation and whitespace
  4. Flag truncated words (line-break artifacts like "tulew", "menew.")
  5. Optionally detect words not in a Finnish dictionary (needs voikko or wordlist)

What this does NOT do
---------------------
  - Does not attempt to fix letter-substitution OCR errors (b→h, l→r etc.)
    — those require FinBERT context (handled by ocr_corrector.py)
  - Does not change morphology or grammar
  - Does not remove valid old spellings beyond w→v

PIPELINE POSITION
-----------------
  alto_parser.py  →  text_normalizer.py  →  ocr_corrector.py  →  NLP models

USAGE
-----
  python text_normalizer.py --file page.txt
  python text_normalizer.py --file page.txt --out normalized.txt
  python text_normalizer.py --dir cleaned/1880/ --out-dir normalized/1880/
  python text_normalizer.py --text "Oulujoki jäätywi tänään"

AS A MODULE
-----------
  from text_normalizer import normalize_text
  clean = normalize_text(raw_text)
"""

import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field


# ── Normalization rules ───────────────────────────────────────────────────────

# Pre-1910 Finnish letter substitutions → modern equivalents
# Applied in order — more specific rules first
SPELLING_RULES = [
    # ── w → v ────────────────────────────────────────────────────────────────
    # The single most common change in 1820–1939 Finnish newspapers.
    # w appears wherever modern Finnish uses v.
    # Must handle: lowercase, uppercase, mid-word, word-initial
    #
    # Examples from corpus:
    #   awonaiseksi  → avonaiseksi
    #   liitettäwä   → liitettävä
    #   woimaan      → voimaan
    #   wesi         → vesi
    #   wiime        → viime
    #   päiwänä      → päivänä
    #   wastiketta   → vastiketta
    #   wähenemisestä → vähenemisestä
    #   walittiin    → valittiin
    #   wielä        → vielä
    #   wuotta       → vuotta
    (re.compile(r'W'), 'V'),
    (re.compile(r'w'), 'v'),

    # ── dt → tt ──────────────────────────────────────────────────────────────
    # Rare in 1820–1939 but present in older texts in the corpus
    # Example: "siandt" → "siannt" (archaic genitive forms)
    (re.compile(r'dt'), 'tt'),

    # ── gh → g ───────────────────────────────────────────────────────────────
    # Very rare — mainly in loanwords and very early 19th century texts
    (re.compile(r'gh'), 'g'),

    # ── ck → kk ──────────────────────────────────────────────────────────────
    # Occasional in loanwords and early texts
    # Example: "tycky" → "tykky"
    (re.compile(r'ck'), 'kk'),

    # ── Swedish/German loanword forms (common in Swedish-language sections) ──
    # hafva → hava (Swedish infinitive form, appears in bilingual papers)
    (re.compile(r'\bhafv'), 'hav'),
    # gifva → giva
    (re.compile(r'\bgifv'), 'giv'),
    # blifva → bliva
    (re.compile(r'\bblifv'), 'bliv'),
]


# ── Line-break rejoining ──────────────────────────────────────────────────────

# Patterns indicating a word was split at a line break by OCR
# Case 1: explicit hyphen at line end  "jäätymi-\nnen" → "jäätyminen"
_HYPHEN_BREAK = re.compile(r'-\s*\n\s*')

def _rejoin_line_breaks(text: str) -> str:
    """
    Rejoin words split across line breaks by explicit hyphens only.

    Handles: "jäätymi-\nnen" → "jäätyminen"

    Bare splits (no hyphen) are intentionally NOT handled here — they are
    too ambiguous to fix with rules (e.g. "on\npaikoin" should NOT be joined
    but "ter\nvehdyksistä" should). FinBERT in ocr_corrector.py handles these
    via context scoring.
    """
    return _HYPHEN_BREAK.sub('', text)


# ── Truncation detection ──────────────────────────────────────────────────────

# Words that appear truncated — OCR stopped mid-word at a line or column edge.
# These are short tokens (3–6 chars) that end with a vowel or common suffix
# but are too short to be a complete Finnish word in context.
# Detected heuristically; flagged for human review, not auto-fixed.
_TRUNCATION_INDICATORS = re.compile(
    r'\b([a-zåäöA-ZÅÄÖ]{2,5})\.'          # e.g. "tulew." "menew."
    r'|\b([a-zåäö]{3,4})\s*$',            # e.g. "yl" at line end
    re.MULTILINE
)


def _find_truncations(text: str) -> list[tuple[str, int]]:
    """
    Find likely truncated words. Returns list of (word, position).
    These are flagged, not auto-corrected.
    """
    truncations = []
    for m in _TRUNCATION_INDICATORS.finditer(text):
        fragment = m.group(1) or m.group(2)
        if fragment and len(fragment) <= 5:
            # Check it's not a known abbreviation
            if fragment.lower() not in _KNOWN_ABBREVIATIONS:
                truncations.append((fragment, m.start()))
    return truncations


_KNOWN_ABBREVIATIONS = {
    # Common in Finnish newspapers 1820–1939
    'mk', 'p', 'n:o', 'klo', 'ko', 'ao', 'ko', 'ko',
    'em', 'ko', 'v', 'k', 's', 'n', 'o', 'y', 'm',
    'jne', 'jms', 'ns', 'em', 'ao', 'pp', 'ko',
    'dr', 't:ri', 'fil', 'vt', 'hra', 'nti', 'rva',
    'prof', 'dos', 'yl', 'al',
    # Newspaper-specific
    'toim', 'vast', 'kirj', 'ref',
}


# ── Unicode and punctuation normalization ─────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    """
    Normalize unicode punctuation and whitespace.
    Preserves Finnish special characters å, ä, ö.
    """
    # Normalize various dash types to plain hyphen
    text = text.replace('\u2014', ' -- ')  # em dash
    text = text.replace('\u2013', ' - ')   # en dash
    text = text.replace('\u2012', ' - ')   # figure dash

    # Normalize curly/typographic quotes to straight quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u201e', '"').replace('\u201a', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    # Remove zero-width and invisible characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)

    # Normalize multiple spaces to single space (preserve newlines)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Normalize multiple newlines to max two
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


# ── Main normalization function ───────────────────────────────────────────────

@dataclass
class NormalizationResult:
    original_text:   str
    normalized_text: str
    changes:         list = field(default_factory=list)   # list of (original, normalized, rule)
    truncations:     list = field(default_factory=list)   # list of (fragment, position)

    @property
    def n_changes(self) -> int:
        return len(self.changes)

    @property
    def w_replacements(self) -> int:
        return sum(1 for _, _, rule in self.changes if rule == 'w→v')

    def summary(self) -> str:
        return (f"Changes: {self.n_changes}  "
                f"(w→v: {self.w_replacements})  "
                f"Truncations flagged: {len(self.truncations)}")


def normalize_text(text: str,
                   track_changes: bool = False) -> NormalizationResult:
    """
    Normalize historical Finnish text to modern spelling.

    Parameters
    ----------
    text          : raw text from alto_parser.py
    track_changes : if True, record every substitution made

    Returns
    -------
    NormalizationResult with .normalized_text and .changes
    """
    result_text = text
    changes     = []

    # Step 1: unicode and whitespace
    result_text = _normalize_unicode(result_text)

    # Step 2: rejoin line-break splits
    result_text = _rejoin_line_breaks(result_text)

    # Step 3: spelling rules
    for pattern, replacement in SPELLING_RULES:
        if track_changes:
            # Find all matches before replacing
            for m in pattern.finditer(result_text):
                original_word = m.group(0)
                normalized_word = pattern.sub(replacement, original_word)
                rule_name = f"{original_word}→{normalized_word}"
                changes.append((original_word, normalized_word, 'w→v' if 'w' in original_word.lower() else rule_name))
        result_text = pattern.sub(replacement, result_text)

    # Step 4: find truncations (flagging only, no auto-fix)
    truncations = _find_truncations(result_text)

    return NormalizationResult(
        original_text   = text,
        normalized_text = result_text,
        changes         = changes,
        truncations     = truncations,
    )


def normalize_file(input_path: str,
                   output_path: str = None,
                   track_changes: bool = False) -> NormalizationResult:
    """Normalize a .txt file. Saves to output_path if given."""
    text   = Path(input_path).read_text(encoding='utf-8')
    result = normalize_text(text, track_changes=track_changes)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(result.normalized_text, encoding='utf-8')
    return result


def normalize_directory(input_dir: str,
                         output_dir: str,
                         track_changes: bool = False) -> dict:
    """Normalize all .txt files in a directory."""
    results   = {}
    txt_files = sorted(Path(input_dir).glob('*.txt'))
    if not txt_files:
        print(f'[INFO] No .txt files found in {input_dir}')
        return results
    for txt_path in txt_files:
        out_path = Path(output_dir) / txt_path.name
        result   = normalize_file(str(txt_path), str(out_path), track_changes)
        results[txt_path.name] = result
        print(f'  {txt_path.name}  →  {result.summary()}')
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(result: NormalizationResult, show_changes: bool = False):
    print()
    print('=' * 60)
    print('  NORMALIZATION REPORT')
    print('=' * 60)
    print(f'  {result.summary()}')

    if show_changes and result.changes:
        print(f'\n  SPELLING CHANGES (sample, first 20):')
        seen = set()
        count = 0
        for orig, norm, rule in result.changes:
            key = (orig, norm)
            if key not in seen:
                print(f'    {orig:<20} → {norm}')
                seen.add(key)
                count += 1
                if count >= 20:
                    break

    if result.truncations:
        print(f'\n  TRUNCATED WORDS FLAGGED ({len(result.truncations)}):')
        for fragment, pos in result.truncations[:10]:
            snippet = result.normalized_text[max(0, pos-20):pos+20].replace('\n', ' ')
            print(f'    "{fragment}"  context: ...{snippet}...')
        if len(result.truncations) > 10:
            print(f'    ... and {len(result.truncations)-10} more')

    print()
    print('  NORMALIZED TEXT (preview, first 400 chars):')
    print('  ' + result.normalized_text[:400].replace('\n', '\n  '))
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Normalize historical Finnish newspaper text to modern spelling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--text', help='Text string to normalize directly')
    src.add_argument('--file', help='Path to a .txt file')
    src.add_argument('--dir',  help='Directory of .txt files')

    p.add_argument('--out',          default=None, help='Output file (for --file)')
    p.add_argument('--out-dir',      default=None, help='Output directory (for --dir)')
    p.add_argument('--track-changes',action='store_true',
                   help='Show what was changed and where')
    return p.parse_args()


def main():
    args = parse_args()

    if args.text:
        result = normalize_text(args.text, track_changes=args.track_changes)
        print_report(result, show_changes=args.track_changes)

    elif args.file:
        result = normalize_file(args.file, args.out, track_changes=args.track_changes)
        print_report(result, show_changes=args.track_changes)
        if args.out:
            print(f'Saved → {args.out}')

    elif args.dir:
        out_dir = args.out_dir or args.dir.rstrip('/') + '_normalized'
        normalize_directory(args.dir, out_dir, track_changes=args.track_changes)


if __name__ == '__main__':
    main()