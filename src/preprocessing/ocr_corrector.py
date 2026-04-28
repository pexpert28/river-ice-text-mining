#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_corrector.py
================
Detect and correct OCR errors in historical Finnish newspaper text using
a two-layer approach:

  Layer 1 — Rule-based detection (fast, no model needed)
    Catches systematic OCR patterns: digit/letter substitutions,
    impossible character sequences, known noise patterns.

  Layer 2 — FinBERT contextual scoring (TurkuNLP/bert-base-finnish-cased-v1)
    Scores each token in context using masked language modelling.
    Tokens with very low probability in context are flagged as likely
    OCR errors and optionally replaced with the model's top suggestion.

The two layers complement each other: rules catch clear-cut cases fast,
FinBERT catches words that look plausible in isolation but are wrong in
context (e.g. "jäätywi" is a real old spelling but "jäätywz" is OCR noise).

PIPELINE POSITION
-----------------
  alto_parser.py  →  text_normalizer.py  →  ocr_corrector.py  →  NLP models

USAGE
-----
  pip install transformers torch

  # Correct a single text string:
  python ocr_corrector.py --text "Oulujoki jäätywi tänään klo 7 e.pp."

  # Correct a .txt file produced by alto_parser / text_normalizer:
  python ocr_corrector.py --file cleaned/1880/binding_page-00001.txt

  # Correct all .txt files in a directory:
  python ocr_corrector.py --dir cleaned/1880/ --out-dir corrected/1880/

  # Detect only — show flagged tokens without replacing:
  python ocr_corrector.py --file page.txt --detect-only

  # Skip FinBERT (rules only, fast):
  python ocr_corrector.py --file page.txt --rules-only

  # Adjust FinBERT confidence threshold (0.0–1.0, default 0.01):
  python ocr_corrector.py --file page.txt --threshold 0.005

AS A MODULE
-----------
  from ocr_corrector import OCRCorrector

  corrector = OCRCorrector(use_model=True)
  result = corrector.correct_text("Joki jäätywi tänään.")
  print(result.corrected_text)
  for flag in result.flags:
      print(flag)   # OCRFlag(original, suggestion, reason, confidence)
"""

import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class OCRFlag:
    """One flagged token with its context and suggested correction."""
    original:    str
    suggestion:  str          # best replacement, or "" if no suggestion
    reason:      str          # "rule" or "model"
    rule_name:   str  = ""    # which rule triggered (if reason=="rule")
    confidence:  float = 0.0  # model probability of suggestion (0–1)
    position:    int   = 0    # character offset in original text
    context:     str   = ""   # surrounding words for human review


@dataclass
class CorrectionResult:
    """Output of correcting one piece of text."""
    original_text:  str
    corrected_text: str
    flags:          list = field(default_factory=list)   # list[OCRFlag]

    @property
    def n_flagged(self) -> int:
        return len(self.flags)

    @property
    def n_corrected(self) -> int:
        return sum(1 for f in self.flags if f.suggestion and f.suggestion != f.original)

    def summary(self) -> str:
        return (f"Flagged: {self.n_flagged} tokens  |  "
                f"Corrected: {self.n_corrected} tokens")


# ── Layer 1: Rule-based detection ─────────────────────────────────────────────

# Common OCR digit↔letter substitutions in 19th century Finnish scans
DIGIT_LETTER_MAP = {
    "0": "o",  "1": "l",  "1": "i",  "3": "s",
    "5": "s",  "6": "b",  "7": "t",  "8": "B",
}

# Regex patterns that strongly indicate OCR garbage
# Each entry: (rule_name, pattern, description)
RULE_PATTERNS = [
    # Digit embedded in alphabetic word: "jäätyw1nen", "us7sssa"
    ("digit_in_word",
     re.compile(r'\b[a-zA-ZåäöÅÄÖ]+[0-9][a-zA-ZåäöÅÄÖ]+\b'),
     "Digit embedded inside a word"),

    # Long run of same character: "sssss", "aaaaa"
    ("repeated_chars",
     re.compile(r'\b\w*(.)\1{3,}\w*\b'),
     "Character repeated 4+ times"),

    # All-caps token longer than 7 chars that is not a known abbreviation
    # Catches: SEESCETTTE, HNBOULET, BSETSIS
    ("long_allcaps",
     re.compile(r'\b[A-ZÅÄÖ]{8,}\b'),
     "All-caps token longer than 7 chars"),

    # Mixed case chaos: "tÄlLauHliTa"
    ("mixed_case_chaos",
     re.compile(r'\b(?:[a-zåäö][A-ZÅÄÖ]){3,}\b'),
     "Alternating lower/upper case"),

    # Non-word characters embedded: "jää|tyy", "Oulu+joki"
    ("embedded_symbols",
     re.compile(r'\b[a-zA-ZåäöÅÄÖ]+[|+\\#@$%^&*<>][a-zA-ZåäöÅÄÖ]+\b'),
     "Symbol embedded inside a word"),

    # 5+ consecutive consonants anywhere inside a word — excluding y which is
    # always a vowel in Finnish (sounds like German ü).
    # Threshold raised from 4 to 5 to avoid flagging valid Finnish clusters
    # like -nty-, -ksy-, -rty-, -yst- which are extremely common.
    # Still catches: Virorgrg (rgrg=4, but caught via low_vowel_ratio),
    # genuine garbage like "bstrs", "rgrgrg"
    # 5+ consecutive true consonants (y excluded — always a vowel in Finnish).
    # Threshold raised from 4 to 5; catches genuine garbage like "rgrgrg", "bstrs"
    # but no longer flags normal Finnish clusters like -nty-, -ksy-, -rty-
    ("internal_cluster_5",
     re.compile("(?i)[bcdfghjklmnpqrstvwxz]{5,}"),
     "5+ consecutive consonants inside a word (y excluded)"),

    # Word starts with 3+ true consonants (y excluded).
    # Catches Tllauihlita (Tll-) while leaving yhtiö, yhdistys, hywin alone.
    ("bad_consonant_start",
     re.compile("(?<![a-zA-Z\u00e5\u00e4\u00f6y])[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{3}[a-zA-Z\u00e5\u00e4\u00f6\u00c5\u00c4\u00d6]+"),
     "Word starts with 3+ consonants, y excluded (unusual in Finnish/Swedish)"),

    # Consonant soup: word with very low vowel ratio (y counts as vowel).
    ("low_vowel_ratio",
     re.compile("[a-zA-Z\u00e5\u00e4\u00f6\u00c5\u00c4\u00d6]{5,}"),
     "Very low vowel ratio (consonant soup)"),
]

# Finnish vowels — y is always a vowel in Finnish (sounds like German ü)
VOWELS = set("aeiouåäöyAEIOUÅÄÖY")

def _is_consonant_soup(word: str, min_length: int = 5, max_vowel_ratio: float = 0.26) -> bool:
    """
    Return True if word has suspiciously few vowels.
    Threshold 0.26 catches Virorgrg (i+o = 2/8 = 25%).
    y counts as a vowel.
    """
    letters = [c for c in word if c.isalpha()]
    if len(letters) < min_length:
        return False
    vowel_count = sum(1 for c in letters if c in VOWELS)
    return (vowel_count / len(letters)) < max_vowel_ratio
    vowel_count = sum(1 for c in letters if c in VOWELS)
    return (vowel_count / len(letters)) < max_vowel_ratio

# Known old Finnish/Swedish spellings that are NOT errors
# (these look unusual but are historically correct)
HISTORICAL_WHITELIST = {
    # Old Finnish w→v forms — valid pre-1910
    "waipui", "wapahtaja", "waimoni", "wanhempi", "wastaan", "waimo",
    "weljet", "weli", "wastata", "walittu", "warma", "wanhus",
    "wesi", "wetäytyi", "wiime", "wiisi", "wuosi", "wuotta",
    "woiminen", "woitto", "werran", "weljespoika", "wankila",
    # Old Swedish forms
    "hafva", "hafver", "gifva", "blifva", "blifver",
    "äfven", "äfvensom", "ofvan", "ofvanför",
    # Common abbreviations in Finnish newspapers
    "e.pp.", "j.pp.", "a.-p.", "i.-p.", "k:lo", "n:o", "s:n",
    "v:lta", "t:mi", "o.-y", "o.-y.", "a.-y.",
}


def rule_based_flags(text: str) -> list[OCRFlag]:
    """
    Run all rule patterns against the text.
    Returns a list of OCRFlag objects for each suspicious token found.
    """
    flags = []
    words = text.split()

    for i, word in enumerate(words):
        clean = word.strip(".,;:!?\"'()[]")

        # Skip whitelisted historical forms
        if clean.lower() in HISTORICAL_WHITELIST:
            continue

        # Skip pure numbers or single chars
        if not re.search(r'[a-zA-ZåäöÅÄÖ]', clean) or len(clean) < 2:
            continue

        # Context window for human review
        ctx_start = max(0, i - 3)
        ctx_end   = min(len(words), i + 4)
        context   = " ".join(words[ctx_start:ctx_end])
        pos = text.find(word)

        for rule_name, pattern, description in RULE_PATTERNS:
            if rule_name == "low_vowel_ratio":
                if not (pattern.search(clean) and _is_consonant_soup(clean)):
                    continue
            else:
                if not pattern.search(clean):
                    continue

            suggestion = _apply_digit_letter_fix(clean)
            flags.append(OCRFlag(
                original   = word,
                suggestion = suggestion if suggestion != clean else "",
                reason     = "rule",
                rule_name  = rule_name,
                confidence = 0.0,
                position   = pos,
                context    = context,
            ))
            break

    return flags


def _apply_digit_letter_fix(word: str) -> str:
    """Replace digit→letter substitutions in a word."""
    result = word
    for digit, letter in DIGIT_LETTER_MAP.items():
        result = result.replace(digit, letter)
    return result


# ── Layer 2: FinBERT contextual scoring ───────────────────────────────────────

class FinBERTScorer:
    """
    Wraps TurkuNLP/bert-base-finnish-cased-v1 for masked language modelling.

    For each token in the text, masks it and asks the model what it
    predicts — if the original token has very low probability, it is
    flagged as a likely OCR error and the top prediction is returned
    as a suggested correction.
    """

    MODEL_NAME = "TurkuNLP/bert-base-finnish-cased-v1"
    MAX_TOKENS = 512   # BERT hard limit

    def __init__(self, threshold: float = 0.01):
        """
        Parameters
        ----------
        threshold : float
            Tokens whose fill-mask probability is below this value are
            flagged. Default 0.01 (1%). Lower = more sensitive.
        """
        self.threshold = threshold
        self._model     = None
        self._tokenizer = None
        self._pipeline  = None

    def _load(self):
        """Lazy-load the model on first use."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            print(f"  [FinBERT] Loading {self.MODEL_NAME} ... (first run downloads ~440 MB)")
            self._pipeline = pipeline(
                "fill-mask",
                model     = self.MODEL_NAME,
                tokenizer = self.MODEL_NAME,
                top_k     = 5,
                device    = -1,   # CPU; change to 0 for GPU
            )
            print(f"  [FinBERT] Model loaded.")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for FinBERT scoring.\n"
                "Install with: pip install transformers torch"
            )

    def score_text(self, text: str) -> list[OCRFlag]:
        """
        Score all tokens in the text. Returns flags for low-probability tokens.

        Strategy:
          - Split text into sentences (to stay within 512 token limit)
          - For each word token: replace with [MASK], run fill-mask
          - If original word is not in top-5 predictions OR has prob < threshold,
            flag it with the top prediction as suggestion
        """
        self._load()
        flags = []
        sentences = _split_sentences(text)

        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue

            for i, word in enumerate(words):
                # Skip very short tokens, punctuation, numbers, whitelist
                clean = word.strip(".,;:!?\"'()[]")
                if (len(clean) < 3
                        or not re.search(r'[a-zA-ZåäöÅÄÖ]', clean)
                        or clean.lower() in HISTORICAL_WHITELIST):
                    continue

                # Build masked sentence
                masked_words    = words.copy()
                masked_words[i] = "[MASK]"
                masked_sentence = " ".join(masked_words)

                # Truncate if too long
                if len(masked_sentence.split()) > self.MAX_TOKENS - 2:
                    continue

                try:
                    predictions = self._pipeline(masked_sentence)
                except Exception:
                    continue

                # predictions is a list of dicts: {token_str, score, ...}
                top_tokens = [p["token_str"].strip() for p in predictions]
                top_scores = {p["token_str"].strip(): p["score"] for p in predictions}

                # Find probability of original word in predictions
                original_prob = top_scores.get(clean, 0.0)

                if original_prob < self.threshold:
                    # Suggest the top prediction
                    suggestion = top_tokens[0] if top_tokens else ""

                    # Context
                    ctx_start = max(0, i - 3)
                    ctx_end   = min(len(words), i + 4)
                    context   = " ".join(words[ctx_start:ctx_end])

                    flags.append(OCRFlag(
                        original   = word,
                        suggestion = suggestion,
                        reason     = "model",
                        rule_name  = "finbert_low_prob",
                        confidence = top_scores.get(suggestion, 0.0),
                        position   = text.find(word),
                        context    = context,
                    ))

        return flags


def _split_sentences(text: str, max_words: int = 60) -> list[str]:
    """
    Split text into sentence-sized chunks that fit within BERT's token limit.
    Splits on sentence-ending punctuation first, then on word count.
    """
    # Split on sentence boundaries
    raw = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for chunk in raw:
        words = chunk.split()
        # Further split long chunks by word count
        for start in range(0, len(words), max_words):
            chunks.append(" ".join(words[start:start + max_words]))
    return [c for c in chunks if c.strip()]


# ── Main corrector class ──────────────────────────────────────────────────────

class OCRCorrector:
    """
    Main interface. Combines rule-based and FinBERT-based OCR correction.

    Parameters
    ----------
    use_model   : bool   — use FinBERT for contextual scoring (default True)
    rules_only  : bool   — skip FinBERT, rules only (faster)
    threshold   : float  — FinBERT probability threshold (default 0.01)
    auto_correct: bool   — apply suggestions automatically (default False)
                          If False, flags are returned but text is unchanged
                          unless the correction is certain (rule-based digit fix)
    """

    def __init__(self,
                 use_model:    bool  = True,
                 rules_only:   bool  = False,
                 threshold:    float = 0.01,
                 auto_correct: bool  = False):
        self.use_model    = use_model and not rules_only
        self.threshold    = threshold
        self.auto_correct = auto_correct
        self._bert        = FinBERTScorer(threshold=threshold) if self.use_model else None

    def correct_text(self, text: str) -> CorrectionResult:
        """
        Run both detection layers on a text string.
        Returns a CorrectionResult with flags and (optionally) corrected text.
        """
        if not text.strip():
            return CorrectionResult(
                original_text  = text,
                corrected_text = text,
                flags          = [],
            )

        # Layer 1: rules
        flags = rule_based_flags(text)
        rule_flagged_words = {f.original for f in flags}

        # Layer 2: FinBERT (skip words already flagged by rules)
        if self.use_model and self._bert:
            model_flags = self._bert.score_text(text)
            # Deduplicate — don't double-flag the same word
            for mf in model_flags:
                if mf.original not in rule_flagged_words:
                    flags.append(mf)

        # Apply corrections if requested
        corrected = text
        if self.auto_correct and flags:
            corrected = _apply_corrections(text, flags)

        return CorrectionResult(
            original_text  = text,
            corrected_text = corrected,
            flags          = flags,
        )

    def correct_file(self, input_path: str,
                     output_path: Optional[str] = None) -> CorrectionResult:
        """
        Correct a .txt file. Saves corrected version to output_path if given.
        """
        text   = Path(input_path).read_text(encoding="utf-8")
        result = self.correct_text(text)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(result.corrected_text, encoding="utf-8")

        return result

    def correct_directory(self, input_dir: str,
                           output_dir: str) -> dict[str, CorrectionResult]:
        """
        Correct all .txt files in a directory.
        Returns {filename: CorrectionResult}.
        """
        results  = {}
        txt_files = sorted(Path(input_dir).glob("*.txt"))
        if not txt_files:
            print(f"[INFO] No .txt files found in {input_dir}")
            return results

        for txt_path in txt_files:
            out_path = Path(output_dir) / txt_path.name
            result   = self.correct_file(str(txt_path), str(out_path))
            results[txt_path.name] = result
            print(f"  {txt_path.name}  →  {result.summary()}")

        return results


def _apply_corrections(text: str, flags: list[OCRFlag]) -> str:
    """
    Apply flagged corrections to the text.

    Conservative policy:
      - Rule-based digit fixes: apply automatically (high confidence)
      - Model suggestions: apply only if confidence > 0.5
    """
    corrected = text
    # Sort by position descending so replacements don't shift offsets
    sorted_flags = sorted(flags, key=lambda f: f.position, reverse=True)

    for flag in sorted_flags:
        if not flag.suggestion or flag.suggestion == flag.original:
            continue
        should_apply = (
            (flag.reason == "rule" and flag.rule_name == "digit_in_word")
            or (flag.reason == "model" and flag.confidence > 0.5)
        )
        if should_apply:
            corrected = corrected.replace(flag.original, flag.suggestion, 1)

    return corrected


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(result: CorrectionResult, show_all: bool = False):
    """Print a human-readable correction report."""
    print()
    print("=" * 68)
    print(f"  OCR CORRECTION REPORT")
    print("=" * 68)
    print(f"  {result.summary()}")
    print()

    if not result.flags:
        print("  No OCR issues detected.")
        return

    rule_flags  = [f for f in result.flags if f.reason == "rule"]
    model_flags = [f for f in result.flags if f.reason == "model"]

    if rule_flags:
        print(f"  RULE-BASED FLAGS ({len(rule_flags)}):")
        print(f"  {'Token':<22} {'Rule':<22} {'Suggestion':<20}")
        print(f"  {'-'*22} {'-'*22} {'-'*20}")
        for f in rule_flags:
            sugg = f.suggestion or "(no suggestion)"
            print(f"  {f.original[:22]:<22} {f.rule_name[:22]:<22} {sugg[:20]:<20}")
            if show_all:
                print(f"    context: ...{f.context}...")

    if model_flags:
        print()
        print(f"  FINBERT FLAGS ({len(model_flags)}):")
        print(f"  {'Token':<22} {'Suggestion':<22} {'Confidence':>10}")
        print(f"  {'-'*22} {'-'*22} {'-'*10}")
        for f in model_flags:
            sugg = f.suggestion or "(no suggestion)"
            print(f"  {f.original[:22]:<22} {sugg[:22]:<22} {f.confidence:>10.3f}")
            if show_all:
                print(f"    context: ...{f.context}...")

    print()
    if result.corrected_text != result.original_text:
        print("  CORRECTED TEXT (auto-correct was on):")
        print(f"  {result.corrected_text[:300]}")
    else:
        print("  NOTE: auto-correct is OFF. Run with --auto-correct to apply suggestions.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect and correct OCR errors in Finnish historical newspaper text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text",  help="Text string to correct directly")
    src.add_argument("--file",  help="Path to a .txt file to correct")
    src.add_argument("--dir",   help="Directory of .txt files to correct")

    p.add_argument("--out",          default=None,  help="Output file path (for --file)")
    p.add_argument("--out-dir",      default=None,  help="Output directory (for --dir)")
    p.add_argument("--detect-only",  action="store_true",
                   help="Flag errors but do not apply corrections")
    p.add_argument("--rules-only",   action="store_true",
                   help="Skip FinBERT, use rule-based detection only (faster)")
    p.add_argument("--auto-correct", action="store_true",
                   help="Apply high-confidence corrections automatically")
    p.add_argument("--threshold",    type=float, default=0.01,
                   help="FinBERT probability threshold for flagging (default: 0.01)")
    p.add_argument("--verbose",      action="store_true",
                   help="Show context snippets for each flag")
    return p.parse_args()


def main():
    args = parse_args()

    corrector = OCRCorrector(
        use_model    = not args.rules_only,
        rules_only   = args.rules_only,
        threshold    = args.threshold,
        auto_correct = args.auto_correct and not args.detect_only,
    )

    if args.text:
        result = corrector.correct_text(args.text)
        print_report(result, show_all=args.verbose)

    elif args.file:
        print(f"Processing: {args.file}")
        result = corrector.correct_file(args.file, args.out)
        print_report(result, show_all=args.verbose)
        if args.out:
            print(f"Saved corrected text → {args.out}")

    elif args.dir:
        out_dir = args.out_dir or args.dir.rstrip("/") + "_corrected"
        print(f"Processing directory: {args.dir}")
        print(f"Output directory:     {out_dir}\n")
        results = corrector.correct_directory(args.dir, out_dir)
        total_flagged   = sum(r.n_flagged   for r in results.values())
        total_corrected = sum(r.n_corrected for r in results.values())
        print(f"\nTotal: {len(results)} files  |  "
              f"Flagged: {total_flagged} tokens  |  "
              f"Corrected: {total_corrected} tokens")


if __name__ == "__main__":
    main()