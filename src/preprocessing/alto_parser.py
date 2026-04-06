#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alto_parser.py
==============
Parse ALTO XML files downloaded from digi.kansalliskirjasto.fi
and extract clean plain text.

ALTO (Analyzed Layout and Text Object) is the XML format used by the
National Library to store OCR results. Each file represents one newspaper page.

USAGE
-----
  # As a script — parse a single file:
  python alto_parser.py path/to/file.xml

  # As a module — use in your pipeline:
  from src.preprocessing.alto_parser import parse_alto_file, parse_alto_directory

FUNCTIONS
---------
  parse_alto_file(path)         → str   : Extract text from one ALTO XML file
  parse_alto_directory(dir)     → dict  : Parse all XMLs in a directory
  clean_ocr_text(text)          → str   : Basic OCR noise cleaning
"""

import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_alto_file(xml_path: str) -> str:
    """
    Extract all word content from an ALTO XML file.
    Returns a single string of space-joined words.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # ALTO XML uses namespaces — handle both namespaced and bare tags
        words = []
        for el in root.iter():
            content = el.get("CONTENT")
            if content is not None and content.strip():
                words.append(content.strip())

        return " ".join(words)

    except ET.ParseError as e:
        print(f"[WARN] Could not parse {xml_path}: {e}")
        return ""
    except FileNotFoundError:
        print(f"[ERROR] File not found: {xml_path}")
        return ""


def clean_ocr_text(text: str) -> str:
    """
    Basic cleaning for Finnish OCR text from 19th/early 20th century newspapers.

    Common OCR issues in this corpus:
      - Long hyphen sequences used as separators
      - Repeated punctuation from column borders
      - Stray single characters from OCR artifacts
    """
    # Remove long separator lines (e.g. "--------" or "========")
    text = re.sub(r'[-=_]{3,}', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove lines that are only punctuation/symbols
    lines = [ln for ln in text.splitlines() if re.search(r'[a-zåäöA-ZÅÄÖ]', ln)]
    return "\n".join(lines).strip()


def parse_alto_directory(directory: str,
                         clean: bool = True) -> dict:
    """
    Parse all ALTO XML files in a directory (non-recursive).
    Returns a dict mapping filename → extracted text.
    """
    results = {}
    xml_files = sorted(Path(directory).glob("*.xml"))

    if not xml_files:
        print(f"[INFO] No XML files found in {directory}")
        return results

    for xml_path in xml_files:
        text = parse_alto_file(str(xml_path))
        if clean:
            text = clean_ocr_text(text)
        results[xml_path.name] = text

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python alto_parser.py <path/to/file.xml or directory/>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        results = parse_alto_directory(target)
        for fname, text in results.items():
            print(f"\n{'='*60}\n{fname}\n{'='*60}")
            print(text[:500] + ("..." if len(text) > 500 else ""))
    else:
        text = parse_alto_file(target)
        text = clean_ocr_text(text)
        print(text)
