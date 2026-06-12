"""
classify_river_date.py — river name and date extractor for historical Finnish newspapers

Handles:
  - Heavily inflected Finnish river names (stem matching)
  - Pre-1945 date formats: "Maaliskuun 11 p:nä", "11 p:nä maaliskuuta", "11/3 1920"
  - Modern date formats: "11.3.1920", "11.3."
  - Year extraction from filename (e.g. "1902_page-003.txt") as fallback

Can be used standalone or imported by extract_ice_files.py.
"""

import re
from pathlib import Path


# ── River definitions ──────────────────────────────────────────────────────────
# Each entry: canonical name → list of stems to match (lowercase, no diacritics
# needed because we match after normalize_historical).
# Stems are matched as word-boundary prefixes so "tornion" matches
# "tornionjoella", "tornionjoen", "tornionjoki", "tornionjoesta" etc.
# Order matters: longer/more specific stems first to avoid shadowing.

RIVERS = {
    "Tornionjoki": [
        r"tornionjok",      # tornionjoki, tornionjoen, tornionjoella …
        r"tornionjoe",
        r"tornion",         # Tornion (city name doubles as river ref in context)
    ],
    "Kemijoki": [
        r"kemijok",
        r"kemijoe",
        r"kemin",           # Kemin kaupunki / Kemijoen suu
        r"kemijo",
    ],
    "Oulujoki": [
        r"oulujok",
        r"oulujoe",
        r"oulunjo",
        r"oulujo",
    ],
    "Kiiminkijoki": [
        r"kiiminkijok",
        r"kiiminkijoe",
        r"kiiminki",
    ],
    "Tenojoki": [
        r"tenojok",
        r"tenojoe",
        r"teno",
    ],
    "Iiijoki": [
        r"iiijok",
        r"iiijoe",
        r"iijok",
        r"iijoe",
    ],
    "Kokemäenjoki": [
        r"kokemäenjok",
        r"kokemäenjoe",
        r"kokemäe",
        r"kokemäenjo",
    ],
    "Kymiijoki": [
        r"kymijok",
        r"kymijoe",
        r"kymiä",
    ],
    "Vuoksi": [
        r"vuoksi",
        r"vuokse",
        r"vuoksen",
    ],
}

# Compile patterns: match stem followed by a Finnish suffix character or end of word
_RIVER_PATTERNS = {
    river: [re.compile(r"\b" + stem, re.IGNORECASE) for stem in stems]
    for river, stems in RIVERS.items()
}


# ── Finnish month lookup ───────────────────────────────────────────────────────
# Covers modern and archaic genitive/partitive forms found in old newspapers.

FINNISH_MONTHS = {
    # Month number → list of Finnish name stems (lowercase)
    1:  ["tammiku",  "tammikuu",  "januari",  "januarik"],
    2:  ["helmiku",  "helmikuu",  "februari", "februari"],
    3:  ["maalisk",  "maaliskuu", "mars",     "martisk"],
    4:  ["huhtiku",  "huhtikuu",  "april",    "aprilik"],
    5:  ["toukoku",  "toukokuu",  "maj",      "maiku"],
    6:  ["kesäku",   "kesäkuu",   "juni",     "junik"],
    7:  ["heinäku",  "heinäkuu",  "juli",     "julik"],
    8:  ["eloku",    "elokuu",    "augusti",  "augustik"],
    9:  ["syysku",   "syyskuu",   "septemb",  "septembrik"],
    10: ["lokaku",   "lokakuu",   "oktob",    "oktobrik"],
    11: ["marrasku", "marraskuu", "novemb",   "novembrik"],
    12: ["jouluku",  "joulukuu",  "decemb",   "decembrik"],
}

# Reverse lookup: stem → month number
_MONTH_STEM_TO_NUM = {}
for num, stems in FINNISH_MONTHS.items():
    for stem in stems:
        _MONTH_STEM_TO_NUM[stem] = num

# Generate month pattern dynamically from dictionary (longer stems first to avoid shadowing)
_MONTH_STEMS_PATTERN = "|".join(sorted(set(_MONTH_STEM_TO_NUM.keys()), key=len, reverse=True))

# ── Date patterns ──────────────────────────────────────────────────────────────
# Listed from most specific to least specific.

# "11 p:nä Maaliskuuta 1920"  /  "Maaliskuun 11 p:nä 1920"
# Month pattern generated from _MONTH_STEM_TO_NUM to auto-support new archaic forms
_PAT_FINN_LONG = re.compile(
    r"(?:(\d{1,2})\s*p[:\.]?n[äa]\s+)?(" + _MONTH_STEMS_PATTERN + r"\w*)"
    r"(?:\s+(\d{1,2})\s*p[:\.]?n[äa])?"
    r"(?:\s+(\d{4}))?",
    re.IGNORECASE,
)

# "11.3.1920"  or  "11.3."  or  "11/3/1920"  or  "11/3"
_PAT_NUMERIC = re.compile(
    r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{4}|\d{2}))?\b"
)

# Bare year: "vuonna 1903" / "v. 1903" / standalone 4-digit year
_PAT_YEAR = re.compile(
    r"(?:vuonna|v\.)\s*(\d{4})\b|\b(1[89]\d{2})\b"
)


def _month_num_from_text(text: str) -> int | None:
    """Match a Finnish/Swedish month word to a month number."""
    t = text.lower()
    # Try longest match first
    for stem in sorted(_MONTH_STEM_TO_NUM, key=len, reverse=True):
        if t.startswith(stem) or stem in t:
            return _MONTH_STEM_TO_NUM[stem]
    return None


def extract_dates(text: str) -> list[dict]:
    """
    Extract all date mentions from a text block.
    Returns list of dicts: {day, month, year, raw, format}
    day/month/year may be None if not found.
    """
    dates = []
    seen_spans = []

    def overlaps(start, end):
        return any(s < end and start < e for s, e in seen_spans)

    # 1. Numeric dates  11.3.1920 / 11/3
    for m in _PAT_NUMERIC.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        day   = int(m.group(1))
        month = int(m.group(2))
        year_raw = m.group(3)
        year = int(year_raw) if year_raw and len(year_raw) == 4 else \
               (1900 + int(year_raw) if year_raw and len(year_raw) == 2 else None)
        if 1 <= day <= 31 and 1 <= month <= 12:
            dates.append({"day": day, "month": month, "year": year,
                          "raw": m.group(0), "format": "numeric"})
            seen_spans.append((m.start(), m.end()))

    # 2. Finnish long form: "Maaliskuun 11 p:nä" / "11 p:nä maaliskuuta"
    for m in _PAT_FINN_LONG.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        month_word = m.group(2)
        month = _month_num_from_text(month_word)
        if month is None:
            continue
        day_a, day_b = m.group(1), m.group(3)
        day = int(day_a or day_b) if (day_a or day_b) else None
        year_raw = m.group(4)
        year = int(year_raw) if year_raw else None
        if day and not (1 <= day <= 31):
            day = None
        dates.append({"day": day, "month": month, "year": year,
                      "raw": m.group(0).strip(), "format": "finnish_long"})
        seen_spans.append((m.start(), m.end()))

    # 3. Bare year mentions (used as fallback if no full date found)
    for m in _PAT_YEAR.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        year = int(m.group(1) or m.group(2))
        dates.append({"day": None, "month": None, "year": year,
                      "raw": m.group(0).strip(), "format": "year_only"})
        seen_spans.append((m.start(), m.end()))

    return dates


def year_from_filename(filename: str) -> int | None:
    """Extract a 4-digit year from the filename as a last-resort fallback."""
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", filename)
    return int(m.group(1)) if m else None


def date_from_filename(filename: str) -> str | None:
    """Extract publishing date in YYYY-MM-DD format from filename."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    return m.group(0) if m else None


# ── River detection ────────────────────────────────────────────────────────────

def detect_rivers(text: str) -> list[dict]:
    """
    Return list of {river, matched_text, position} for all river mentions.
    """
    hits = []
    for river, patterns in _RIVER_PATTERNS.items():
        for pat in patterns:
            for m in pat.finditer(text):
                # Extract the full word for context
                word_end = m.end()
                while word_end < len(text) and (text[word_end].isalpha() or text[word_end] in "äöå"):
                    word_end += 1
                hits.append({
                    "river":        river,
                    "matched_text": text[m.start():word_end],
                    "position":     m.start(),
                })
    # Sort by position, deduplicate overlapping matches (keep longest)
    hits.sort(key=lambda x: x["position"])
    deduped = []
    last_end = -1
    for h in hits:
        start = h["position"]
        end   = start + len(h["matched_text"])
        if start >= last_end:
            deduped.append(h)
            last_end = end
    return deduped


# ── Sentence-level classifier ──────────────────────────────────────────────────

def classify_sentence_full(sentence: str, filename: str = "") -> dict:
    """
    Run river + date classification on a single sentence.
    Returns:
      rivers  — list of river names mentioned (or ["unknown"] if none found)
      dates   — list of date dicts found in sentence
      year    — best year guess (from sentence, else from filename)
      publishing_date — date extracted from filename in YYYY-MM-DD format
    """
    rivers = detect_rivers(sentence)
    dates  = extract_dates(sentence)

    # Best year: prefer full date > year_only > filename
    year = None
    for d in dates:
        if d["year"] and d["format"] != "year_only":
            year = d["year"]
            break
    if year is None:
        for d in dates:
            if d["year"]:
                year = d["year"]
                break
    if year is None:
        year = year_from_filename(filename)

    # Use "unknown" if no rivers found
    river_list = [r["river"] for r in rivers] if rivers else ["unknown"]

    # Extract publishing date from filename
    publishing_date = date_from_filename(filename)

    return {
        "rivers": river_list,
        "river_matches": rivers,
        "dates":  dates,
        "year":   year,
        "publishing_date": publishing_date,
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_SENTENCES = [
        "Maaliskuun 11 p:nä Tornionjoki jäätyi täysin.",
        "Kemijoen jäänlähtö tapahtui 15.5.1912.",
        "Oulujoki awautui jäistä jo huhtikuussa vuonna 1905.",
        "Tulwa rannoilla oli suuri — Kiiminkijoen wesi nousi.",
        "Tenojoen jääpato aiheutti wahinkoja 3/4 1899.",
        "11 p:nä huhtikuuta Kemijoen jäät lähtiwät liikkeelle.",
        "Ei jokia eikä päivämäärää tässä lauseessa.",
        "Tornionjoen wesi oli korkealla maaliskuulla 1903.",
    ]

    print("River & date classification — test\n")
    for sent in TEST_SENTENCES:
        result = classify_sentence_full(sent, filename="1903-03-15_page-001.txt")
        print(f"  Sentence : {sent}")
        print(f"  Rivers   : {result['rivers']}")
        print(f"  Dates    : {[d['raw'] for d in result['dates']] or '—'}")
        print(f"  Year     : {result['year'] or '—'}")
        print(f"  Publishing date: {result['publishing_date'] or '—'}")
        print()
