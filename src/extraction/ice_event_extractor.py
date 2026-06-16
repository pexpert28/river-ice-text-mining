#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ice_event_extractor.py
======================
Keyword-based baseline extractor for river ice events in historical
Finnish/Swedish newspaper text.

Event types:
  freeze    — jäätyminen, jäätyi, jää peitti ...
  breakup   — jäänlähtö, jää lähti, jäät lähtivät ...
  thickness — jään paksuus, X cm jäätä ...
  jam_flood — jääpato, jäätulva, hyytö ...
              (summer-only tulva mentions are suppressed)

Each returned IceEvent carries:
  pub_date        : publication date from the filename (always present)
  event_date      : date extracted from text, or pub_date if not found
  date_confidence : "exact" | "inferred" | "publication"

USAGE
-----
  from src.extraction.ice_event_extractor import extract_ice_events

  events = extract_ice_events(
      "Oulujoki jäätyi 15. marraskuuta.",
      pub_date="1902-11-17"
  )
"""

import re
from dataclasses import dataclass, field


# ── Keyword dictionaries ───────────────────────────────────────────────────────

ICE_KEYWORDS: dict[str, list[str]] = {
    "freeze": [
        "jäätyi", "jäätyminen", "jäätynyt", "jää peitti", "jää muodostui",
        "jää tuli", "alkoi jäätyä", "kiintojää",
        # Swedish
        "isläggning", "isen lade sig", "isen bildades",
    ],
    "breakup": [
        "jäänlähtö", "jää lähti", "jää meni", "jää suli", "jäät lähtivät",
        "jäät menivät", "jään lähtö", "jäät sulava",
        # Swedish
        "islossning", "isen gick", "isen lossnade",
    ],
    "thickness": [
        "jään paksuus", "paksu jää", "cm jäätä", "cm:n jää", "tuuman jäätä",
    ],
    "jam_flood": [
        "jääpato", "jääpatoja", "jäätulva", "hyytö", "ajojää",
        # generic tulva is intentionally excluded here;
        # it is caught separately and filtered by _is_ice_flood()
        "tulva",
        # Swedish
        "isdamm", "isflak", "översvämning",
    ],
}

TARGET_RIVERS: list[str] = [
    "oulujoki", "kemijoki", "tornionjoki", "kokemäenjoki", "kiiminkijoki",
    "oulu", "kemi", "tornio", "kokemäki", "kiimink",
]


# ── Date extraction ────────────────────────────────────────────────────────────

_MONTH_NAMES_FI = (
    "tammikuu|helmikuu|maaliskuu|huhtikuu|toukokuu|kesäkuu|"
    "heinäkuu|elokuu|syyskuu|lokakuu|marraskuu|joulukuu"
)
_MONTH_NAMES_SV = (
    "januari|februari|mars|april|maj|juni|"
    "juli|augusti|september|oktober|november|december"
)

# "15. marraskuuta", "15 marraskuuta", "15. november"
_DATE_EXACT = re.compile(
    rf'\b(\d{{1,2}})[.\s]\s*({_MONTH_NAMES_FI}|{_MONTH_NAMES_SV})[a-zåäö]*',
    re.IGNORECASE,
)

# Relative references that signal the event is near but not on pub_date
_DATE_RELATIVE = re.compile(
    r'\b(eilen|toissapäivänä|viime\s+\w+na|muutama\s+päivä\s+sitten'
    r'|i\s+går|förra\s+\w+en)\b',
    re.IGNORECASE,
)


def _extract_event_date(snippet: str, pub_date: str) -> tuple[str, str]:
    """
    Try to extract an event date from the snippet text.

    Returns
    -------
    (event_date, date_confidence) where date_confidence is one of:
      "exact"       — explicit day+month found in text
      "inferred"    — relative reference found; pub_date used but flagged
      "publication" — no date found; pub_date used as fallback
    """
    m = _DATE_EXACT.search(snippet)
    if m:
        return m.group(0).strip(), "exact"
    if _DATE_RELATIVE.search(snippet):
        return pub_date, "inferred"
    return pub_date, "publication"


# ── Ice-flood vs summer-flood discrimination ───────────────────────────────────

# Strong positive evidence: ice co-occurs with flood
_ICE_FLOOD_COTERMS = re.compile(
    r'\b(jääpato|jäätulva|hyytö|ajojää|jäälohkare|jäämassa|jäiden\s+lähdöst[äa]'
    r'|isdamm|isflak|islossning)\b',
    re.IGNORECASE,
)

# Strong negative evidence: clearly a rain/melt flood
_SUMMER_FLOOD_TERMS = re.compile(
    r'\b(kesätulva|rankkasade|sadevedest[äa]|sulamisvedest[äa]'
    r'|lumen\s+sula|sommarflöde|regn(?:flöde|vatten))\b',
    re.IGNORECASE,
)

# Ice-relevant months for Finland: October–May (months 10–12, 1–5)
_ICE_MONTHS = {10, 11, 12, 1, 2, 3, 4, 5}


def _is_ice_flood(snippet: str, pub_date: str) -> bool:
    """
    Return True if a flood mention is likely ice-related.

    Priority:
      1. Explicit ice co-term → True
      2. Explicit summer-flood term → False
      3. Season from pub_date → True if Oct–May, False otherwise
      4. Unknown date → False (conservative)
    """
    if _ICE_FLOOD_COTERMS.search(snippet):
        return True
    if _SUMMER_FLOOD_TERMS.search(snippet):
        return False
    try:
        month = int(pub_date[5:7])
        return month in _ICE_MONTHS
    except (ValueError, IndexError, TypeError):
        return False


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class IceEvent:
    event_type:       str
    keywords:         list[str]
    snippet:          str
    river_hint:       str  = ""
    pub_date:         str  = ""   # publication date from filename (always set)
    event_date:       str  = ""   # extracted or inferred from text
    date_confidence:  str  = "publication"  # "exact" | "inferred" | "publication"


# ── River detection ────────────────────────────────────────────────────────────

def _detect_river(text: str) -> str:
    """Return the first matching target river name, or empty string."""
    lower = text.lower()
    for river in TARGET_RIVERS:
        if river in lower:
            return river
    return ""


# ── Core extractor ─────────────────────────────────────────────────────────────

def extract_ice_events(
    text: str,
    pub_date: str = "",
    snippet_window: int = 150,
) -> list[IceEvent]:
    """
    Scan text for ice event mentions.

    Parameters
    ----------
    text          : normalized article text from the pipeline
    pub_date      : publication date string "YYYY-MM-DD" from the filename
    snippet_window: characters of context around the matched keyword

    Returns
    -------
    List of IceEvent objects. One event per keyword hit (may have duplicates
    for the same event described multiple times — dedup upstream if needed).
    """
    events: list[IceEvent] = []
    lower = text.lower()

    for event_type, keywords in ICE_KEYWORDS.items():
        for kw in keywords:
            idx = lower.find(kw.lower())
            while idx != -1:
                # Build context snippet
                start = max(0, idx - snippet_window)
                end   = min(len(text), idx + len(kw) + snippet_window)
                snippet = text[start:end].strip()

                # Gate generic tulva/översvämning through ice-flood check
                if event_type == "jam_flood" and kw in ("tulva", "översvämning"):
                    if not _is_ice_flood(snippet, pub_date):
                        idx = lower.find(kw.lower(), idx + 1)
                        continue

                event_date, date_conf = _extract_event_date(snippet, pub_date)
                river = _detect_river(snippet)

                events.append(IceEvent(
                    event_type      = event_type,
                    keywords        = [kw],
                    snippet         = snippet,
                    river_hint      = river,
                    pub_date        = pub_date,
                    event_date      = event_date,
                    date_confidence = date_conf,
                ))

                idx = lower.find(kw.lower(), idx + 1)

    return events


def events_to_dicts(events: list[IceEvent]) -> list[dict]:
    """Serialize events to plain dicts for CSV/JSON output."""
    return [
        {
            "event_type":      e.event_type,
            "keywords":        "|".join(e.keywords),
            "river_hint":      e.river_hint,
            "pub_date":        e.pub_date,
            "event_date":      e.event_date,
            "date_confidence": e.date_confidence,
            "snippet":         e.snippet,
        }
        for e in events
    ]