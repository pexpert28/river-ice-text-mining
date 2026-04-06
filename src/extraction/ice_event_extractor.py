#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ice_event_extractor.py
======================
Extract river ice events from Finnish newspaper text using
keyword matching and (optionally) a Finnish language model.

Ice event types targeted:
  - Freeze date       (jäätyminen, jää peitti, jäätyi)
  - Ice break-up      (jäänlähtö, jää lähti, jää meni)
  - Ice thickness     (jään paksuus, X cm jäätä)
  - Ice jam / flood   (jääpatoja, tulva)

USAGE
-----
  from src.extraction.ice_event_extractor import extract_ice_events

  events = extract_ice_events("Oulujoki jäätyi tänään 15. marraskuuta.")
  # → [{"type": "freeze", "snippet": "Oulujoki jäätyi...", "keywords": ["jäätyi"]}]
"""

import re
from dataclasses import dataclass, field


# ── Keyword dictionaries (extend as needed) ───────────────────────────────────

ICE_KEYWORDS = {
    "freeze": [
        "jäätyi", "jäätyminen", "jäätynyt", "jää peitti", "jää muodostui",
        "jää tuli", "alkoi jäätyä", "kiintojää",
    ],
    "breakup": [
        "jäänlähtö", "jää lähti", "jää meni", "jää suli", "jäät lähtivät",
        "jäät menivät", "jään lähtö", "jäät sulava",
    ],
    "thickness": [
        "jään paksuus", "paksu jää", "cm jäätä", "cm:n jää",
        "tuuman jäätä",
    ],
    "jam_flood": [
        "jääpato", "jääpatoja", "tulva", "jäätulva", "hyytö",
    ],
}

# Target rivers for the project (used to flag relevant mentions)
TARGET_RIVERS = [
    "oulujoki", "kemijoki", "tornionjoki", "kokemäenjoki",
    "oulu", "kemi", "tornio", "kokemäki",
]


@dataclass
class IceEvent:
    event_type:  str
    keywords:    list
    snippet:     str
    river_hint:  str  = ""    # detected river name, if any


def extract_ice_events(text: str,
                       snippet_window: int = 150) -> list[IceEvent]:
    """
    Scan text for ice event mentions.
    Returns a list of IceEvent objects.

    Parameters
    ----------
    text            : OCR text from one newspaper page
    snippet_window  : characters of context to include around a keyword match
    """
    text_lower = text.lower()
    events     = []

    for event_type, keywords in ICE_KEYWORDS.items():
        for kw in keywords:
            for match in re.finditer(re.escape(kw), text_lower):
                start   = max(0, match.start() - snippet_window)
                end     = min(len(text), match.end() + snippet_window)
                snippet = text[start:end].replace("\n", " ").strip()

                # Check if a target river name appears in the snippet
                river_hint = ""
                snippet_lower = snippet.lower()
                for river in TARGET_RIVERS:
                    if river in snippet_lower:
                        river_hint = river
                        break

                events.append(IceEvent(
                    event_type = event_type,
                    keywords   = [kw],
                    snippet    = snippet,
                    river_hint = river_hint,
                ))

    return events


def events_to_dicts(events: list[IceEvent]) -> list[dict]:
    """Convert IceEvent objects to plain dicts (for CSV export)."""
    return [
        {
            "event_type":  e.event_type,
            "keywords":    ", ".join(e.keywords),
            "river_hint":  e.river_hint,
            "snippet":     e.snippet,
        }
        for e in events
    ]
