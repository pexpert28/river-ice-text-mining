#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_repo.py
=============
Scaffolds the full directory and file structure for the
finnish-river-ice-nlp GitHub repository.

USAGE
-----
  python setup_repo.py                        # creates ./finnish-river-ice-nlp/
  python setup_repo.py --path /some/other/dir # custom location

Run once. Safe to re-run — existing files are never overwritten.
"""

import argparse
import os
import sys
from pathlib import Path

# ── File contents ─────────────────────────────────────────────────────────────

README = """\
# finnish-river-ice-nlp

Crowdsourcing and Text Mining Digitized Finnish Newspapers to Reconstruct River Ice Events.

This project combines crowdsourcing, NLP, and machine learning to extract historical
river ice observations (freeze dates, ice break-up, ice thickness) from Finnish
digitized newspapers (1820–1939), scoped to major Finnish rivers.

## Project Structure

```
finnish-river-ice-nlp/
├── src/
│   ├── fetching/        ← Download tools for digi.kansalliskirjasto.fi
│   ├── preprocessing/   ← ALTO XML parsing and text cleaning
│   ├── extraction/      ← NLP/ML ice event extraction
│   └── analysis/        ← Hydrological analysis of extracted events
├── notebooks/           ← Jupyter notebooks for exploration and evaluation
├── data/
│   ├── manifests/       ← Search result CSVs (tracked)
│   ├── raw/             ← Downloaded ALTO XML files (gitignored)
│   └── processed/       ← Cleaned text and extracted events (gitignored)
├── crowdsourcing/       ← Annotation guidelines and crowdsourcing materials
└── docs/                ← API notes and data source documentation
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/finnish-river-ice-nlp.git
cd finnish-river-ice-nlp
pip install -r requirements.txt
```

## Quickstart: Fetch Newspaper Texts

```bash
# Search for ice-related articles (1850–1939), no download yet:
python src/fetching/digi_fetcher.py --query "jää" --start 1850-01-01 --end 1939-12-31

# Download matching pages as plain text:
python src/fetching/digi_fetcher.py --query "jäätyminen" --start 1850-01-01 --end 1939-12-31 --download --format txt

# Explore which newspapers are available in a date range:
python src/fetching/explore_digi_newspapers.py --start 1850 --end 1939 --no-probe
```

## Team

| Role | Responsibilities |
|---|---|
| Hydrologist | Domain expertise, crowdsourcing management, hydrological analysis, thesis writing |
| ML Expert | Data pipeline, NLP/ML model development, technical infrastructure |

## Data Sources

- [digi.kansalliskirjasto.fi](https://digi.kansalliskirjasto.fi) — National Library of Finland digitized newspapers
- [api.finna.fi](https://api.finna.fi) — Finna metadata API

## License

MIT
"""

GITIGNORE = """\
# Downloaded data (can be large — regenerate with digi_fetcher.py)
data/raw/
data/processed/
digi_output/

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp

# Logs and temp
*.log
*.tmp
"""

REQUIREMENTS = """\
# Core
requests>=2.31.0
tqdm>=4.66.0

# Data handling
pandas>=2.0.0
lxml>=4.9.0

# NLP
transformers>=4.38.0
torch>=2.0.0

# Notebooks
jupyter>=1.0.0
ipykernel>=6.0.0

# Utilities
python-dotenv>=1.0.0
"""

LICENSE = """\
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# ── src/fetching/ ─────────────────────────────────────────────────────────────

FETCHING_INIT = """\
# fetching
# Tools for discovering and downloading newspaper texts from
# digi.kansalliskirjasto.fi
"""

DIGI_FETCHER_STUB = """\
# digi_fetcher.py
# Place the full digi_fetcher.py script here.
# Download it from the project root or copy from the shared drive.
#
# Usage:
#   python digi_fetcher.py --query "jää" --start 1850-01-01 --end 1939-12-31
#   python digi_fetcher.py --query "jäätyminen" --start 1850-01-01 --end 1939-12-31 --download --format txt
"""

EXPLORE_STUB = """\
# explore_digi_newspapers.py
# Place the full explore_digi_newspapers.py script here.
#
# Usage:
#   python explore_digi_newspapers.py --start 1850 --end 1939 --no-probe
"""

# ── src/preprocessing/ ────────────────────────────────────────────────────────

PREPROCESSING_INIT = """\
# preprocessing
# ALTO XML parsing and text cleaning utilities.
"""

ALTO_PARSER = """\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
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
\"\"\"

import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_alto_file(xml_path: str) -> str:
    \"\"\"
    Extract all word content from an ALTO XML file.
    Returns a single string of space-joined words.
    \"\"\"
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
    \"\"\"
    Basic cleaning for Finnish OCR text from 19th/early 20th century newspapers.

    Common OCR issues in this corpus:
      - Long hyphen sequences used as separators
      - Repeated punctuation from column borders
      - Stray single characters from OCR artifacts
    \"\"\"
    # Remove long separator lines (e.g. "--------" or "========")
    text = re.sub(r'[-=_]{3,}', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove lines that are only punctuation/symbols
    lines = [ln for ln in text.splitlines() if re.search(r'[a-zåäöA-ZÅÄÖ]', ln)]
    return "\\n".join(lines).strip()


def parse_alto_directory(directory: str,
                         clean: bool = True) -> dict:
    \"\"\"
    Parse all ALTO XML files in a directory (non-recursive).
    Returns a dict mapping filename → extracted text.
    \"\"\"
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
            print(f"\\n{'='*60}\\n{fname}\\n{'='*60}")
            print(text[:500] + ("..." if len(text) > 500 else ""))
    else:
        text = parse_alto_file(target)
        text = clean_ocr_text(text)
        print(text)
"""

# ── src/extraction/ ───────────────────────────────────────────────────────────

EXTRACTION_INIT = """\
# extraction
# NLP and ML components for extracting river ice events from newspaper text.
"""

ICE_EXTRACTOR = """\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
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
\"\"\"

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
    \"\"\"
    Scan text for ice event mentions.
    Returns a list of IceEvent objects.

    Parameters
    ----------
    text            : OCR text from one newspaper page
    snippet_window  : characters of context to include around a keyword match
    \"\"\"
    text_lower = text.lower()
    events     = []

    for event_type, keywords in ICE_KEYWORDS.items():
        for kw in keywords:
            for match in re.finditer(re.escape(kw), text_lower):
                start   = max(0, match.start() - snippet_window)
                end     = min(len(text), match.end() + snippet_window)
                snippet = text[start:end].replace("\\n", " ").strip()

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
    \"\"\"Convert IceEvent objects to plain dicts (for CSV export).\"\"\"
    return [
        {
            "event_type":  e.event_type,
            "keywords":    ", ".join(e.keywords),
            "river_hint":  e.river_hint,
            "snippet":     e.snippet,
        }
        for e in events
    ]
"""

# ── src/analysis/ ─────────────────────────────────────────────────────────────

ANALYSIS_INIT = """\
# analysis
# Hydrological analysis of extracted river ice events.
"""

HYDRO_ANALYSIS = """\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
hydro_analysis.py
=================
Analyse extracted river ice events for hydrological patterns.

Expected input: CSV produced by the extraction pipeline, with columns:
  binding_id, date, river_hint, event_type, snippet

Planned analyses:
  - Freeze / break-up date time series per river
  - Long-term trend detection (is freeze date shifting over decades?)
  - Comparison with instrumental records where available
\"\"\"

# TODO: implement after extraction pipeline is validated
# This module is the hydrologist student's primary working area.


def load_events(csv_path: str):
    \"\"\"Load extracted events from CSV into a pandas DataFrame.\"\"\"
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def freeze_dates_per_river(df):
    \"\"\"Return DataFrame of freeze events grouped by river and year.\"\"\"
    freeze = df[df["event_type"] == "freeze"].copy()
    freeze["year"] = freeze["date"].dt.year
    return freeze.groupby(["river_hint", "year"]).size().reset_index(name="count")


def breakup_dates_per_river(df):
    \"\"\"Return DataFrame of break-up events grouped by river and year.\"\"\"
    breakup = df[df["event_type"] == "breakup"].copy()
    breakup["year"] = breakup["date"].dt.year
    return breakup.groupby(["river_hint", "year"]).size().reset_index(name="count")
"""

# ── notebooks/ ────────────────────────────────────────────────────────────────

NB_EXPLORATION = """\
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 01 — Data Exploration\\n",
              "Explore which Finnish newspapers are available in the target date range\\n",
              "and how many issues contain ice-related keywords."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "sys.path.insert(0, '..')\\n",
    "\\n",
    "# Run the coverage explorer\\n",
    "# !python ../src/fetching/explore_digi_newspapers.py --start 1850 --end 1939 --no-probe\\n",
    "\\n",
    "import pandas as pd\\n",
    "# df = pd.read_csv('../data/manifests/newspaper_coverage.csv')\\n",
    "# df.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"""

NB_EXTRACTION = """\
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 02 — Text Extraction\\n",
              "Parse ALTO XML files and run the ice event extractor."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "sys.path.insert(0, '..')\\n",
    "\\n",
    "from src.preprocessing.alto_parser import parse_alto_file, clean_ocr_text\\n",
    "from src.extraction.ice_event_extractor import extract_ice_events, events_to_dicts\\n",
    "\\n",
    "# Example usage:\\n",
    "# text = parse_alto_file('../data/raw/issn/year/binding_page-00001.xml')\\n",
    "# text = clean_ocr_text(text)\\n",
    "# events = extract_ice_events(text)\\n",
    "# for e in events:\\n",
    "#     print(e)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"""

NB_EVALUATION = """\
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 03 — Model Evaluation\\n",
              "Evaluate Finnish NLP model performance on ice event extraction."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: load annotated examples from crowdsourcing\\n",
    "# Compare keyword-based extractor vs. Poro-34B or TurkuNLP model\\n",
    "print('Model evaluation notebook — to be implemented')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"""

# ── crowdsourcing/ ────────────────────────────────────────────────────────────

ANNOTATION_GUIDELINES = """\
# Annotation Guidelines — River Ice Events

## Task Overview

You will be shown short excerpts (snippets) from historical Finnish newspapers
(1820–1939). Your task is to decide whether each snippet describes a **river
ice event** and, if so, label it with the correct event type.

## Event Types

| Label | Finnish term | Description |
|---|---|---|
| `freeze` | jäätyminen | River freezes over or ice begins to form |
| `breakup` | jäänlähtö | Ice breaks up or leaves the river in spring |
| `thickness` | jään paksuus | Ice thickness is mentioned |
| `jam_flood` | jääpato / tulva | Ice jam causing flooding |
| `none` | — | Snippet does not describe a river ice event |

## Instructions

1. Read the snippet carefully.
2. Identify whether a **river** is mentioned. If yes, note the river name.
3. Assign one of the labels above.
4. If unsure, use the `uncertain` flag.

## Examples

**Snippet:**
> "Oulujoki jäätyi tänään 15. marraskuuta. Jää on jo noin 10 cm paksu."

**Label:** `freeze` | **River:** Oulujoki

---

**Snippet:**
> "Kemijoen jäänlähtö tapahtui viime viikolla aikaisemmin kuin tavallisesti."

**Label:** `breakup` | **River:** Kemijoki

---

**Snippet:**
> "Kaupungissa pidettiin eilen markkinat."

**Label:** `none`

## Quality Notes

- OCR quality varies. Some words may be misspelled due to scanning errors.
- Dates in 19th century Finnish newspapers often use the old calendar.
- Swedish-language newspapers use: *isläggning* (freeze), *islossning* (breakup).
"""

# ── docs/ ─────────────────────────────────────────────────────────────────────

API_NOTES = """\
# API Notes — digi.kansalliskirjasto.fi

## Key Endpoints

### Newspaper Titles
```
GET https://digi.kansalliskirjasto.fi/api/newspaper/titles?language=fi
```
Returns JSON list of all available newspaper titles with ISSN, date range, language.

### Binding Search (main search API)
```
GET https://digi.kansalliskirjasto.fi/api/dam/binding-search?query=jää&startDate=1880-01-01&endDate=1920-12-31&formats=NEWSPAPER
```
Returns JSON with matching newspaper issues (bindings). Paginated via `scrollId`.

Key response fields:
- `rows[].bindingId`       — unique issue ID
- `rows[].baseUrl`         — base URL for file downloads
- `rows[].date`            — issue date (YYYY-MM-DD)
- `rows[].publicationId`   — ISSN
- `rows[].pageNumber`      — page where keyword was found
- `rows[].textHighlights`  — keyword-in-context snippet
- `altoXmlTemplate`        — URL template, e.g. `/page-{{page}}.xml`
- `altoTxtTemplate`        — URL template, e.g. `/page-{{page}}.txt`
- `bindingPageCounts`      — {bindingId: total page count}
- `scrollId`               — cursor for next page of results

### ALTO XML (full OCR text)
```
GET https://digi.kansalliskirjasto.fi/sanomalehti/binding/{bindingId}/page-00001.xml
```

### METS Manifest (list all pages in an issue)
```
GET https://digi.kansalliskirjasto.fi/sanomalehti/binding/{bindingId}/mets
```

### Page Image
```
GET https://digi.kansalliskirjasto.fi/sanomalehti/binding/{bindingId}/image/1
```

### OAI-PMH (bulk metadata harvesting)
```
GET https://digi.kansalliskirjasto.fi/interfaces/OAI-PMH?verb=ListRecords&metadataPrefix=marc21&set=sanomalehti
```

## Copyright

Materials published before end of 1939 are openly available.
Swedish-language newspapers available until end of 1949.
"""

DATA_SOURCES = """\
# Data Sources

## Primary: digi.kansalliskirjasto.fi

National Library of Finland digitized newspaper collection.

- **Coverage:** Finnish newspapers 1771–1939 (openly accessible)
- **Format:** ALTO XML (OCR text), METS manifests, page images (JPG), PDFs
- **Access:** Free, no authentication required
- **API:** See `docs/api_notes.md`

## Secondary: api.finna.fi

Finna aggregates metadata from Finnish archives, libraries, and museums.

- **Use in this project:** Metadata discovery, cross-referencing ISSNs
- **Does NOT provide:** Full text content (use digi.kansalliskirjasto.fi for that)
- **Docs:** https://api.finna.fi/swagger-ui/

## Target Rivers

| River | Finnish name | Region |
|---|---|---|
| Oulu River | Oulujoki | Northern Ostrobothnia |
| Kemi River | Kemijoki | Lapland |
| Tornio River | Tornionjoki | Lapland / Swedish border |
| Kokemäki River | Kokemäenjoki | Satakunta |

## Reference / Validation Data

Instrumental river ice records (where available) for comparison with
newspaper-derived event dates:
- Finnish Environment Institute (SYKE): https://www.syke.fi
- Historical instrumental records from university archives
"""

# ── Manifest placeholder ──────────────────────────────────────────────────────

MANIFESTS_README = """\
# data/manifests/

This folder contains CSV manifest files produced by `digi_fetcher.py`.

Each manifest represents one search query and lists all matching newspaper
issues (bindings) with metadata: title, date, ISSN, binding ID, page count,
keyword highlights, and direct URL.

Manifests are lightweight and are tracked in git.
The actual downloaded ALTO XML files live in `data/raw/` and are gitignored.

## Naming convention

  {keyword}_{start}_{end}_manifest.csv

## Columns

| Column | Description |
|---|---|
| bindingId | Unique issue ID on digi.kansalliskirjasto.fi |
| bindingTitle | Newspaper name |
| publicationId | ISSN |
| date | Issue date (YYYY-MM-DD) |
| year | Year extracted from date |
| pageNumber | Page where keyword was found |
| pageCount | Total pages in the issue |
| textHighlights | Keyword-in-context snippet from the API |
| baseUrl | Base URL for file downloads |
| url | Direct link to the issue on digi.kansalliskirjasto.fi |
"""

# ── Scaffold logic ─────────────────────────────────────────────────────────────

def write_file(path: Path, content: str):
    """Write content to path. Skip if file already exists."""
    if path.exists():
        print(f"  [skip]  {path}  (already exists)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  [create] {path}")


def scaffold(root: Path):
    print(f"\nScaffolding repository at: {root.resolve()}\n")

    # Root files
    write_file(root / "README.md",        README)
    write_file(root / ".gitignore",       GITIGNORE)
    write_file(root / "requirements.txt", REQUIREMENTS)
    write_file(root / "LICENSE",          LICENSE)

    # src/fetching/
    write_file(root / "src" / "fetching" / "__init__.py",            FETCHING_INIT)
    write_file(root / "src" / "fetching" / "digi_fetcher.py",        DIGI_FETCHER_STUB)
    write_file(root / "src" / "fetching" / "explore_digi_newspapers.py", EXPLORE_STUB)

    # src/preprocessing/
    write_file(root / "src" / "preprocessing" / "__init__.py", PREPROCESSING_INIT)
    write_file(root / "src" / "preprocessing" / "alto_parser.py",    ALTO_PARSER)

    # src/extraction/
    write_file(root / "src" / "extraction" / "__init__.py",          EXTRACTION_INIT)
    write_file(root / "src" / "extraction" / "ice_event_extractor.py", ICE_EXTRACTOR)

    # src/analysis/
    write_file(root / "src" / "analysis" / "__init__.py",            ANALYSIS_INIT)
    write_file(root / "src" / "analysis" / "hydro_analysis.py",      HYDRO_ANALYSIS)

    # notebooks/
    write_file(root / "notebooks" / "01_data_exploration.ipynb",     NB_EXPLORATION)
    write_file(root / "notebooks" / "02_text_extraction.ipynb",      NB_EXTRACTION)
    write_file(root / "notebooks" / "03_model_evaluation.ipynb",     NB_EVALUATION)

    # data/
    write_file(root / "data" / "manifests" / "README.md",            MANIFESTS_README)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    print(f"  [create] {root / 'data' / 'raw'}  (empty, gitignored)")
    print(f"  [create] {root / 'data' / 'processed'}  (empty, gitignored)")

    # crowdsourcing/
    write_file(root / "crowdsourcing" / "annotation_guidelines.md",  ANNOTATION_GUIDELINES)

    # docs/
    write_file(root / "docs" / "api_notes.md",                       API_NOTES)
    write_file(root / "docs" / "data_sources.md",                    DATA_SOURCES)

    # Summary
    print()
    print("=" * 56)
    print("  Repository scaffold complete.")
    print("=" * 56)
    print()
    print("  Next steps:")
    print(f"  1. cd {root.name}")
    print(f"  2. Copy digi_fetcher.py and explore_digi_newspapers.py")
    print(f"     into src/fetching/")
    print(f"  3. git init && git add . && git commit -m 'Initial scaffold'")
    print(f"  4. Create repo on GitHub and push:")
    print(f"       git remote add origin https://github.com/YOUR_USERNAME/{root.name}.git")
    print(f"       git push -u origin main")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaffold the finnish-river-ice-nlp GitHub repository."
    )
    parser.add_argument(
        "--path", default=".",
        help="Parent directory where the repo folder will be created (default: current dir)"
    )
    parser.add_argument(
        "--name", default="finnish-river-ice-nlp",
        help="Repo folder name (default: finnish-river-ice-nlp)"
    )
    args = parser.parse_args()

    root = Path(args.path) / args.name
    scaffold(root)
