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
