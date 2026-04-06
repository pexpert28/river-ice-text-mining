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
