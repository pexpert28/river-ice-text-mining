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
