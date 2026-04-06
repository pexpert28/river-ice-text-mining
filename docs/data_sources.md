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
