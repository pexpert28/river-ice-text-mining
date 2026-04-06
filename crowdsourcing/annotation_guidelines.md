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
