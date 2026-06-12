import json
import csv
from pathlib import Path
import argparse

def json_dir_to_csv(json_dir, out_csv):
    rows = []

    for json_file in Path(json_dir).glob("**/*.json"):
        if json_file.name == "_summary.json":
            continue

        data = json.loads(json_file.read_text(encoding="utf-8"))

        for m in data["matches"]:
            rows.append({
            "source_file": data["source_file"],
            "file_year": data["file_year"],
            "publishing_date": m.get("publishing_date"),
            "sentence_index": m["sentence_index"],
            "sentence": m["sentence"],
            "normalized": m["normalized"],
            "ice_categories": ";".join(m["ice_categories"]),
            "ice_patterns": ";".join(m["ice_patterns"]),
            "rivers": ";".join(m["rivers"]),
            "year": m["year"],
            })


    # Write CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written → {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsondir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    Path(args.outdir).mkdir(exist_ok=True)

    out_csv = Path(args.outdir) / "ice_events.csv"
    json_dir_to_csv(args.jsondir, out_csv)
