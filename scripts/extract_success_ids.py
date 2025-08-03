#!/usr/bin/env python3
import argparse, json, csv
from pathlib import Path

def load_json_any(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # try normal JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    # try NDJSON
    items = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items

def first_success(samples):
    for idx, s in enumerate(samples):
        if s.get("success", False) is True:
            sid = s.get("sample_id")
            return (sid if isinstance(sid, int) and sid > 0 else idx + 1, idx)
    return (None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("."), help="directory containing *_with_reasoning.json")
    ap.add_argument("--out_per_file", type=Path, default=Path("success_per_file.csv"))
    ap.add_argument("--out_best", type=Path, default=Path("success_best_overall.csv"))
    args = ap.parse_args()

    input_dir: Path = args.dir
    paths = set()
    paths.update(input_dir.glob("*_with_reasoning.json"))
    paths.update(input_dir.glob("*_reasoning.json"))

    print(f"Searching in: {input_dir.resolve()}")
    print(f"Matched {len(paths)} files")

    rows = []
    files_scanned = 0
    problems_seen = 0
    successes_found = 0

    for path in sorted(paths):
        files_scanned += 1
        data = load_json_any(path)
        if not isinstance(data, list):
            continue
        for block in data:
            pid = block.get("problem_id")
            samples = block.get("samples", [])
            if pid is None or not isinstance(samples, list) or len(samples) == 0:
                continue
            problems_seen += 1
            trials_to_success, _ = first_success(samples)
            if trials_to_success is not None:
                successes_found += 1
                rows.append({
                    "file": path.name,
                    "problem_id": pid,
                    "trials_to_success": trials_to_success,
                    "total_samples": len(samples),
                })

    # write per-file successes
    with args.out_per_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "problem_id", "trials_to_success", "total_samples"])
        writer.writeheader()
        writer.writerows(rows)

    # best overall (min trials per problem)
    best = {}
    for r in rows:
        pid = r["problem_id"]
        if pid not in best or r["trials_to_success"] < best[pid]["trials_to_success"]:
            best[pid] = r
    with args.out_best.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "problem_id", "trials_to_success", "total_samples"])
        writer.writeheader()
        writer.writerows(best.values())

    print(f"Scanned files   : {files_scanned}")
    print(f"Problems seen   : {problems_seen}")
    print(f"Successes found : {successes_found}")
    print(f"Wrote {len(rows)} rows to {args.out_per_file}")
    print(f"Wrote {len(best)} best rows to {args.out_best}")

if __name__ == "__main__":
    main()
