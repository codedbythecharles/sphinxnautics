#!/usr/bin/env python3
import argparse, csv
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

def read_success_map(csv_path):
    m = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["problem_id"]
            tts = int(row["trials_to_success"])
            # keep the minimum if duplicates appear
            if pid not in m or tts < m[pid]:
                m[pid] = tts
    return m

def detect_id_field(ds: Dataset):
    if "problem_id" in ds.column_names:
        return "problem_id"
    if "id" in ds.column_names:
        return "id"
    raise ValueError("Neither 'problem_id' nor 'id' column found in the dataset.")

def filter_and_annotate(ds: Dataset, success_map, id_field, num_proc=8):
    # filter to successful ids
    ds = ds.filter(
        lambda ex: str(ex[id_field]) in success_map,
        num_proc=num_proc
    )
    # add trials_to_success
    def add_tts(ex):
        key = str(ex[id_field])
        return {"trials_to_success": success_map[key]}
    ds = ds.map(add_tts, num_proc=num_proc)
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="success_best_overall.csv (or success_per_file.csv)")
    ap.add_argument("--dataset_in", required=True, help="HF hub name or load_from_disk path")
    ap.add_argument("--split", default=None, help="split name if loading from hub (e.g. train)")
    ap.add_argument("--dataset_out", required=True, help="output path for save_to_disk")
    ap.add_argument("--num_proc", type=int, default=8)
    args = ap.parse_args()

    success_map = read_success_map(args.csv)
    print(f"Loaded {len(success_map)} successful problem IDs.")

    # load dataset
    if (args.dataset_in.endswith(".arrow")
        or args.dataset_in.endswith(".parquet")
        or args.dataset_in.endswith(".json")
        or args.dataset_in.endswith(".jsonl")):
        # user gave a file — load as a dataset
        ds = Dataset.from_file(args.dataset_in)
    else:
        # try load_from_disk first, then hub
        try:
            ds_any = load_from_disk(args.dataset_in)
        except Exception:
            if args.split is None:
                raise ValueError("When loading from the Hub, you must pass --split.")
            ds_any = load_dataset(args.dataset_in, split=args.split)

        if isinstance(ds_any, DatasetDict):
            if args.split is None:
                raise ValueError("DatasetDict detected. Please pass --split.")
            ds = ds_any[args.split]
        else:
            ds = ds_any

    id_field = detect_id_field(ds)
    print(f"Using id_field='{id_field}'")

    n_before = len(ds)
    ds = filter_and_annotate(ds, success_map, id_field, num_proc=args.num_proc)
    n_after = len(ds)
    print(f"Filtered: {n_before} → {n_after}")

    ds.save_to_disk(args.dataset_out)
    print(f"Saved to {args.dataset_out}")

if __name__ == "__main__":
    main()
