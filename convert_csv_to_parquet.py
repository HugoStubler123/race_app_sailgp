#!/usr/bin/env python3
"""
Convert all CSV log files in logs/ to Parquet format.
Parquet files are ~5-10x smaller and ~10-50x faster to load than CSV.

Usage:
    python3 convert_csv_to_parquet.py          # convert only
    python3 convert_csv_to_parquet.py --push    # convert + git add/commit/push
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


LOGS_DIR = Path(__file__).parent / "logs"


def convert_all():
    csv_files = sorted(LOGS_DIR.glob("log_*.csv"))
    if not csv_files:
        print("No CSV files found in logs/")
        return []

    converted = []
    for csv_path in csv_files:
        parquet_path = csv_path.with_suffix(".parquet")
        if parquet_path.exists():
            csv_size = csv_path.stat().st_size
            pq_size = parquet_path.stat().st_size
            print(f"  [skip] {parquet_path.name} already exists ({pq_size/1e6:.1f} MB vs CSV {csv_size/1e6:.1f} MB)")
            converted.append(parquet_path)
            continue

        print(f"  Converting {csv_path.name} ...", end=" ", flush=True)
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
        csv_size = csv_path.stat().st_size
        pq_size = parquet_path.stat().st_size
        ratio = csv_size / pq_size if pq_size > 0 else 0
        print(f"OK  ({pq_size/1e6:.1f} MB, {ratio:.1f}x smaller)")
        converted.append(parquet_path)

    return converted


def git_push(parquet_files):
    """Stage parquet files, commit, and push."""
    repo_root = Path(__file__).parent

    # Stage parquet files
    paths = [str(p.relative_to(repo_root)) for p in parquet_files]
    print(f"\nStaging {len(paths)} parquet file(s)...")
    subprocess.run(["git", "add"] + paths, cwd=repo_root, check=True)

    # Commit
    msg = f"Add parquet log files ({len(paths)} files)"
    print(f"Committing: {msg}")
    result = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=repo_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        if "nothing to commit" in result.stdout:
            print("  Nothing new to commit.")
            return
        print(f"  Commit failed: {result.stderr}")
        return

    # Push
    print("Pushing to remote...")
    subprocess.run(["git", "push"], cwd=repo_root, check=True)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV logs to Parquet")
    parser.add_argument("--push", action="store_true", help="Git add + commit + push after conversion")
    args = parser.parse_args()

    print(f"Scanning {LOGS_DIR} for CSV files...\n")
    converted = convert_all()

    if not converted:
        print("\nNo files to process.")
        return

    print(f"\n{len(converted)} parquet file(s) ready.")

    if args.push:
        git_push(converted)
    else:
        print("\nRun with --push to also git add/commit/push the parquet files.")


if __name__ == "__main__":
    main()
