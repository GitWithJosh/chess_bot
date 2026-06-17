"""
TEMPORARY fix: drop high-duplication chunks before uploading to Kaggle.

Reads chunk_analysis.json and copies every chunk with dup_ratio <= THRESHOLD
into a fresh output dir, renumbered contiguously (chunk_000.npz, ...).
Originals in processed_data/ are left untouched.

The high dup rate is suspected to come from repeated game segments in the
source PGNs (game_rep_ratio dominates dup_ratio; natural_dup is ~1%), so
filtering on dup_ratio is effectively the source-repeat filter.

Run from chess_bot root:
    python supervised_learning/filter_chunks.py
"""

import json
import os
import shutil

HERE      = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.join(HERE, "processed_data")
OUT_DIR   = os.path.join(HERE, "processed_data_filtered")
ANALYSIS  = os.path.join(HERE, "chunk_analysis.json")
THRESHOLD = 0.25   # drop chunks whose dup_ratio exceeds this

def main():
    with open(ANALYSIS) as f:
        analysis = json.load(f)

    kept = [r for r in analysis
            if r.get("error") is None and r.get("dup_ratio", 1.0) <= THRESHOLD]
    kept.sort(key=lambda r: r["file"])

    dropped = len(analysis) - len(kept)
    kept_pos = sum(r.get("n", 0) for r in kept)
    print(f"Chunks: {len(analysis)} total | {len(kept)} kept | {dropped} dropped "
          f"(dup_ratio > {THRESHOLD})")
    print(f"Kept positions: {kept_pos:,}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    for out_idx, r in enumerate(kept):
        src = os.path.join(SRC_DIR, r["file"])
        dst = os.path.join(OUT_DIR, f"chunk_{out_idx:03d}.npz")
        shutil.copy2(src, dst)
        if out_idx % 50 == 0 or out_idx == len(kept) - 1:
            print(f"  {out_idx + 1}/{len(kept)}  {r['file']} -> "
                  f"chunk_{out_idx:03d}.npz (dup={r['dup_ratio']:.1%})", flush=True)

    print(f"\nDone -> {OUT_DIR}/  ({len(kept)} chunks)")

if __name__ == "__main__":
    main()
