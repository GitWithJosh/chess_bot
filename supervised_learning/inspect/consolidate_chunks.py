"""
Consolidate worker chunk files into uniform chunk_000.npz files.
Delete this script after use.

Run from chess_bot root:
    python supervised_learning/consolidate_chunks.py
"""

import glob
import os
import numpy as np

OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processed_data")
CHUNK_SIZE = 50_000

src_files = sorted(glob.glob(os.path.join(OUT_DIR, "chunk_src_*.npz")))
if not src_files:
    print("No worker chunk files found.")
    raise SystemExit

print(f"Found {len(src_files)} worker chunks to consolidate.\n")

boards_buf   = np.empty((CHUNK_SIZE, 8, 8, 20), dtype=np.float16)
policies_buf = np.empty((CHUNK_SIZE, 1858),       dtype=np.float32)
values_buf   = np.empty((CHUNK_SIZE, 3),           dtype=np.float32)
buf_idx  = 0
out_idx  = 0

def _save():
    global buf_idx, out_idx
    path = os.path.join(OUT_DIR, f"chunk_{out_idx:03d}.npz")
    np.savez_compressed(
        path,
        boards=boards_buf[:buf_idx],
        policies=policies_buf[:buf_idx],
        values=values_buf[:buf_idx],
    )
    print(f"  chunk_{out_idx:03d}.npz  {buf_idx:,} positions")
    out_idx += 1
    buf_idx = 0

for src in src_files:
    data = np.load(src)
    n = len(data["boards"])
    i = 0
    while i < n:
        space = CHUNK_SIZE - buf_idx
        take  = min(space, n - i)
        boards_buf[buf_idx:buf_idx + take]   = data["boards"][i:i + take]
        policies_buf[buf_idx:buf_idx + take] = data["policies"][i:i + take]
        values_buf[buf_idx:buf_idx + take]   = data["values"][i:i + take]
        buf_idx += take
        i       += take
        if buf_idx >= CHUNK_SIZE:
            _save()
    data.close()
    os.remove(src)
    print(f"  removed {os.path.basename(src)}", flush=True)

if buf_idx:
    _save()

print(f"\nDone — {out_idx} chunk file(s) in {OUT_DIR}/")
