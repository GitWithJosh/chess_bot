"""
Supervised training of BigNetwork on processed lichess data.

Uses the SAME architecture/config as the RL side (imports BigNetwork from
reinforcement_learning/networks/big_network.py) so the SL-vs-RL comparison is
apples-to-apples.

Data: supervised_learning/processed_data/chunk_*.npz, each containing
    boards   float16 (N, 8, 8, 112)
    policies float32 (N, 1858)   one-hot move
    values   float32 (N, 3)      one-hot WDL
Trains by calling net.model.fit() directly (NOT net.train(), which would
re-encode the already-final WDL targets).

Runs locally or on Kaggle (auto-detects /kaggle paths). Resumable across
sessions: re-running picks up from the latest checkpoint + state file.

    python supervised_learning/train_supervised.py
"""

import csv
import glob
import json
import os
import random
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make BigNetwork importable both locally and on Kaggle
# ---------------------------------------------------------------------------

def _find_repo_root() -> str:
    """Locate the chess_bot repo root (the dir containing reinforcement_learning/)."""
    # Local run: this file is at <root>/supervised_learning/train_supervised.py
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, ".."))
    if os.path.isdir(os.path.join(candidate, "reinforcement_learning")):
        return candidate
    # Kaggle (or elsewhere): search common mount points for the package
    for base in ("/kaggle/input", "/kaggle/working", os.getcwd()):
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if "reinforcement_learning" in dirnames:
                return dirpath
    raise RuntimeError(
        "Could not locate the chess_bot repo root (the folder containing "
        "'reinforcement_learning/'). Set REPO_ROOT manually."
    )


REPO_ROOT = _find_repo_root()
# big_network.py uses `from monte_carlo_tree_search...` / `from helpers...`,
# which resolve relative to the reinforcement_learning/ package dir.
RL_DIR = os.path.join(REPO_ROOT, "reinforcement_learning")
for p in (REPO_ROOT, RL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
# Converter (pulled in transitively) opens "reinforcement_learning/move_lookup.json"
# relative to CWD — only matters if instantiated, but chdir keeps things safe.
os.chdir(REPO_ROOT)

from networks.big_network import BigNetwork  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ON_KAGGLE = os.path.isdir("/kaggle")

if ON_KAGGLE:
    # Kaggle: chunks attached as a dataset under /kaggle/input/<name>/.
    # Auto-pick the first input dir that contains chunk_*.npz.
    _candidates = glob.glob("/kaggle/input/*/")
    DATA_DIR = next(
        (d for d in _candidates if glob.glob(os.path.join(d, "**", "chunk_*.npz"), recursive=True)),
        "/kaggle/input",
    )
    OUT_DIR = "/kaggle/working/checkpoints"
else:
    DATA_DIR = os.path.join(REPO_ROOT, "supervised_learning", "processed_data")
    OUT_DIR  = os.path.join(REPO_ROOT, "supervised_learning", "checkpoints")

EPOCHS            = 1        # passes over the full dataset
BATCH_SIZE        = 512
FIRST_N_SAVE_ALL  = 10       # checkpoint every chunk for the first N processed
SAVE_EVERY        = 5        # thereafter, checkpoint every Nth chunk
VAL_CHUNKS        = 2        # hold out this many chunks for validation (0 = off)
SHUFFLE_SEED      = 42

CKPT_PREFIX  = "sl_big"
STATE_FILE   = os.path.join(OUT_DIR, "train_state.json")
METRICS_CSV  = os.path.join(OUT_DIR, "metrics.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_chunks() -> list[str]:
    files = glob.glob(os.path.join(DATA_DIR, "**", "chunk_*.npz"), recursive=True)
    files = [f for f in files if os.path.basename(f).startswith("chunk_")]
    return sorted(files)


def _load_chunk(path: str):
    with np.load(path) as d:
        boards   = d["boards"].astype(np.float32)   # Keras wants float32 input
        policies = d["policies"]
        values   = d["values"]
    return boards, policies, values


def _should_checkpoint(global_chunk_count: int) -> bool:
    """global_chunk_count = number of TRAIN chunks fitted so far (1-based)."""
    if global_chunk_count <= FIRST_N_SAVE_ALL:
        return True
    return global_chunk_count % SAVE_EVERY == 0


def _ckpt_path(global_chunk_count: int) -> str:
    return os.path.join(OUT_DIR, f"{CKPT_PREFIX}_{global_chunk_count:05d}.weights.h5")


def _save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _load_state() -> dict | None:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def _append_metrics(row: dict, header: list[str]) -> None:
    exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    all_chunks = _list_chunks()
    if not all_chunks:
        print(f"ERROR: no chunk_*.npz found under {DATA_DIR}")
        sys.exit(1)

    # Split off a fixed validation set (last VAL_CHUNKS files, deterministic)
    if VAL_CHUNKS > 0 and len(all_chunks) > VAL_CHUNKS:
        train_chunks = all_chunks[:-VAL_CHUNKS]
        val_chunks   = all_chunks[-VAL_CHUNKS:]
    else:
        train_chunks, val_chunks = all_chunks, []

    print(f"=== Supervised Training ===")
    print(f"Repo root : {REPO_ROOT}")
    print(f"Data dir  : {DATA_DIR}")
    print(f"Out dir   : {OUT_DIR}")
    print(f"Chunks    : {len(train_chunks)} train | {len(val_chunks)} val")
    print(f"Epochs    : {EPOCHS} | batch {BATCH_SIZE}\n")

    # --- Build model (same config as RL side) ---
    net = BigNetwork()
    # Add a policy top-1 accuracy metric for monitoring move-match.
    # Safe to recompile here: no training has happened, so optimizer state is empty.
    net.model.compile(
        optimizer=net.model.optimizer,
        loss={
            "policy_output": "categorical_crossentropy",
            "value_output":  "categorical_crossentropy",
        },
        metrics={"policy_output": "accuracy"},
    )

    # --- Resume (restarts from the last CHECKPOINT, not the last fitted chunk,
    #     so no training is silently lost between checkpoint boundaries) ---
    state = _load_state()
    start_epoch = 0
    resume_skip = 0            # chunks to skip within start_epoch's shuffled order
    global_chunk_count = 0
    if state is not None:
        ckpt = state.get("last_checkpoint")
        if ckpt and os.path.exists(ckpt):
            net.load(ckpt)
            start_epoch        = state["epoch"]
            resume_skip        = state["pos_in_epoch"]
            global_chunk_count = state["global_chunk_count"]
            print(f"Resumed from {os.path.basename(ckpt)} "
                  f"(epoch {start_epoch}, {resume_skip} chunks into epoch, "
                  f"{global_chunk_count} total)\n")

    # --- Preload validation set once (small) ---
    val_data = None
    if val_chunks:
        vb, vp, vv = [], [], []
        for vc in val_chunks:
            b, p, v = _load_chunk(vc)
            vb.append(b); vp.append(p); vv.append(v)
        val_data = (
            np.concatenate(vb),
            {"policy_output": np.concatenate(vp), "value_output": np.concatenate(vv)},
        )
        del vb, vp, vv

    metrics_header = [
        "epoch", "global_chunk", "chunk_file", "n_pos",
        "loss", "policy_loss", "value_loss", "policy_acc",
        "val_loss", "val_policy_loss", "val_value_loss", "val_policy_acc",
        "seconds",
    ]

    for epoch in range(start_epoch, EPOCHS):
        # Per-epoch deterministic order: seeded by epoch alone, so it's identical
        # whether reached fresh or via resume (no dependence on call history).
        order = list(range(len(train_chunks)))
        random.Random(SHUFFLE_SEED + epoch).shuffle(order)

        skip = resume_skip if epoch == start_epoch else 0

        for pos_in_epoch, ci in enumerate(order):
            if pos_in_epoch < skip:
                continue   # already trained + checkpointed in a previous session

            chunk_path = train_chunks[ci]
            t0 = time.time()

            boards, policies, values = _load_chunk(chunk_path)

            hist = net.model.fit(
                boards,
                {"policy_output": policies, "value_output": values},
                batch_size=BATCH_SIZE,
                epochs=1,
                shuffle=True,          # shuffle WITHIN the chunk
                verbose=0,  # type: ignore[arg-type]  # keras stubless: infers str from "auto" default
            )
            global_chunk_count += 1
            h = hist.history

            row = {
                "epoch": epoch,
                "global_chunk": global_chunk_count,
                "chunk_file": os.path.basename(chunk_path),
                "n_pos": len(boards),
                "loss":        round(h["loss"][-1], 5),
                "policy_loss": round(h["policy_output_loss"][-1], 5),
                "value_loss":  round(h["value_output_loss"][-1], 5),
                "policy_acc":  round(h["policy_output_accuracy"][-1], 5),
                "val_loss": "", "val_policy_loss": "",
                "val_value_loss": "", "val_policy_acc": "",
                "seconds": round(time.time() - t0, 1),
            }

            del boards, policies, values

            # Always checkpoint the final chunk of an epoch so epoch boundaries
            # have clean resume state.
            is_last_in_epoch = pos_in_epoch == len(order) - 1
            do_ckpt = _should_checkpoint(global_chunk_count) or is_last_in_epoch

            # Validate at checkpoint boundaries (keeps val cost bounded)
            if do_ckpt and val_data is not None:
                # return_dict=True is required: in Keras 3, evaluate's metrics_names
                # lumps per-output accuracy under a "compile_metrics" placeholder,
                # so positional unpacking would drop val_policy_acc.
                vmap = net.model.evaluate(
                    val_data[0], val_data[1], batch_size=BATCH_SIZE,
                    verbose=0, return_dict=True,  # type: ignore[arg-type]
                )
                row["val_loss"]        = round(vmap.get("loss", 0.0), 5)
                row["val_policy_loss"] = round(vmap.get("policy_output_loss", 0.0), 5)
                row["val_value_loss"]  = round(vmap.get("value_output_loss", 0.0), 5)
                row["val_policy_acc"]  = round(vmap.get("policy_output_accuracy", 0.0), 5)

            _append_metrics(row, metrics_header)

            msg = (f"[e{epoch} {pos_in_epoch+1}/{len(order)}] "
                   f"{row['chunk_file']}  loss {row['loss']}  "
                   f"pacc {row['policy_acc']}  ({row['seconds']}s)")
            if row["val_policy_acc"] != "":
                msg += f"  | val pacc {row['val_policy_acc']}"
            print(msg, flush=True)

            if do_ckpt:
                ckpt_path = _ckpt_path(global_chunk_count)
                net.save(ckpt_path)
                # pos_in_epoch+1 = chunks completed in this epoch as of this checkpoint.
                # Next epoch resets to 0 (handled by start_epoch comparison on resume).
                _save_state({
                    "epoch": epoch if not is_last_in_epoch else epoch + 1,
                    "pos_in_epoch": (pos_in_epoch + 1) if not is_last_in_epoch else 0,
                    "global_chunk_count": global_chunk_count,
                    "last_checkpoint": ckpt_path,
                })
                print(f"    saved {os.path.basename(ckpt_path)}", flush=True)

    # Always save a final checkpoint
    final_path = os.path.join(OUT_DIR, f"{CKPT_PREFIX}_final.weights.h5")
    net.save(final_path)
    print(f"\nDone. Final weights: {final_path}")


if __name__ == "__main__":
    main()
