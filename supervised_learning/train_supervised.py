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
    if "__file__" in dir():
        here = os.path.dirname(os.path.abspath(__file__))  # type: ignore[name-defined]
        candidate = os.path.normpath(os.path.join(here, ".."))
        if os.path.isdir(os.path.join(candidate, "reinforcement_learning")):
            return candidate
    # Kaggle notebook cell (no __file__) or elsewhere: search common mount points
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
CHUNK_SIZE        = 50_000   # expected positions per full chunk
FIRST_N_SAVE_ALL  = 12       # checkpoint every chunk for the first N processed
SAVE_EVERY_MID    = 4        # then every Nth chunk until MID_UNTIL
MID_UNTIL         = 40       # switch to SAVE_EVERY after this many chunks
SAVE_EVERY        = 40       # thereafter, checkpoint every Nth chunk
VAL_CHUNKS        = 4        # hold out this many chunks for validation (0 = off)
SHUFFLE_SEED      = 42

# TEMPORARY: cap training to the first N chunks of the shuffled order per epoch
# (for quick test runs). Set to None to train on all chunks.
LIMIT_CHUNKS      = 200

CKPT_PREFIX  = "sl"
STATE_FILE   = os.path.join(OUT_DIR, "train_state.json")
METRICS_CSV  = os.path.join(OUT_DIR, "metrics.csv")

# Chunks identified as corrupted (game_rep_ratio >= 25%) by analyse_chunks.py.
# Caused by a resume-bug in process_pgn_data.py: chunk-boundary saves happened
# mid-file before _mark_done() was called, so those PGN files were reprocessed
# on resume and their positions duplicated.
EXCLUDED_CHUNKS: frozenset[str] = frozenset({
    "chunk_007.npz", "chunk_008.npz", "chunk_024.npz", "chunk_026.npz",
    "chunk_028.npz", "chunk_030.npz", "chunk_034.npz", "chunk_035.npz",
    "chunk_036.npz", "chunk_037.npz", "chunk_039.npz", "chunk_040.npz",
    "chunk_041.npz", "chunk_043.npz", "chunk_070.npz", "chunk_072.npz",
    "chunk_074.npz", "chunk_076.npz", "chunk_080.npz", "chunk_086.npz",
    "chunk_092.npz", "chunk_129.npz", "chunk_130.npz", "chunk_139.npz",
    "chunk_140.npz", "chunk_141.npz", "chunk_142.npz", "chunk_143.npz",
    "chunk_144.npz", "chunk_145.npz", "chunk_148.npz", "chunk_168.npz",
    "chunk_197.npz", "chunk_204.npz", "chunk_230.npz", "chunk_237.npz",
    "chunk_240.npz", "chunk_241.npz", "chunk_242.npz", "chunk_243.npz",
    "chunk_244.npz", "chunk_245.npz", "chunk_246.npz", "chunk_247.npz",
    "chunk_248.npz", "chunk_249.npz", "chunk_250.npz", "chunk_251.npz",
    "chunk_252.npz", "chunk_253.npz", "chunk_254.npz", "chunk_255.npz",
    "chunk_256.npz", "chunk_257.npz", "chunk_258.npz", "chunk_259.npz",
    "chunk_260.npz", "chunk_261.npz", "chunk_262.npz", "chunk_263.npz",
    "chunk_264.npz", "chunk_300.npz", "chunk_301.npz", "chunk_302.npz",
    "chunk_303.npz", "chunk_304.npz", "chunk_305.npz", "chunk_306.npz",
    "chunk_307.npz", "chunk_308.npz", "chunk_309.npz", "chunk_310.npz",
    "chunk_311.npz", "chunk_312.npz", "chunk_313.npz", "chunk_314.npz",
    "chunk_315.npz", "chunk_339.npz", "chunk_341.npz", "chunk_346.npz",
    "chunk_351.npz", "chunk_354.npz", "chunk_356.npz", "chunk_369.npz",
    "chunk_370.npz", "chunk_371.npz", "chunk_372.npz", "chunk_373.npz",
    "chunk_375.npz", "chunk_378.npz", "chunk_380.npz", "chunk_381.npz",
    "chunk_382.npz", "chunk_383.npz", "chunk_384.npz", "chunk_385.npz",
    "chunk_394.npz", "chunk_401.npz", "chunk_402.npz", "chunk_405.npz",
    "chunk_407.npz", "chunk_414.npz", "chunk_419.npz", "chunk_424.npz",
    "chunk_427.npz", "chunk_429.npz", "chunk_431.npz", "chunk_433.npz",
    "chunk_436.npz", "chunk_438.npz", "chunk_441.npz", "chunk_446.npz",
    "chunk_468.npz", "chunk_470.npz", "chunk_472.npz", "chunk_473.npz",
    "chunk_475.npz", "chunk_480.npz", "chunk_482.npz", "chunk_486.npz",
    "chunk_490.npz", "chunk_491.npz", "chunk_492.npz", "chunk_511.npz",
    "chunk_513.npz", "chunk_516.npz", "chunk_518.npz", "chunk_522.npz",
    "chunk_524.npz", "chunk_526.npz", "chunk_553.npz", "chunk_556.npz",
    "chunk_558.npz", "chunk_560.npz", "chunk_564.npz", "chunk_566.npz",
    "chunk_568.npz", "chunk_576.npz", "chunk_577.npz", "chunk_578.npz",
    "chunk_579.npz", "chunk_581.npz", "chunk_584.npz", "chunk_585.npz",
    "chunk_586.npz", "chunk_587.npz", "chunk_588.npz", "chunk_589.npz",
    "chunk_590.npz", "chunk_591.npz", "chunk_592.npz", "chunk_595.npz",
    "chunk_597.npz", "chunk_599.npz", "chunk_601.npz", "chunk_603.npz",
    "chunk_610.npz", "chunk_611.npz", "chunk_612.npz", "chunk_638.npz",
    "chunk_640.npz", "chunk_642.npz", "chunk_644.npz", "chunk_646.npz",
    "chunk_653.npz", "chunk_656.npz",
})

# Learning-rate schedule: reduce on plateau (val_policy_acc, checked every chunk).
LR_INIT       = 0.001  # starting LR — matches BigNetwork Adam default
LR_FACTOR     = 0.5    # multiply by this on each plateau
LR_PATIENCE   = 3      # consecutive chunks without improvement before reducing
LR_MIN        = 1e-9   # floor

BEST_CKPT_PATH = os.path.join(OUT_DIR, "sl_best.weights.h5")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_chunks() -> list[str]:
    files = glob.glob(os.path.join(DATA_DIR, "**", "chunk_*.npz"), recursive=True)
    files = [f for f in files if os.path.basename(f).startswith("chunk_")]
    return sorted(files)


def _chunk_size(path: str) -> int:
    # Read only the npy header inside the zip — no decompression of the data.
    import zipfile, struct, re
    with zipfile.ZipFile(path) as zf:
        with zf.open("boards.npy") as f:
            f.read(6)                          # magic \x93NUMPY
            major = f.read(1)[0]               # version major
            f.read(1)                          # version minor
            if major == 1:
                hlen = struct.unpack("<H", f.read(2))[0]
            else:                              # version 2+
                hlen = struct.unpack("<I", f.read(4))[0]
            header = f.read(hlen).decode()
    m = re.search(r"'shape'\s*:\s*\((\d+)", header) or re.search(r'"shape"\s*:\s*\((\d+)', header)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse shape from npy header in {path}: {header!r}")


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
    if global_chunk_count <= MID_UNTIL:
        return (global_chunk_count - FIRST_N_SAVE_ALL) % SAVE_EVERY_MID == 0
    return (global_chunk_count - MID_UNTIL) % SAVE_EVERY == 0


def _ckpt_path(global_chunk_count: int) -> str:
    return os.path.join(OUT_DIR, f"{CKPT_PREFIX}_{global_chunk_count:03d}.weights.h5")


def _save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _load_state() -> dict | None:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def _set_lr(model, lr: float) -> None:
    try:
        model.optimizer.learning_rate.assign(lr)
    except AttributeError:
        model.optimizer.learning_rate = lr


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

    n_before = len(all_chunks)
    all_chunks = [c for c in all_chunks if os.path.basename(c) not in EXCLUDED_CHUNKS]
    n_excluded = n_before - len(all_chunks)
    if n_excluded:
        print(f"  ({n_excluded} corrupted chunk(s) excluded by EXCLUDED_CHUNKS)")

    # Use only full-sized chunks (50k positions) for both train and val, so
    # undersized tail chunks (last chunk of each worker) don't skew either.
    full_chunks = [c for c in all_chunks if _chunk_size(c) >= CHUNK_SIZE]
    tail_chunks = [c for c in all_chunks if c not in set(full_chunks)]

    if VAL_CHUNKS > 0 and len(full_chunks) > VAL_CHUNKS:
        val_chunks   = full_chunks[-VAL_CHUNKS:]
        val_set      = set(val_chunks)
        train_chunks = [c for c in full_chunks if c not in val_set]
    else:
        train_chunks, val_chunks = full_chunks, []

    print(f"  ({len(tail_chunks)} undersized tail chunk(s) excluded from train and val)")

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
    current_lr       = LR_INIT
    best_val_acc     = 0.0
    lr_plateau_count = 0

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
            current_lr         = state.get("current_lr", LR_INIT)
            best_val_acc       = state.get("best_val_acc", 0.0)
            lr_plateau_count   = state.get("lr_plateau_count", 0)
            _set_lr(net.model, current_lr)
            print(f"Resumed from {os.path.basename(ckpt)} "
                  f"(epoch {start_epoch}, {resume_skip} chunks into epoch, "
                  f"{global_chunk_count} total, lr {current_lr:.2e})\n")

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
        "learning_rate", "seconds",
    ]

    for epoch in range(start_epoch, EPOCHS):
        # Per-epoch deterministic order: seeded by epoch alone, so it's identical
        # whether reached fresh or via resume (no dependence on call history).
        order = list(range(len(train_chunks)))
        random.Random(SHUFFLE_SEED + epoch).shuffle(order)

        # TEMPORARY: train on only the first LIMIT_CHUNKS of the shuffled order.
        if LIMIT_CHUNKS is not None:
            order = order[:LIMIT_CHUNKS]

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
                "learning_rate": current_lr,
                "seconds": round(time.time() - t0, 1),
            }

            del boards, policies, values

            # Always checkpoint the final chunk of an epoch so epoch boundaries
            # have clean resume state.
            is_last_in_epoch = pos_in_epoch == len(order) - 1
            do_ckpt = _should_checkpoint(global_chunk_count) or is_last_in_epoch

            # Validate every chunk — cheap on P100, gives clean signal for
            # ReduceLROnPlateau and lets us track the best model precisely.
            if val_data is not None:
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

                val_acc = float(row["val_policy_acc"])
                if val_acc > best_val_acc + 1e-4:
                    best_val_acc     = val_acc
                    lr_plateau_count = 0
                    net.save(BEST_CKPT_PATH)
                    print(f"    new best val pacc {val_acc:.5f} → sl_best.weights.h5",
                          flush=True)
                else:
                    lr_plateau_count += 1
                    if lr_plateau_count >= LR_PATIENCE:
                        old_lr = current_lr
                        new_lr = max(LR_MIN, old_lr * LR_FACTOR)
                        if new_lr < old_lr - 1e-12:
                            _set_lr(net.model, new_lr)
                            current_lr = new_lr
                            print(f"    LR {old_lr:.2e} → {new_lr:.2e}"
                                  f"  (plateau {lr_plateau_count} chunks)", flush=True)
                        lr_plateau_count = 0

                row["learning_rate"] = current_lr

            _append_metrics(row, metrics_header)

            msg = (f"[e{epoch} {pos_in_epoch+1}/{len(order)}] "
                   f"{row['chunk_file']}  loss {row['loss']}  "
                   f"pacc {row['policy_acc']}  ({row['seconds']}s)")
            if row["val_policy_acc"] != "":
                msg += f"  | val pacc {row['val_policy_acc']}  lr {current_lr:.1e}"
            print(msg, flush=True)

            if do_ckpt:
                ckpt_path = _ckpt_path(global_chunk_count)
                net.save(ckpt_path)
                # pos_in_epoch+1 = chunks completed in this epoch as of this checkpoint.
                # Next epoch resets to 0 (handled by start_epoch comparison on resume).
                _save_state({
                    "epoch":              epoch if not is_last_in_epoch else epoch + 1,
                    "pos_in_epoch":       (pos_in_epoch + 1) if not is_last_in_epoch else 0,
                    "global_chunk_count": global_chunk_count,
                    "last_checkpoint":    ckpt_path,
                    "current_lr":         current_lr,
                    "best_val_acc":       best_val_acc,
                    "lr_plateau_count":   lr_plateau_count,
                })
                print(f"    saved {os.path.basename(ckpt_path)}", flush=True)

    # Always save a final checkpoint
    final_path = os.path.join(OUT_DIR, "sl_final.weights.h5")
    net.save(final_path)
    print(f"\nDone. Final weights: {final_path}")


if __name__ == "__main__":
    main()
