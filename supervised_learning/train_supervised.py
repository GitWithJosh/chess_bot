"""
Supervised training of BigNetwork on processed lichess data.

Uses the SAME architecture/config as the RL side (imports BigNetwork from
reinforcement_learning/networks/big_network.py) so the SL-vs-RL comparison is
apples-to-apples.

Data: supervised_learning/processed_data/chunk_*.npz, each containing
    boards   float16 (N, 8, 8, 20)
    policies float32 (N, 1858)   one-hot move; ALL-ZERO for value-only rows
                                 (drawn tablebase positions) -> 0 policy loss,
                                 and policies.sum(1) doubles as the per-sample
                                 policy weight so metrics skip those rows
    values   float32 (N, 3)      one-hot WDL
    sources  uint8   (N,)        per-sample source id (newer chunks only; see
                                 SOURCE_NAMES) — enables per-source val metrics
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

import keras  # noqa: E402  (pulled in transitively by BigNetwork; needed for the loss below)

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
VAL_CHUNKS        = 2        # hold out this many chunks for validation (0 = off); 2 x 50k = 100k
SHUFFLE_SEED      = 42

# TEMPORARY: cap training to the first N chunks of the shuffled order per epoch
# (for quick test runs). Set to None to train on all chunks.
LIMIT_CHUNKS      = None

CKPT_PREFIX  = "sl"
STATE_FILE   = os.path.join(OUT_DIR, "train_state.json")
METRICS_CSV  = os.path.join(OUT_DIR, "metrics.csv")

# Learning-rate schedule: fixed lc0/AlphaZero-style steps over training progress
# (fraction of all planned chunk-fits; derived from global_chunk_count, so it is
# resume-safe). Replaces the earlier ReduceLROnPlateau logic, which fired on
# val-loss noise (patience 10 chunks vs ±0.02 per-chunk eval noise): the 2026-07
# run had collapsed to lr<=6.25e-6 by chunk 172/398, so the last half of the
# data trained at an effectively zero LR.
LR_SCHEDULE = (
    (0.0, 1e-4),   # base LR
    (0.7, 1e-5),   # /10 after 70% of the planned chunks
    (0.9, 1e-6),   # /10 again after 90%
)

# Decoupled weight decay (AdamW). AlphaZero used L2 = 1e-4; AdamW applies it
# as decoupled decay (scaled by LR each step), which is the cleaner formulation.
WEIGHT_DECAY  = 1e-4

BEST_CKPT_PATH = os.path.join(OUT_DIR, "sl_best.weights.h5")

# Per-sample source ids in the chunks' "sources" array. MUST stay in sync with
# dataset_common.SOURCE_IDS (duplicated here so this script stays standalone on
# Kaggle). Defender-side puzzle rows count as "puzzle", drawn tablebase rows as
# "tablebase". Older chunks have no "sources" array -> per-source metrics skipped.
SOURCE_NAMES = {0: "game", 1: "puzzle", 2: "tablebase"}

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
        sources  = d["sources"] if "sources" in d.files else None
    return boards, policies, values, sources


def _targets_and_weights(policies: np.ndarray, values: np.ndarray):
    """(y, sample_weight) for fit, in MODEL-OUTPUT ORDER (lists).

    Keras 3 constraints (verified locally): sample_weight must mirror y's
    structure AND resolve against the model's OUTPUT structure. BigNetwork
    builds its model with LIST outputs [policy, value], so both y and
    sample_weight must be lists here — dict forms fail (a partial dict raises
    ValueError, a full dict fails with KeyError: 0 during path resolution).

    Policy weight = policies.sum(1): value-only rows (drawn tablebase
    positions, all-zero policy target) get 0, one-hot rows get 1. Their policy
    CE and gradient are already exactly 0; the weight additionally keeps them
    out of the policy ACCURACY metric (weighted metrics divide by sum(w)).
    The reported policy LOSS still averages over all rows (Keras divides the
    weighted sum by the row count, not by sum(w)), so with 833 value-only rows
    per 50k chunk it reads ~1.7% below the true per-one-hot-row CE — a
    constant reporting dilution, not a training effect. The value head gets
    all-ones (identical to unweighted).
    """
    psw = policies.sum(axis=1).astype(np.float32)
    y = [policies, values]
    sw = [psw, np.ones(len(psw), dtype=np.float32)]
    return y, sw


def _val_metrics(net, vx, vpol, vval, batch_size):
    """One batched forward pass over the val set -> per-sample metric arrays.

    Replaces model.evaluate for validation: evaluate only returns whole-set
    aggregates, so per-source tracking would cost one extra evaluate per
    source — re-running the very same forward passes on each slice. A single
    predict pass + numpy metrics gives the aggregate AND any slicing for one
    pass total. Formulas replicate Keras: policy CE from logits via a stable
    log-softmax, value CE with 1e-7 clipping.
    """
    n = len(vx)
    ptgt = vpol.argmax(axis=1)      # zero-policy rows -> index 0; masked by weight
    vtgt = vval.argmax(axis=1)
    ploss  = np.zeros(n, np.float32)
    pmatch = np.zeros(n, np.float32)
    vloss  = np.zeros(n, np.float32)
    vmatch = np.zeros(n, np.float32)
    for i in range(0, n, batch_size):
        logits, probs = net.predict_batch(vx[i:i + batch_size])
        k = len(logits)
        rows = np.arange(k)
        mx = logits.max(axis=1)
        lse = mx + np.log(np.exp(logits - mx[:, None]).sum(axis=1))
        t = ptgt[i:i + k]
        ploss[i:i + k]  = lse - logits[rows, t]
        pmatch[i:i + k] = logits.argmax(axis=1) == t
        pv = np.clip(probs[rows, vtgt[i:i + k]], 1e-7, 1.0)
        vloss[i:i + k]  = -np.log(pv)
        vmatch[i:i + k] = probs.argmax(axis=1) == vtgt[i:i + k]
    return ploss, pmatch, vloss, vmatch


def _agg_metrics(per_sample, pweight, mask=None) -> dict:
    """Aggregate _val_metrics arrays over an optional boolean mask.

    Semantics (verified against model.evaluate on a real chunk): "loss" and
    the accuracies match Keras to float noise. The per-output LOSSES here are
    the exact sample-weighted means (weighted sum / row count, so value-only
    rows dilute the policy loss like they do in fit); Keras's reported
    per-output loss trackers instead average per-batch means, which makes them
    depend on batch packing — they differ from these exact values by <0.5%.
    The policy accuracy divides by sum(weights) (like weighted_metrics), so
    value-only rows are excluded from it.
    """
    ploss, pmatch, vloss, vmatch = per_sample
    pw = pweight
    if mask is not None:
        ploss, pmatch, vloss, vmatch, pw = (
            a[mask] for a in (ploss, pmatch, vloss, vmatch, pw))
    out = {
        "policy_loss": float((ploss * pw).sum() / max(len(vloss), 1)),
        "policy_acc":  float((pmatch * pw).sum() / max(pw.sum(), 1e-9)),
        "value_loss":  float(vloss.mean()) if len(vloss) else 0.0,
        "value_acc":   float(vmatch.mean()) if len(vmatch) else 0.0,
    }
    out["loss"] = out["policy_loss"] + out["value_loss"]
    return out


def _scheduled_lr(chunks_done: int, total_planned: int) -> float:
    """Step LR from LR_SCHEDULE based on the fraction of planned chunks done."""
    frac = chunks_done / max(1, total_planned)
    lr = LR_SCHEDULE[0][1]
    for threshold, value in LR_SCHEDULE:
        if frac >= threshold:
            lr = value
    return lr


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

    n_per_epoch = len(train_chunks) if LIMIT_CHUNKS is None else min(LIMIT_CHUNKS, len(train_chunks))
    total_planned = EPOCHS * n_per_epoch

    print(f"  ({len(tail_chunks)} undersized tail chunk(s) excluded from train and val)")

    print(f"=== Supervised Training ===")
    print(f"Repo root : {REPO_ROOT}")
    print(f"Data dir  : {DATA_DIR}")
    print(f"Out dir   : {OUT_DIR}")
    print(f"Chunks    : {len(train_chunks)} train | {len(val_chunks)} val")
    print(f"Epochs    : {EPOCHS} | batch {BATCH_SIZE}")
    print(f"Optimizer : AdamW (weight_decay {WEIGHT_DECAY}, kernels only)")
    print(f"LR steps  : " + "  ".join(f"{int(f*100)}%->{lr:.0e}" for f, lr in LR_SCHEDULE)
          + f"  (of {total_planned} planned chunks)")
    print(f"Per-chunk metrics are shown as train/val.\n")

    # --- Build model (same config as RL side) ---
    net = BigNetwork()
    # Recompile to (a) add top-1 accuracy metrics for both heads and (b) swap
    # the optimizer to AdamW with decoupled weight decay (BigNetwork builds
    # with plain Adam, no regularisation). Safe here: no training has happened
    # yet, so there's no optimizer state to discard.
    optimizer = keras.optimizers.AdamW(learning_rate=LR_SCHEDULE[0][1], weight_decay=WEIGHT_DECAY)
    # Decay only the conv/dense kernels: pulling BN params and biases toward
    # zero is standard to avoid (they don't contribute to capacity the way
    # kernels do). Patterns are matched against variable paths like
    # "se_block_0_bn1/gamma" and "policy_conv2/bias".
    optimizer.exclude_from_weight_decay(var_names=["_bn", "/bias"])
    net.model.compile(
        optimizer=optimizer,
        loss={
            # MUST match BigNetwork._compile: the policy head emits RAW LOGITS, so
            # policy loss is from_logits=True. A plain "categorical_crossentropy"
            # here would silently mis-train (treats logits as probabilities → NaNs).
            "policy_output": keras.losses.CategoricalCrossentropy(from_logits=True),
            "value_output":  "categorical_crossentropy",
        },
        # Top-1 match for both heads, as WEIGHTED metrics (verified locally):
        # Keras 3 applies sample_weight to losses only; metrics passed via
        # `metrics=` stay unweighted. weighted_metrics divide by sum(w), which
        # is what excludes the value-only rows (policy weight 0) from policy
        # accuracy. History/evaluate key names are unchanged
        # (<output>_accuracy). argmax is softmax-invariant, so policy accuracy
        # is correct on logits; value accuracy = argmax WDL correct.
        weighted_metrics={"policy_output": "accuracy", "value_output": "accuracy"},
    )

    # --- Resume (restarts from the last CHECKPOINT, not the last fitted chunk,
    #     so no training is silently lost between checkpoint boundaries) ---
    best_val_loss = float("inf")

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
            best_val_loss      = state.get("best_val_loss", float("inf"))
            print(f"Resumed from {os.path.basename(ckpt)} "
                  f"(epoch {start_epoch}, {resume_skip} chunks into epoch, "
                  f"{global_chunk_count} total, lr "
                  f"{_scheduled_lr(global_chunk_count, total_planned):.2e})\n")

    # --- Preload validation set once (small) ---
    # Validation runs ONE batched forward pass per chunk (_val_metrics); the
    # aggregate and the per-source slices are all computed from the same
    # per-sample arrays, so per-source tracking costs no extra network time.
    # If every val chunk carries a "sources" array we keep a boolean mask per
    # source (no array copies); otherwise per-source metrics are skipped.
    # Per-sample POLICY weights come from the targets themselves: value-only
    # rows (drawn tablebase positions) carry an all-zero policy vector ->
    # weight 0, one-hot rows -> weight 1.
    val_data = None                # (vx, vpol, vval, pweight)
    val_masks: list[tuple[str, np.ndarray]] = []   # (source_name, bool mask)
    if val_chunks:
        vb, vp, vv, vs = [], [], [], []
        for vc in val_chunks:
            b, p, v, s = _load_chunk(vc)
            vb.append(b); vp.append(p); vv.append(v); vs.append(s)
        vx = np.concatenate(vb)
        vpol, vval = np.concatenate(vp), np.concatenate(vv)
        val_data = (vx, vpol, vval, vpol.sum(axis=1).astype(np.float32))
        if all(s is not None for s in vs):
            src = np.concatenate(vs)
            for sid in np.unique(src):
                name = SOURCE_NAMES.get(int(sid), f"src{int(sid)}")
                val_masks.append((name, src == sid))
            print("Per-source val subsets: "
                  + ", ".join(f"{n} ({int(m.sum()):,})" for n, m in val_masks) + "\n")
        else:
            print("Val chunks carry no 'sources' array — per-source val metrics off.\n")
        del vb, vp, vv, vs

    metrics_header = [
        "epoch", "global_chunk", "chunk_file", "n_pos",
        "loss", "policy_loss", "value_loss", "policy_acc", "value_acc",
        "val_loss", "val_policy_loss", "val_value_loss",
        "val_policy_acc", "val_value_acc",
        "learning_rate", "seconds",
    ]
    for name, _ in val_masks:
        metrics_header += [f"val_ploss_{name}", f"val_pacc_{name}",
                           f"val_vloss_{name}", f"val_vacc_{name}"]

    current_lr = _scheduled_lr(global_chunk_count, total_planned)

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

            new_lr = _scheduled_lr(global_chunk_count, total_planned)
            if new_lr != current_lr:
                print(f"    LR {current_lr:.0e} → {new_lr:.0e}  "
                      f"({global_chunk_count}/{total_planned} chunks done)", flush=True)
            current_lr = new_lr
            _set_lr(net.model, current_lr)

            boards, policies, values, _ = _load_chunk(chunk_path)

            y_fit, sw_fit = _targets_and_weights(policies, values)
            hist = net.model.fit(
                boards,
                y_fit,
                sample_weight=sw_fit,
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
                "value_acc":   round(h["value_output_accuracy"][-1], 5),
                "learning_rate": current_lr,
                "seconds": 0.0,
            }

            del boards, policies, values

            # Always checkpoint the final chunk of an epoch so epoch boundaries
            # have clean resume state.
            is_last_in_epoch = pos_in_epoch == len(order) - 1
            do_ckpt = _should_checkpoint(global_chunk_count) or is_last_in_epoch

            # Validate every chunk — ONE forward pass over the val set; the
            # aggregate and every per-source slice come from the same
            # per-sample arrays (see _val_metrics / _agg_metrics).
            src_msg = ""
            if val_data is not None:
                vx, vpol, vval, pweight = val_data
                per_sample = _val_metrics(net, vx, vpol, vval, BATCH_SIZE)
                agg = _agg_metrics(per_sample, pweight)
                row["val_loss"]        = round(agg["loss"], 5)
                row["val_policy_loss"] = round(agg["policy_loss"], 5)
                row["val_value_loss"]  = round(agg["value_loss"], 5)
                row["val_policy_acc"]  = round(agg["policy_acc"], 5)
                row["val_value_acc"]   = round(agg["value_acc"], 5)

                # Per-source val metrics (only when the chunks carry sources).
                src_parts = []
                for name, mask in val_masks:
                    sm = _agg_metrics(per_sample, pweight, mask)
                    row[f"val_ploss_{name}"] = round(sm["policy_loss"], 5)
                    row[f"val_pacc_{name}"]  = round(sm["policy_acc"], 5)
                    row[f"val_vloss_{name}"] = round(sm["value_loss"], 5)
                    row[f"val_vacc_{name}"]  = round(sm["value_acc"], 5)
                    src_parts.append(
                        f"{name} v{sm['value_loss']:.3f}/a{sm['value_acc']:.2f}"
                        f" p{sm['policy_loss']:.3f}/a{sm['policy_acc']:.2f}"
                    )
                if src_parts:
                    src_msg = "    per-source val:  " + "  |  ".join(src_parts)

                val_loss = float(row["val_loss"])
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    net.save(BEST_CKPT_PATH)
                    print(f"    new best val loss {val_loss:.5f} → sl_best.weights.h5",
                          flush=True)

            row["seconds"] = round(time.time() - t0, 1)
            for hcol in metrics_header:
                row.setdefault(hcol, "")
            _append_metrics(row, metrics_header)

            # Per-chunk line shows train/val side by side for both heads, so over-
            # vs under-fitting is readable live (not just in metrics.csv).
            def _pair(train, val):
                """'train/val' if a validation set exists, else just 'train'."""
                return f"{train}" if val == "" else f"{train}/{val}"

            msg = (f"[e{epoch} {pos_in_epoch+1}/{len(order)}] "
                   f"{row['chunk_file']}  ({row['seconds']}s)  lr {current_lr:.1e}  | "
                   f"loss {_pair(row['loss'], row['val_loss'])}  "
                   f"pacc {_pair(row['policy_acc'], row['val_policy_acc'])}  "
                   f"ploss {_pair(row['policy_loss'], row['val_policy_loss'])}  "
                   f"vloss {_pair(row['value_loss'], row['val_value_loss'])}  "
                   f"vacc {_pair(row['value_acc'], row['val_value_acc'])}")
            print(msg, flush=True)
            if src_msg:
                print(src_msg, flush=True)

            if do_ckpt:
                ckpt_path = _ckpt_path(global_chunk_count)
                net.save(ckpt_path)
                # pos_in_epoch+1 = chunks completed in this epoch as of this checkpoint.
                # Next epoch resets to 0 (handled by start_epoch comparison on resume).
                # The LR is NOT stored: it re-derives from global_chunk_count.
                _save_state({
                    "epoch":              epoch if not is_last_in_epoch else epoch + 1,
                    "pos_in_epoch":       (pos_in_epoch + 1) if not is_last_in_epoch else 0,
                    "global_chunk_count": global_chunk_count,
                    "last_checkpoint":    ckpt_path,
                    "best_val_loss":      best_val_loss,
                })
                print(f"    saved {os.path.basename(ckpt_path)}", flush=True)

    # Always save a final checkpoint
    final_path = os.path.join(OUT_DIR, "sl_final.weights.h5")
    net.save(final_path)
    print(f"\nDone. Final weights: {final_path}")


if __name__ == "__main__":
    main()
