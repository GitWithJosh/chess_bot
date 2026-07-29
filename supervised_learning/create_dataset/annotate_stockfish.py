"""
Annotate routed manifest positions with Stockfish WDL -> soft value targets.

Run from the chess_bot root, between routing and encoding:

    python supervised_learning/create_dataset/build_dataset.py --route-only
    python supervised_learning/create_dataset/annotate_stockfish.py     # overnight
    python supervised_learning/create_dataset/build_dataset.py --encode-only

WHY: the game-outcome label marks every position of a won game "win", even the
ones that were objectively balanced — measured on a 12k sample, Stockfish d12
calls 48% of positions from won games drawn, and agrees with the outcome label
only 78% of the time. Blending the outcome one-hot with Stockfish's WDL (see
dataset_common.blend_value, VALUE_LAMBDA) keeps the label the net is asked to
predict but removes most of that back-projected noise.

Depth 12 is deliberate: it is the mean depth the net's own search reaches at
1000 simulations, so the value head is taught an evaluation it could plausibly
verify at play time rather than one from a much deeper search.

SCOPE: games and puzzles only. Tablebase rows are skipped — Syzygy labels are
exact ground truth and a d12 search would only degrade them.

RESUMABLE: results append to sf_cache/sf_XX.csv shards as they are produced.
Re-running skips every FEN already in the cache, so an interrupted run (or a
deliberate stop) resumes with no lost work and no double effort. Ctrl+C is
handled cleanly — in-flight results are flushed before exit.

PROGRESS: printed every ~20s, and mirrored to sf_cache/progress.json (current
state) and sf_cache/progress.log (append-only history), so an overnight run can
be checked after the fact.
"""

import argparse
import glob
import json
import os
import queue
import signal
import sys
import threading
import time

import numpy as np

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_common as dc  # noqa: E402

BUILD_DIR = os.path.join(dc.repo_root(), "supervised_learning", "processed_data", "_build")
POOL_DIR = os.path.join(dc.repo_root(), "supervised_learning", "pools")
WORKLIST = os.path.join(dc.SF_CACHE_DIR, "worklist.txt")
PROGRESS_JSON = os.path.join(dc.SF_CACHE_DIR, "progress.json")
PROGRESS_LOG = os.path.join(dc.SF_CACHE_DIR, "progress.log")

STOCKFISH = os.environ.get("STOCKFISH_PATH", r"C:\stockfish\stockfish.exe")
DEPTH = 12
N_ENGINES = 8          # 16 cores; leave headroom for the writer thread + OS
ENGINE_HASH_MB = 64
ENGINE_THREADS = 1      # one core each; parallelism comes from running many
BLOCK = 1000            # FENs per work unit. Also bounds how long a graceful
                        # Ctrl+C waits (one block ~= 75 s at 13 pos/s/engine).
SOFT_TIME_CAP = 30.0    # seconds per position, alongside the depth limit.
                        # NOT a search-quality knob: depth 12 takes ~0.08 s, so
                        # this never binds in practice. It exists because
                        # SimpleEngine._timeout_for returns None when Limit has
                        # no .time — meaning analyse() would wait FOREVER on a
                        # wedged engine, and the main thread joins on workers,
                        # so one hang would strand the whole 24 h run. With a
                        # time set, the wrapper's hard timeout becomes
                        # popen timeout + this, and a hang raises instead.
HASH_BATCH = 500_000    # rows per vectorised batch when building the work list.
                        # Per-ROW numpy calls cost ~8 us of dispatch overhead,
                        # which over 23.5M rows is ~3 min of pure overhead — so
                        # hashing accumulates into batches and the set-membership
                        # test runs once per batch, not once per row.
FLUSH_EVERY = 20.0      # seconds between cache flushes / progress updates

# Tablebase rows are already exact; never send them to Stockfish.
SKIP_SOURCE_IDS = {dc.SOURCE_IDS["tablebase"]}

# Two-stage shutdown. _finish = "take no NEW block, but finish the one you are
# holding" (first Ctrl+C); _stop = "drop everything now" (second Ctrl+C, or an
# internal failure). Results already computed are flushed either way.
_finish = threading.Event()
_stop = threading.Event()


# ---------------------------------------------------------------------------
# Work list
# ---------------------------------------------------------------------------

def _split_routed(line: str) -> tuple[str, str, str, str]:
    """-> (fen, move_idx, wdl, src_id) from a routed line, joined or not.

    A manifest that has already been through join_manifests carries three extra
    fields (w,d,l). Blindly rsplit(",", 3) on such a line silently folds them
    into the FEN and reads the loss-permille as the source id, which yields
    unparseable FENs downstream — so the field count is checked, not assumed.
    """
    parts = line.rsplit(",", 6)
    if len(parts) == 7:
        return parts[0], parts[1], parts[2], parts[3]
    fen, mi, w, s = line.rsplit(",", 3)
    return fen, mi, w, s


def _iter_manifest_rows(paths: list[str], routed: bool):
    """Yield (fen, src_id) from routed chunk manifests or raw pool files."""
    for path in paths:
        if not routed:
            base = os.path.basename(path)
            if base.startswith("pool_tablebase"):
                continue
            src = dc.SOURCE_IDS["puzzle" if base.startswith("pool_puzzles") else "game"]
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if routed:
                    fen, _mi, _w, s = _split_routed(line)
                    yield fen, int(s)
                else:
                    yield line.rsplit(",", 2)[0], src


def build_worklist(paths: list[str], routed: bool) -> int:
    """Write the deduped, still-uncached FENs to WORKLIST. Returns its length.

    Two streaming passes so peak memory stays a few hundred MB rather than the
    ~2 GB it would take to hold 15M FEN strings at once:
      pass 1 collects 64-bit hashes only and works out which are wanted;
      pass 2 re-reads and writes out the FEN the first time each hash appears.

    Emission follows manifest order, which is pool order, which is game order —
    so consecutive work-list entries are consecutive plies and a worker's block
    gets real transposition-table reuse.
    """
    print("Pass 1/2: hashing manifest rows ...", flush=True)
    parts: list[np.ndarray] = []
    buf: list[int] = []
    total_rows = n_annotatable = 0
    for fen, src in _iter_manifest_rows(paths, routed):
        total_rows += 1
        if src in SKIP_SOURCE_IDS:
            continue
        buf.append(dc.fen_hash(fen))
        if len(buf) >= HASH_BATCH:
            parts.append(np.array(buf, dtype=np.uint64))
            n_annotatable += len(buf)
            buf.clear()
    if buf:
        parts.append(np.array(buf, dtype=np.uint64))
        n_annotatable += len(buf)
    del buf
    h = np.concatenate(parts) if parts else np.empty(0, np.uint64)
    del parts
    uniq = np.unique(h)
    print(f"  {total_rows:,} rows | {len(h):,} annotatable | {len(uniq):,} unique FENs",
          flush=True)

    cached_keys, _ = dc.load_sf_cache()
    if len(cached_keys):
        wanted = uniq[~np.isin(uniq, cached_keys, assume_unique=True)]
        print(f"  {len(cached_keys):,} already cached -> {len(wanted):,} still to do",
              flush=True)
    else:
        wanted = uniq
        print(f"  no cache yet -> {len(wanted):,} to do", flush=True)
    del h, uniq, cached_keys

    if len(wanted) == 0:
        open(WORKLIST, "w").close()
        return 0

    print("Pass 2/2: writing work list ...", flush=True)
    wanted.sort()
    emitted = np.zeros(len(wanted), dtype=bool)
    n = 0
    # Written to a temp file and renamed only on success: a run killed midway
    # must NOT leave a short work list behind, because the resume path reuses
    # whatever WORKLIST it finds and would then silently skip the missing tail.
    tmp = WORKLIST + ".tmp"

    def emit(out, fens: list[str], hashes: list[int]) -> int:
        if not fens:
            return 0
        h = np.array(hashes, dtype=np.uint64)
        idx = np.searchsorted(wanted, h)
        np.clip(idx, 0, len(wanted) - 1, out=idx)
        cand = np.nonzero(wanted[idx] == h)[0]
        written = 0
        for j in cand:                      # only over hits, not every row
            i = idx[j]
            if not emitted[i]:              # first occurrence wins (dedupe)
                emitted[i] = True
                out.write(fens[j] + "\n")
                written += 1
        return written

    with open(tmp, "w", encoding="utf-8") as out:
        fens: list[str] = []
        hashes: list[int] = []
        for fen, src in _iter_manifest_rows(paths, routed):
            if src in SKIP_SOURCE_IDS:
                continue
            fens.append(fen)
            hashes.append(dc.fen_hash(fen))
            if len(fens) >= HASH_BATCH:
                n += emit(out, fens, hashes)
                fens, hashes = [], []
        n += emit(out, fens, hashes)
    os.replace(tmp, WORKLIST)
    print(f"  work list: {n:,} positions -> {WORKLIST}", flush=True)
    return n


def load_worklist() -> list[str]:
    with open(WORKLIST, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _open_engine():
    # setpgrp: put each engine in its OWN process group so a console Ctrl+C is
    # NOT delivered to them. Without it, Ctrl+C kills all 16 engines at once and
    # every worker has to restart one just to finish its in-flight block — the
    # "16 errors, 16 engine restarts" seen on a clean stop. It also means the
    # graceful finish keeps its warm transposition tables.
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH, setpgrp=True)
    eng.configure({"Threads": ENGINE_THREADS, "Hash": ENGINE_HASH_MB,
                   "UCI_ShowWDL": True})
    return eng


def _analyse(eng, fen: str):
    """-> (cp, w, d, l) side-to-move POV, or None if the position is unusable."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return None          # malformed row: skip it, don't take the worker down
    # info=INFO_SCORE, not the INFO_ALL default: the driver is GIL-bound (one
    # Python core pegged while 16 engines sit at ~70% each), and INFO_ALL parses
    # every info line of every iterative-deepening iteration, converting whole
    # PVs into Move objects we throw away. Score and WDL survive the narrower
    # selector — measured present on 3520/3520 positions — for 1.16x throughput.
    info = eng.analyse(board, chess.engine.Limit(depth=DEPTH, time=SOFT_TIME_CAP),
                       info=chess.engine.INFO_SCORE)
    raw_score = info.get("score")
    assert raw_score is not None
    score = raw_score.pov(board.turn)              # Score, side-to-move relative
    wdl_info = info.get("wdl")
    # info["wdl"] is a PovWdl and needs .pov(); Score.wdl() returns a plain Wdl
    # that is ALREADY side-to-move relative and has no .pov() at all — calling
    # it there raises AttributeError, which would kill the worker outright.
    wdl = (wdl_info.pov(board.turn) if wdl_info is not None
           else score.wdl(ply=board.ply()))
    cp = score.score(mate_score=dc.SF_MATE_CP)
    cp = max(-dc.SF_MATE_CP, min(dc.SF_MATE_CP, int(cp)))
    return cp, wdl.wins, wdl.draws, wdl.losses


def worker(wid: int, blocks: "queue.Queue", results: "queue.Queue", stats: dict,
           lock: threading.Lock) -> None:
    """Pull blocks of FENs, analyse them, push (fen, cp, w, d, l) rows.

    The engine is reopened on crash: over a ~26 h run a single hung or dead
    Stockfish process would otherwise silently cost 1/14th of the throughput,
    and losing the whole run to one bad position is not acceptable.
    """
    eng = None
    try:
        eng = _open_engine()
        while not _stop.is_set():
            # Graceful stop: never START a block once asked to wind down. The
            # block already in hand is finished first, so no partial-block work
            # is thrown away and the cache lands on a clean boundary.
            if _finish.is_set():
                return
            try:
                fens = blocks.get_nowait()
            except queue.Empty:
                return
            out = []
            for fen in fens:
                if _stop.is_set():        # hard stop may abandon a block midway;
                    break                 # whatever is in `out` is still saved
                for attempt in (0, 1):
                    try:
                        got = _analyse(eng, fen)
                        if got is not None:
                            out.append((fen, *got))
                        break
                    except (chess.engine.EngineTerminatedError,
                            chess.engine.EngineError, BrokenPipeError,
                            TimeoutError) as exc:
                        # TimeoutError = the engine wedged on this position.
                        # Restart it; the position is retried once, then skipped
                        # (it stays in the work list for a later run either way).
                        with lock:
                            stats["errors"] += 1
                        if attempt == 0:
                            try:
                                if eng is not None:
                                    eng.quit()
                            except Exception:
                                pass
                            try:
                                eng = _open_engine()
                                with lock:
                                    stats["restarts"] += 1
                            except Exception:
                                _stop.set()
                                break
                        else:
                            print(f"  [w{wid}] skipping position after retry: "
                                  f"{type(exc).__name__}", flush=True)
                    except Exception as exc:
                        # Anything unanticipated: skip this ONE position rather
                        # than losing the worker. Over an unattended 24 h run a
                        # dead thread is invisible — it just looks like the job
                        # got slower — so no exception may escape this loop.
                        with lock:
                            stats["errors"] += 1
                        print(f"  [w{wid}] unexpected {type(exc).__name__} on a "
                              f"position, skipping: {exc}", flush=True)
                        break
            if out:
                # Bounded put: if the writer has died the queue never drains, and
                # an unbounded put would hang this worker (and thus the join in
                # annotate()) forever.
                try:
                    results.put(out, timeout=60)
                except queue.Full:
                    print(f"  [w{wid}] writer not draining — stopping", flush=True)
                    _stop.set()
                    return
            with lock:
                stats["done"] += len(out)
    finally:
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                pass


def writer(results: "queue.Queue", stats: dict, lock: threading.Lock,
           total: int, t0: float, already: int) -> None:
    """Drain results into the sharded cache; print + persist progress."""
    handles: dict[int, object] = {}
    pending = 0
    last = time.time()

    def flush(final: bool = False) -> None:
        nonlocal pending, last
        for h in handles.values():
            h.flush()
        pending = 0
        with lock:
            done, errors, restarts = stats["done"], stats["errors"], stats["restarts"]
        el = time.time() - t0
        rate = done / max(el, 1e-9)
        remaining = max(total - done, 0)
        eta = remaining / rate if rate > 0 else 0.0
        # Report against the WHOLE job, not just this run's slice — after a
        # resume, "12% of this run" says nothing about how close the dataset is.
        grand_total = already + total
        grand_done = already + done
        pct = 100.0 * grand_done / max(grand_total, 1)
        msg = (f"  {grand_done:,}/{grand_total:,} ({pct:5.1f}%)  {rate:6.1f} pos/s  "
               f"elapsed {el/3600:5.2f} h  ETA {eta/3600:5.2f} h"
               + (f"  [{errors} errors, {restarts} engine restarts]"
                  if errors or restarts else ""))
        print(msg, flush=True)
        state = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "depth": DEPTH, "engines": N_ENGINES,
            "annotated_total": grand_done,
            "target_total": grand_total,
            "remaining": max(grand_total - grand_done, 0),
            "percent": round(pct, 2),
            "done_this_run": done, "total_this_run": total,
            "cached_before_run": already,
            "rate_pos_per_s": round(rate, 2),
            "elapsed_hours": round(el / 3600, 3),
            "eta_hours": round(eta / 3600, 3),
            "errors": errors, "engine_restarts": restarts,
            "run_finished": bool(final),
        }
        # Written via temp+rename: this file is rewritten every 20 s all night,
        # so a kill lands mid-write eventually, and a truncated progress.json is
        # exactly the thing you do not want to find in the morning.
        tmp_json = PROGRESS_JSON + ".tmp"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_json, PROGRESS_JSON)
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(state) + "\n")
        last = time.time()

    try:
        while True:
            try:
                rows = results.get(timeout=1.0)
            except queue.Empty:
                if _stop.is_set() or stats.get("workers_done"):
                    break
                if time.time() - last >= FLUSH_EVERY:
                    flush()
                continue
            if rows is None:
                break
            for fen, cp, w, d, l in rows:
                shard = dc.fen_hash(fen) % dc.SF_SHARDS
                h = handles.get(shard)
                if h is None:
                    h = open(dc.sf_shard_path(shard), "a", encoding="utf-8")
                    handles[shard] = h
                h.write(f"{fen},{cp},{w},{d},{l}\n")
                pending += 1
            if pending >= 20_000 or time.time() - last >= FLUSH_EVERY:
                flush()
    finally:
        flush(final=True)
        for h in handles.values():
            h.close()


def annotate(fens: list[str], n_engines: int, already: int) -> tuple[bool, int]:
    """Annotate `fens` -> (finished_whole_list, positions_done_this_run)."""
    blocks: queue.Queue = queue.Queue()
    for i in range(0, len(fens), BLOCK):
        blocks.put(fens[i:i + BLOCK])
    results: queue.Queue = queue.Queue(maxsize=64)
    stats = {"done": 0, "errors": 0, "restarts": 0, "workers_done": False}
    lock = threading.Lock()
    t0 = time.time()

    wt = threading.Thread(target=writer,
                          args=(results, stats, lock, len(fens), t0, already),
                          daemon=True)
    wt.start()
    workers = [threading.Thread(target=worker,
                                args=(w, blocks, results, stats, lock), daemon=True)
               for w in range(n_engines)]
    for t in workers:
        t.start()
    try:
        for t in workers:
            while t.is_alive():
                t.join(timeout=0.5)
                # If the writer dies, nothing drains the results queue and every
                # worker eventually blocks on a full put. Fail loudly instead.
                if not wt.is_alive():
                    print("  writer thread died — stopping workers", flush=True)
                    _stop.set()
    except KeyboardInterrupt:          # only if no handler is installed
        _stop.set()
        for t in workers:
            t.join(timeout=30)
    stats["workers_done"] = True
    # Non-blocking: if the writer already exited the queue may be full, and a
    # plain put() would hang here forever after the work is otherwise done.
    # The writer also exits on workers_done, so the sentinel is belt-and-braces.
    try:
        results.put_nowait(None)
    except queue.Full:
        pass
    wt.join(timeout=120)
    with lock:
        done = stats["done"]
    return not (_stop.is_set() or _finish.is_set()), done


# ---------------------------------------------------------------------------
# Join — fold the cache back into the routed manifests for the encoder
# ---------------------------------------------------------------------------

def join_manifests(paths: list[str]) -> bool:
    """Rewrite each routed manifest as fen,move_idx,wdl,src_id,w,d,l.

    Returns True if every manifest was joined, False if interrupted.

    Doing the join once here means build_dataset's encode workers never load the
    cache: 8 processes x ~210 MB of lookup arrays is pure waste when a single
    streaming pass can bake the answer into the manifest. Rows with no
    annotation (tablebase, or a position Stockfish never reached) get -1,-1,-1
    and fall back to the one-hot target at encode time.
    """
    # Sweep stale temp files from a previously killed join. Each would be
    # overwritten anyway the next time its own manifest is processed, but a run
    # that never reaches that file would leave it sitting there indefinitely.
    # They cannot be mistaken for manifests (the c*.txt glob does not match
    # c0010.txt.tmp), so this is tidiness, not correctness.
    stale = sorted(glob.glob(os.path.join(os.path.dirname(paths[0]), "*.tmp")))
    for s in stale:
        try:
            os.remove(s)
        except OSError:
            pass
    if stale:
        print(f"  removed {len(stale)} stale .tmp file(s) from an interrupted run",
              flush=True)

    keys, wdls = dc.load_sf_cache()
    print(f"Joining {len(paths)} manifests against {len(keys):,} annotations ...",
          flush=True)
    hit = miss = 0
    t0 = time.time()
    for n, path in enumerate(paths, 1):
        if _stop.is_set() or _finish.is_set():
            print(f"\n  Join interrupted after {n - 1}/{len(paths)} manifests. "
                  f"Each file is replaced atomically, so the finished ones are "
                  f"intact.\n  Re-run with --join-only to complete it.", flush=True)
            return False

        with open(path, encoding="utf-8") as fin:
            rows = [_split_routed(l) for l in fin.read().splitlines() if l]

        # Hash and look up the WHOLE manifest in one vectorised pass.
        # Doing this per row costs ~5 ms/row, not the ~2 us you would expect:
        # fen_hash returns a PYTHON int, and numpy types one below 2**63 as
        # int64, which mismatches the uint64 keys array and makes it convert all
        # 16.5M elements on every call. Building a uint64 array up front pins
        # the dtype and does one binary search per batch instead of per row.
        todo = [i for i, r in enumerate(rows) if int(r[3]) not in SKIP_SOURCE_IDS]
        wdl_of: dict[int, tuple] = {}
        if todo and len(keys):
            h = np.fromiter((dc.fen_hash(rows[i][0]) for i in todo),
                            dtype=np.uint64, count=len(todo))
            pos = np.searchsorted(keys, h)
            np.clip(pos, 0, len(keys) - 1, out=pos)
            found = keys[pos] == h
            for j, i in enumerate(todo):
                if found[j]:
                    a, b, c = wdls[pos[j]]
                    wdl_of[i] = (int(a), int(b), int(c))

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fout:
            out = []
            for i, (fen, mi, w, s) in enumerate(rows):
                # Rebuilt from the canonical four fields rather than appended to,
                # so re-joining an already-joined manifest refreshes it in place
                # instead of stacking a second w,d,l triple onto every row.
                base = f"{fen},{mi},{w},{s}"
                a = wdl_of.get(i)
                if a is not None:
                    out.append(f"{base},{a[0]},{a[1]},{a[2]}\n")
                    hit += 1
                else:
                    out.append(f"{base},-1,-1,-1\n")
                    if int(s) not in SKIP_SOURCE_IDS:
                        miss += 1
            fout.writelines(out)
        os.replace(tmp, path)

        if n % 25 == 0 or n == len(paths):
            el = time.time() - t0
            eta = el / n * (len(paths) - n)
            print(f"  {n}/{len(paths)} manifests  ({el:.0f}s elapsed, "
                  f"ETA {eta:.0f}s)", flush=True)
    tot = hit + miss
    print(f"  annotated {hit:,}/{tot:,} annotatable rows "
          f"({100*hit/max(tot,1):.2f}%); {miss:,} fall back to one-hot", flush=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global DEPTH, N_ENGINES
    ap = argparse.ArgumentParser(
        description="Annotate the routed manifests with Stockfish WDL.\n\n"
                    "Just run it with no arguments. It works out what is left to "
                    "do, does it, and\njoins the results in when finished. Safe to "
                    "stop (Ctrl+C) and re-run at any time.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", type=int, default=N_ENGINES,
                    help=f"parallel Stockfish processes (default {N_ENGINES}; "
                         f"lower it to free up the machine)")
    ap.add_argument("--depth", type=int, default=DEPTH,
                    help=f"search depth (default {DEPTH}) — do NOT change "
                         f"mid-dataset, the cache does not record depth")
    ap.add_argument("--pools", action="store_true",
                    help="annotate the raw pools instead of the routed manifests")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N positions this run (timed trial)")
    ap.add_argument("--join-only", action="store_true",
                    help="skip annotation, just fold the cache into the manifests")
    ap.add_argument("--no-join", action="store_true",
                    help="annotate only; join later")
    ap.add_argument("--rebuild-worklist", action="store_true",
                    help="recompute the work list even if one exists")
    args = ap.parse_args()

    if args.engines < 1:
        print("ERROR: --engines must be at least 1")
        sys.exit(1)

    DEPTH, N_ENGINES = args.depth, args.engines
    os.makedirs(dc.SF_CACHE_DIR, exist_ok=True)

    if args.pools:
        paths = sorted(glob.glob(os.path.join(POOL_DIR, "pool_*.csv")))
        routed = False
    else:
        paths = sorted(glob.glob(os.path.join(BUILD_DIR, "c*.txt")))
        routed = True
    if not paths:
        where = POOL_DIR if args.pools else BUILD_DIR
        print(f"ERROR: no manifests in {where}.\n"
              f"Run: python supervised_learning/create_dataset/build_dataset.py --route-only")
        sys.exit(1)

    if not os.path.exists(STOCKFISH):
        print(f"ERROR: Stockfish not found at {STOCKFISH} "
              f"(override with STOCKFISH_PATH)")
        sys.exit(1)

    print("=== Stockfish annotation ===")
    print(f"Engine : {STOCKFISH}")
    print(f"Source : {len(paths)} {'routed manifests' if routed else 'pool files'}")
    print(f"Depth  : {DEPTH} | engines {N_ENGINES} | hash {ENGINE_HASH_MB} MB each")
    print(f"Cache  : {dc.SF_CACHE_DIR}\n")

    if args.join_only:
        join_manifests(paths)
        return

    if args.rebuild_worklist or not os.path.exists(WORKLIST):
        build_worklist(paths, routed)
    else:
        print(f"Reusing existing work list ({WORKLIST}); "
              f"pass --rebuild-worklist to recompute.", flush=True)

    fens = load_worklist()
    n_worklist = len(fens)
    keys, _ = dc.load_sf_cache(verbose=False)

    # The work list is written once and reused across resumes, so drop the
    # entries finished by earlier runs rather than re-analysing them. Hashing
    # in batches keeps this from allocating a second 16M-element array on top
    # of the work list itself, which already costs ~2 GB of strings.
    if len(keys):
        print(f"  {len(keys):,} positions in cache; filtering work list ...",
              flush=True)
        keep = np.ones(len(fens), dtype=bool)
        for i in range(0, len(fens), HASH_BATCH):
            h = np.array([dc.fen_hash(f) for f in fens[i:i + HASH_BATCH]],
                         dtype=np.uint64)
            keep[i:i + len(h)] = ~np.isin(h, keys, assume_unique=False)
        fens = [f for f, k in zip(fens, keep) if k]
        del keep
    del keys

    # How much of THIS work list is already done — not the raw cache size. The
    # cache can legitimately hold rows outside the current work list (a previous
    # --pools run, or a re-route with a different seed), and counting those would
    # inflate the totals reported all night.
    already = n_worklist - len(fens)

    # A capped run is BY DEFINITION incomplete, so it must not be allowed to
    # reach the join — joining a partial cache stamps -1,-1,-1 over rows that
    # simply have not been reached yet, and those manifests are this script's
    # own input.
    truncated = args.limit is not None and len(fens) > args.limit
    if args.limit is not None:
        fens = fens[:args.limit]

    completed, done_now = not truncated, 0
    if not fens:
        print("Nothing left to annotate.")
    else:
        est = len(fens) / 180.0 / 3600.0
        print(f"To annotate: {len(fens):,} positions "
              f"(~{est:.1f} h at 180 pos/s)")
        print("Press Ctrl+C once to stop cleanly at the next block boundary "
              "(re-run to resume).\n", flush=True)

        def _sigint(_signum, _frame):
            if _finish.is_set():
                print("\n  Second Ctrl+C — stopping NOW. Results already "
                      "computed are still saved.", flush=True)
                _stop.set()
            else:
                print(f"\n  Ctrl+C — finishing the blocks in flight "
                      f"(up to ~{BLOCK:,} positions each, roughly a minute), "
                      f"then stopping.\n  Press Ctrl+C again to stop "
                      f"immediately.", flush=True)
                _finish.set()

        signal.signal(signal.SIGINT, _sigint)
        ran_to_end, done_now = annotate(fens, N_ENGINES, already)
        completed = ran_to_end and not truncated

    # Joining a PARTIAL cache would stamp -1,-1,-1 onto every not-yet-analysed
    # row, and those manifests are the annotator's own input — so an interrupted
    # run must leave them untouched and let the next run finish the job first.
    if not completed:
        # Derived from counters already in hand. Re-reading the cache and the
        # work list here would cost ~90 s and a second ~2 GB allocation, right
        # when the user is waiting to get their machine back.
        annotated = already + done_now
        left = max(n_worklist - annotated, 0)
        print(f"\n=== Stopped early — nothing lost ===")
        print(f"  annotated so far : {annotated:,} / {n_worklist:,}")
        print(f"  still to do      : {left:,}  (~{left/180/3600:.1f} h)")
        print(f"  The manifests were deliberately left unjoined; the join runs "
              f"once annotation completes.")
        print(f"\n  NEXT: python supervised_learning/create_dataset/"
              f"annotate_stockfish.py")
    elif not args.no_join:
        print()
        join_manifests(paths)
        print(f"\n=== Annotation complete ===")
        print(f"  NEXT: python supervised_learning/create_dataset/"
              f"build_dataset.py --encode-only")


if __name__ == "__main__":
    main()
