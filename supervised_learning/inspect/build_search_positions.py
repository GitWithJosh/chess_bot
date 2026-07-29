"""Build the Stockfish-annotated position suite that the search-parameter sweep
(tune_search_params.py) scores every (c_puct, fpu, sims) cell against.

Runs LOCALLY, not on Kaggle: Stockfish needs no GPU, and Kaggle has no binary.
Output is a small JSON (~100-200 KB) that is committed to the repo AND uploaded
as a tiny Kaggle dataset the sweep mounts read-only.

Why these positions
-------------------
Positions are sampled from the *late* gauntlet PGNs (see CONFIG.PGNS) — the
checkpoints that are genuinely competitive with sl_best (match score ~0.5), so
both sides play real chess. Earlier gauntlet games are sl_best crushing a
near-random net; positions from them are off-distribution for the search we
actually ship. NOTE sl_final is byte-identical to sl_469 (same weights, same 50
games) — never list both or every position doubles.

These are still *raw-policy* games (the gauntlet plays MODE="raw"), so mildly
off-distribution vs. MCTS play — but far closer than any hand-curated suite, and
it is what we have on disk.

Design choices
--------------
  * MultiPV = every legal move. rank/cp of the net's choice is then exact for
    any move, with no invented penalty for "outside the top-N" (which means
    wildly different things in a dead-drawn vs. a sharp position). Verified to
    reproduce the hand-made table in tune_batch_size.py to the centipawn.
  * cp is from the SIDE-TO-MOVE's POV (``score.relative``), best move first, so
    the sweep reads rank = list index and cp_loss = best_cp - chosen_cp.
  * Annotate BROADLY and filter at analysis time. No criticality/sharpness
    filter is applied here: quiet positions are cheap to keep and the sweep can
    drop them with a min-spread threshold, but baking that choice into this slow
    one-off step would freeze a bias we can't revisit without re-annotating.
  * Sampling is decorrelated: skip the seeded/opening plies, then take every
    PLY_STRIDE-th ply so consecutive near-identical positions from one game
    don't dominate. Dedup across games by position (transposition key, matching
    the network's own cache key), then stratify by game phase.
  * Deterministic given SEED + PGNs, and resumable: annotations are flushed to
    the JSON as each position finishes, and a re-run skips positions already in
    it. A crash or Ctrl-C costs at most one position.

Usage:
    python supervised_learning/inspect/build_search_positions.py
    python supervised_learning/inspect/build_search_positions.py --plan-only
    python supervised_learning/inspect/build_search_positions.py --force

Then commit the JSON and upload it as/refresh the Kaggle dataset — see the
banner the script prints at the end for exact paths.
"""

import argparse
import datetime as _dt
import glob
import json
import os
import random
import sys

import chess
import chess.engine
import chess.pgn

# --- import the shared phase / stockfish-detection helpers from phase_acpl ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_SL_DIR = os.path.normpath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.normpath(os.path.join(_SL_DIR, ".."))
if _SL_DIR not in sys.path:
    sys.path.insert(0, _SL_DIR)
from phase_acpl import find_stockfish, game_phase  # noqa: E402

# ============================ CONFIG — edit, then run ============================
# Late, competitive checkpoints only (score ~0.5 vs sl_best). NOT sl_final
# (== sl_469). Paths are relative to the gauntlet PGN dir below.
PGN_DIR = os.path.join(REPO_ROOT, "supervised_learning", "results", "gauntlet_pgns")
PGNS = [
    "sl_best_vs_400.pgn",
    "sl_best_vs_440.pgn",
    "sl_best_vs_469.pgn",
]

SKIP_PLIES = 8            # drop the 3 seeded random plies + a little opening book
PLY_STRIDE = 3            # then keep every 3rd ply -> decorrelates within a game

# Stratified targets per game phase (material-based, from phase_acpl.game_phase).
# Middlegame-heavy because that is where search parameters bite hardest; enough
# endgame that phase is still represented (endgames are where a value estimate
# tends to be hardest, so search params can matter there too).
PHASE_TARGETS = {
    "opening": 60,
    "middlegame": 250,
    "endgame": 120,
}

DEPTH = 16                # Stockfish depth per position (referee strength)
THREADS = 4              # Stockfish Threads (local run — use your cores)
HASH_MB = 512            # Stockfish Hash (MB)
SEED = 12                 # sampling RNG seed (deterministic suite)
STOCKFISH = None          # None = auto-detect (PATH / $STOCKFISH_PATH / C:\stockfish)

OUT_JSON = os.path.join(_HERE, "search_positions.json")
MATE_CP = 30000           # mate folded to +/-(MATE_CP - mate_in) via python-chess
SCHEMA_VERSION = 1
# ================================================================================


def _pos_key(board: chess.Board):
    """Dedup key: transposition key + halfmove clock.

    Matches MCTS._position_key — two positions the network encodes identically
    (plane 17 includes the halfmove clock) are one sample. The full FEN is what
    we store and search from; this only decides duplicates.
    """
    try:
        return (board._transposition_key(), board.halfmove_clock)
    except AttributeError:
        return (" ".join(board.fen().split(" ")[:4]), board.halfmove_clock)


def collect_candidates() -> list[dict]:
    """Deterministically gather unique, decorrelated candidate positions.

    Returns dicts {fen, phase, source} in stable discovery order (first
    occurrence wins on dedup), before any per-phase subsampling.
    """
    seen: set = set()
    candidates: list[dict] = []
    for pgn_name in PGNS:
        path = os.path.join(PGN_DIR, pgn_name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            game_idx = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                game_idx += 1
                board = game.board()
                for ply, mv in enumerate(game.mainline_moves()):
                    # sample the position BEFORE playing ply `mv`
                    if (ply >= SKIP_PLIES
                            and (ply - SKIP_PLIES) % PLY_STRIDE == 0
                            and not board.is_game_over()
                            and board.legal_moves.count() > 1):
                        key = _pos_key(board)
                        if key not in seen:
                            seen.add(key)
                            candidates.append({
                                "fen": board.fen(),
                                "phase": game_phase(board),
                                "source": f"{pgn_name}#g{game_idx}@ply{ply}",
                            })
                    board.push(mv)
    return candidates


def sample_stratified(candidates: list[dict]) -> list[dict]:
    """Take up to PHASE_TARGETS[phase] positions per phase, seeded-random."""
    by_phase: dict[str, list[dict]] = {}
    for c in candidates:
        by_phase.setdefault(c["phase"], []).append(c)

    rng = random.Random(SEED)
    chosen: list[dict] = []
    print("\nphase        available  target   taken")
    for phase in ("opening", "middlegame", "endgame"):
        pool = by_phase.get(phase, [])
        target = PHASE_TARGETS.get(phase, 0)
        rng.shuffle(pool)
        take = pool[:target]
        chosen.extend(take)
        flag = "  <-- SHORT" if len(take) < target else ""
        print(f"{phase:<12} {len(pool):>9}  {target:>6}  {len(take):>6}{flag}")
    # stable order for readable, reproducible output
    chosen.sort(key=lambda c: c["source"])
    return chosen


def annotate(board: chess.Board, engine: chess.engine.SimpleEngine) -> list[list]:
    """MultiPV-all Stockfish annotation: [[uci, cp_relative], ...] best-first."""
    n = board.legal_moves.count()
    infos = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=n)
    moves = []
    for info in infos:
        pv = info.get("pv")
        assert pv is not None
        uci = pv[0].uci()
        score = info.get("score")
        assert score is not None
        cp = score.relative.score(mate_score=MATE_CP)
        moves.append([uci, int(cp)])
    moves.sort(key=lambda m: m[1], reverse=True)  # best (highest, mover POV) first
    if len(moves) != n:
        print(f"    !! got {len(moves)} lines for {n} legal moves "
              f"(engine MultiPV cap?) — position kept as-is")
    return moves


def load_existing() -> dict:
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {}


def write_json(meta: dict, positions: list[dict]) -> None:
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"meta": meta, "positions": positions}, f, indent=1)
    os.replace(tmp, OUT_JSON)  # atomic: a crash mid-write can't corrupt the file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-only", action="store_true",
                    help="sample and print the suite composition, don't annotate")
    ap.add_argument("--force", action="store_true",
                    help="re-annotate positions already present in the JSON")
    args = ap.parse_args()

    candidates = collect_candidates()
    print(f"collected {len(candidates)} unique candidate positions "
          f"from {len(PGNS)} PGNs")
    suite = sample_stratified(candidates)
    print(f"\nsuite size: {len(suite)} positions")

    if args.plan_only:
        print("\n--plan-only: no annotation performed.")
        return

    # Resume: reuse annotations already in the JSON (keyed by FEN) unless --force.
    existing = load_existing()
    done: dict[str, dict] = {}
    if not args.force:
        for p in existing.get("positions", []):
            done[p["fen"]] = p
        if done:
            print(f"resuming: {len(done)} positions already annotated")

    sf_path = find_stockfish(STOCKFISH)
    print(f"Stockfish: {sf_path}  (depth {DEPTH}, MultiPV=all)\n")
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": THREADS, "Hash": HASH_MB})

    meta = {
        "schema_version": SCHEMA_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "sf_depth": DEPTH,
        "sf_path": os.path.basename(sf_path),
        "cp_pov": "side-to-move",
        "mate_cp": MATE_CP,
        "source_pgns": PGNS,
        "skip_plies": SKIP_PLIES,
        "ply_stride": PLY_STRIDE,
        "phase_targets": PHASE_TARGETS,
        "seed": SEED,
        "suite_size": len(suite),
    }

    positions: list[dict] = []
    n_todo = sum(1 for c in suite if c["fen"] not in done)
    print(f"annotating {n_todo} new position(s) "
          f"({len(suite) - n_todo} reused)\n")
    t0 = _dt.datetime.now()
    n_done = 0
    try:
        for i, cand in enumerate(suite, 1):
            fen = cand["fen"]
            if fen in done:
                positions.append(done[fen])
                continue
            board = chess.Board(fen)
            cand["moves"] = annotate(board, engine)
            positions.append(cand)
            done[fen] = cand
            # flush after every position so a crash costs at most this one
            write_json(meta, positions)
            # ETA off actual annotation work only (reuses are instant)
            n_done += 1
            elapsed = (_dt.datetime.now() - t0).total_seconds()
            rate = n_done / elapsed if elapsed else 0
            remaining = (n_todo - n_done) / rate if rate else 0
            print(f"  [{i:>3}/{len(suite)}] {cand['phase']:<11} "
                  f"{cand['source']:<24} {len(cand['moves']):>2} moves  "
                  f"ETA {remaining/60:4.1f}m")
    finally:
        engine.quit()

    write_json(meta, positions)
    print(f"\nwrote {len(positions)} positions -> {OUT_JSON}")
    _print_upload_banner()


def _print_upload_banner() -> None:
    rel = os.path.relpath(OUT_JSON, REPO_ROOT).replace("\\", "/")
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"1. Commit the suite:")
    print(f"     git add {rel}")
    print(f"2. Add search_positions.json to the ROOT of the SAME Kaggle dataset")
    print(f"   the notebook already mounts (the one with the weights file and")
    print(f"   the reinforcement_learning/ folder), then bump the dataset")
    print(f"   version. tune_search_params.py finds it via")
    print(f"     /kaggle/input/**/search_positions.json")
    print(f"   so its exact location in the dataset doesn't matter.")


if __name__ == "__main__":
    main()
