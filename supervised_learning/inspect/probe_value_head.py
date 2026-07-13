"""
Probe the value head for the source<->label shortcuts the mixed dataset can teach.

Three test sets, all built from positions the training pipeline provably never
trained on, with labels that are known with certainty:

  Probe A  (tablebase draws)   Random legal <=5-piece positions that Syzygy says
           are DRAWN. The tablebase extractor skips draws, and >2700 games rarely
           reach 5 pieces, so training contained (almost) no drawn 5-piece
           positions. If the value head learned "few pieces => decisive", it will
           put low probability on draw here. Decisive positions from the same
           material configs are generated too, as the in-distribution control.

  Probe B  (puzzle defenders)  Defender-to-move positions from forced-mate
           (mateIn*) lichess puzzles — the plies process_puzzle_data.py skips.
           These are provably LOST for the side to move, but they look exactly
           like the (all-win) puzzle positions the net trained on. If the net
           says "win" here, it is reading puzzle-ness, not the board.
           Solver-to-move positions from the same puzzles are the control.

  Calibration + Stockfish agreement (games)  Positions sampled from Batch*.pgn,
           balanced across win/draw/loss outcomes like the training chunks.
           Calibration: bucket the predicted win probability and compare with the
           realised outcome frequency. Agreement: for a subsample, compare the
           net's expected score against Stockfish (worst outliers are printed —
           these are the "horribly misjudged" positions, with FENs).

Run from anywhere (paths resolve like the other inspect tools):
    python supervised_learning/inspect/probe_value_head.py            # full run
    python supervised_learning/inspect/probe_value_head.py --quick    # smoke test

Output: printed report + per-position CSVs in supervised_learning/results/probes/.
"""

import argparse
import csv
import glob
import os
import random
import sys
import threading
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ---------------------------------------------------------------------------
# Path setup (same pattern as inspect_network.py)
# ---------------------------------------------------------------------------

def _find_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", ".."))
    if os.path.isdir(os.path.join(cand, "reinforcement_learning")):
        return cand
    raise RuntimeError("Could not locate the chess_bot repo root.")


REPO_ROOT = _find_repo_root()
RL_DIR = os.path.join(REPO_ROOT, "reinforcement_learning")
DATASET_DIR = os.path.join(REPO_ROOT, "supervised_learning", "create_dataset")
for _p in (REPO_ROOT, RL_DIR, DATASET_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO_ROOT)

import chess  # noqa: E402
import chess.engine  # noqa: E402
import chess.pgn  # noqa: E402
import chess.syzygy  # noqa: E402

import dataset_common as dc  # noqa: E402
import process_tablebase_data as ptd  # noqa: E402  (reuses _random_position + MATERIALS)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEIGHTS_DEFAULT = os.path.join(
    REPO_ROOT, "supervised_learning", "checkpoints", "sl_best.weights.h5"
)
OUT_DIR = os.path.join(REPO_ROOT, "supervised_learning", "results", "probes")
PUZZLE_CSV = os.path.join(
    REPO_ROOT, "supervised_learning", "supervised_data", "lichess_db_puzzle.csv"
)
PGN_GLOB = os.path.join(
    REPO_ROOT, "supervised_learning", "supervised_data", "Batch*.pgn"
)
STOCKFISH = r"C:\stockfish\stockfish.exe"

SEED = 20260710
INFER_BATCH = 256

# Mostly-drawn material configs NOT in the training MATERIALS list — broader
# coverage of "few pieces but drawn". (No lone-minor configs: python-chess
# flags KBvK / KNvK as game-over via insufficient material, so _random_position
# rejects them; KNNvK and equal-material pairs probe the same heuristic.)
EXTRA_DRAW_MATERIALS = [
    ("NN", ""), ("R", "R"), ("Q", "Q"), ("B", "B"), ("N", "N"), ("B", "N"),
]

WDL_NAMES = ("win", "draw", "loss")


# ---------------------------------------------------------------------------
# Position generation (all TF-free, runs before the network is loaded)
# ---------------------------------------------------------------------------

def gen_probe_a(n_draw: int, n_extra: int, n_decisive: int, rng: random.Random):
    """Rows (fen, group, true_class) from random <=5-piece Syzygy positions.

    Groups: draw_train (drawn, training materials), draw_extra (drawn, extra
    materials), win_train / loss_train (decisive controls, training materials).
    Cursed wins / blessed losses (|wdl| == 1) are skipped as ambiguous.
    """
    quotas = {
        "draw_train": n_draw,
        "draw_extra": n_extra,
        "win_train": n_decisive,
        "loss_train": n_decisive,
    }
    rows: list[tuple[str, str, int]] = []
    tb = chess.syzygy.open_tablebase(ptd.TB_DIR)
    attempts = 0
    t0 = time.time()
    while any(v > 0 for v in quotas.values()) and attempts < 500_000:
        attempts += 1
        # Steer sampling toward whichever groups still need positions.
        if quotas["draw_extra"] > 0 and (quotas["draw_train"] <= 0 or attempts % 3 == 0):
            material, is_extra = rng.choice(EXTRA_DRAW_MATERIALS), True
        else:
            material, is_extra = rng.choice(ptd.MATERIALS), False
        board = ptd._random_position(rng, material)
        if board is None:
            continue
        try:
            wdl = tb.probe_wdl(board)
        except (chess.syzygy.MissingTableError, KeyError):
            continue
        if abs(wdl) == 1:
            continue
        if wdl == 0:
            group = "draw_extra" if is_extra else "draw_train"
            true = dc.WDL_DRAW
        elif is_extra:
            continue  # decisive controls only from training materials
        else:
            group = "win_train" if wdl >= 2 else "loss_train"
            true = dc.WDL_WIN if wdl >= 2 else dc.WDL_LOSS
        if quotas[group] <= 0:
            continue
        quotas[group] -= 1
        rows.append((board.fen(), group, true))
    tb.close()
    print(f"  probe A: {len(rows):,} positions "
          f"({attempts:,} samples, {time.time() - t0:.0f}s)")
    return rows


def gen_probe_b(n_defender: int, n_solver: int):
    """Rows (fen, group, true_class) from forced-mate (mateIn*) puzzles.

    Same parsing/validation as process_puzzle_data._solver_positions, but the
    line must END IN CHECKMATE and the defender-to-move plies are kept too:
    group=defender is provably lost, group=solver provably winning.
    """
    csv.field_size_limit(10 ** 7)
    rows: list[tuple[str, str, int]] = []
    n_def = n_sol = puzzles = 0
    t0 = time.time()
    with open(PUZZLE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if n_def >= n_defender and n_sol >= n_solver:
                break
            if "mateIn" not in row["Themes"]:
                continue
            moves = row["Moves"].split()
            if len(moves) < 2:
                continue
            board = chess.Board(row["FEN"])
            try:
                board.push_uci(moves[0])  # opponent setup move
            except Exception:
                continue
            solver_fens, defender_fens, ok = [], [], True
            for i in range(1, len(moves)):
                try:
                    mv = chess.Move.from_uci(moves[i])
                except Exception:
                    ok = False
                    break
                if mv not in board.legal_moves:
                    ok = False
                    break
                (solver_fens if i % 2 == 1 else defender_fens).append(board.fen())
                board.push(mv)
            if not ok or not board.is_checkmate():
                continue
            puzzles += 1
            for fen in defender_fens:
                if n_def < n_defender:
                    rows.append((fen, "defender", dc.WDL_LOSS))
                    n_def += 1
            for fen in solver_fens:
                if n_sol < n_solver:
                    rows.append((fen, "solver", dc.WDL_WIN))
                    n_sol += 1
    print(f"  probe B: {n_def:,} defender + {n_sol:,} solver positions "
          f"from {puzzles:,} mate puzzles ({time.time() - t0:.0f}s)")
    return rows


def gen_calibration(per_class: int, plies_per_game: int, rng: random.Random):
    """Rows (fen, true_class, ply, n_pieces) sampled from Batch*.pgn.

    A few random plies per game (decorrelated), collected until each WDL class
    has per_class positions — mirroring the 33/33/33 balance the net trained on.
    """
    buckets: dict[int, list[tuple[str, int, int, int]]] = {0: [], 1: [], 2: []}
    games = 0
    t0 = time.time()
    for path in sorted(glob.glob(PGN_GLOB)):
        if all(len(b) >= per_class for b in buckets.values()):
            break
        with open(path, encoding="utf-8", errors="replace") as fh:
            while not all(len(b) >= per_class for b in buckets.values()):
                try:
                    game = chess.pgn.read_game(fh)
                except Exception:
                    continue
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                if result not in ("1-0", "0-1", "1/2-1/2"):
                    continue
                moves = list(game.mainline_moves())
                if len(moves) < 8:
                    continue
                games += 1
                picks = set(rng.sample(range(len(moves)), min(plies_per_game, len(moves))))
                board = game.board()
                for i, mv in enumerate(moves):
                    if i in picks:
                        true = dc.result_to_wdl_class(result, board.turn)
                        if true is not None and len(buckets[true]) < per_class * 2:
                            buckets[true].append(
                                (board.fen(), true, i, chess.popcount(board.occupied))
                            )
                    board.push(mv)
    rows = []
    for c, b in buckets.items():
        rows.extend(rng.sample(b, min(per_class, len(b))))
    rng.shuffle(rows)
    print(f"  calibration: {len(rows):,} positions from {games:,} games "
          f"({time.time() - t0:.0f}s)")
    return rows


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_net(weights_path: str):
    print(f"\nLoading BigNetwork + {os.path.basename(weights_path)} "
          f"(TF import may take a while) ...", flush=True)
    t0 = time.time()
    from networks.big_network import BigNetwork
    net = BigNetwork()
    net.load(weights_path)
    print(f"  loaded in {time.time() - t0:.0f}s")
    return net


def predict_wdl(net, fens: list[str]) -> np.ndarray:
    boards = np.empty((len(fens), 8, 8, 20), dtype=np.float16)
    for i, fen in enumerate(fens):
        boards[i] = dc.board_to_input_tensor(chess.Board(fen))
    out = np.empty((len(fens), 3), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(fens), INFER_BATCH):
        _, wdl = net.predict_batch(boards[i:i + INFER_BATCH])
        out[i:i + len(wdl)] = wdl
        done = i + len(wdl)
        if done % (INFER_BATCH * 20) < INFER_BATCH or done == len(fens):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {done:,}/{len(fens):,}  ({rate:,.0f} pos/s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def report_group(title: str, preds: np.ndarray, expected: int) -> None:
    n = len(preds)
    mean = preds.mean(axis=0)
    share = np.bincount(preds.argmax(axis=1), minlength=3) / n
    print(f"  {title:<42s} n={n:>6,}")
    print(f"    mean predicted  W {mean[0]:.3f}  D {mean[1]:.3f}  L {mean[2]:.3f}")
    print(f"    argmax share    W {share[0]:5.1%}  D {share[1]:5.1%}  L {share[2]:5.1%}")
    print(f"    true class = {WDL_NAMES[expected]}: mean P = {mean[expected]:.3f}, "
          f"argmax-correct = {share[expected]:5.1%}")


def save_csv(name: str, header: list[str], rows: list) -> str:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Stockfish agreement
# ---------------------------------------------------------------------------

def sf_analyse(rows, depth: int, workers: int):
    """rows: (fen, true, ply, pieces). Returns per-row dicts incl. SF expectation."""
    chunks = [rows[i::workers] for i in range(workers)]
    results: list[list[dict]] = [[] for _ in range(workers)]
    counter = {"done": 0}
    lock = threading.Lock()
    t0 = time.time()

    def work(widx: int) -> None:
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        try:
            try:
                eng.configure({"UCI_ShowWDL": True})
            except Exception:
                pass
            for fen, true, ply, pieces in chunks[widx]:
                board = chess.Board(fen)
                try:
                    info = eng.analyse(board, chess.engine.Limit(depth=depth))
                except chess.engine.EngineError:
                    continue
                score = info["score"].pov(board.turn)
                wdl_info = info.get("wdl")
                wdl = (wdl_info.pov(board.turn) if wdl_info is not None
                       else score.wdl(ply=board.ply()))
                results[widx].append({
                    "fen": fen, "true": true, "ply": ply, "pieces": pieces,
                    "cp": score.score(mate_score=32000),
                    "sf_e": wdl.expectation(),
                })
                with lock:
                    counter["done"] += 1
                    if counter["done"] % 200 == 0:
                        rate = counter["done"] / max(time.time() - t0, 1e-9)
                        print(f"    SF {counter['done']:,}/{len(rows):,} "
                              f"({rate:.1f} pos/s)", flush=True)
        finally:
            eng.quit()

    threads = [threading.Thread(target=work, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flat = [r for part in results for r in part]
    print(f"    SF done: {len(flat):,} positions in {time.time() - t0:.0f}s")
    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=WEIGHTS_DEFAULT)
    ap.add_argument("--probe-a", type=int, default=3000, help="drawn TB positions")
    ap.add_argument("--probe-a-extra", type=int, default=1000)
    ap.add_argument("--probe-a-decisive", type=int, default=1500, help="per class")
    ap.add_argument("--probe-b", type=int, default=4000, help="defender positions")
    ap.add_argument("--calib", type=int, default=5000, help="game positions per WDL class")
    ap.add_argument("--sf", type=int, default=1800, help="positions to check with Stockfish")
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test sizes")
    args = ap.parse_args()

    if args.quick:
        args.probe_a, args.probe_a_extra, args.probe_a_decisive = 120, 60, 60
        args.probe_b, args.calib, args.sf, args.sf_workers = 150, 200, 40, 2

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    print("=== Value-head probes ===")
    print(f"weights: {args.weights}\n\nGenerating position sets (TF-free) ...")
    rows_a = gen_probe_a(args.probe_a, args.probe_a_extra, args.probe_a_decisive, rng)
    rows_b = gen_probe_b(args.probe_b, args.probe_b)
    rows_c = gen_calibration(args.calib, plies_per_game=4, rng=rng)

    net = load_net(args.weights)

    # ---- Probe A ---------------------------------------------------------
    print("\n--- Probe A: <=5-piece Syzygy positions (value head vs tablebase truth) ---")
    preds_a = predict_wdl(net, [r[0] for r in rows_a])
    groups_a = {
        "draw_train": "DRAWN, training materials (never trained on)",
        "draw_extra": "DRAWN, extra materials (never trained on)",
        "win_train":  "WON, training materials (control)",
        "loss_train": "LOST, training materials (control)",
    }
    for g, label in groups_a.items():
        idx = [i for i, r in enumerate(rows_a) if r[1] == g]
        if idx:
            report_group(label, preds_a[idx], rows_a[idx[0]][2])
    save_csv("probe_a.csv", ["fen", "group", "true", "p_win", "p_draw", "p_loss"],
             [(r[0], r[1], WDL_NAMES[r[2]], *np.round(preds_a[i], 4))
              for i, r in enumerate(rows_a)])

    # ---- Probe B ---------------------------------------------------------
    print("\n--- Probe B: forced-mate puzzle positions (solver vs defender) ---")
    preds_b = predict_wdl(net, [r[0] for r in rows_b])
    for g, label, exp in (
        ("solver",   "SOLVER to move: provably WINNING (control)", dc.WDL_WIN),
        ("defender", "DEFENDER to move: provably LOST (never trained on)", dc.WDL_LOSS),
    ):
        idx = [i for i, r in enumerate(rows_b) if r[1] == g]
        if idx:
            report_group(label, preds_b[idx], exp)
    save_csv("probe_b.csv", ["fen", "group", "true", "p_win", "p_draw", "p_loss"],
             [(r[0], r[1], WDL_NAMES[r[2]], *np.round(preds_b[i], 4))
              for i, r in enumerate(rows_b)])

    # ---- Calibration ------------------------------------------------------
    print("\n--- Calibration: predicted win% vs realised outcome (balanced game positions) ---")
    preds_c = predict_wdl(net, [r[0] for r in rows_c])
    true_c = np.array([r[1] for r in rows_c])
    print(f"  value CE on this set: "
          f"{-np.log(np.clip(preds_c[np.arange(len(true_c)), true_c], 1e-9, 1)).mean():.4f}"
          f"   (argmax accuracy {np.mean(preds_c.argmax(axis=1) == true_c):.1%})")
    print(f"  {'pred win prob':>14s} {'n':>6s} {'mean pred':>10s} {'actual win%':>12s}")
    ece, n_all = 0.0, len(rows_c)
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        m = (preds_c[:, 0] >= lo) & (preds_c[:, 0] < (hi if hi < 1.0 else 1.01))
        if m.sum() == 0:
            continue
        mean_p, actual = preds_c[m, 0].mean(), (true_c[m] == dc.WDL_WIN).mean()
        ece += m.sum() / n_all * abs(mean_p - actual)
        print(f"  {lo:>6.1f}-{hi:<6.1f} {m.sum():>6,} {mean_p:>10.3f} {actual:>12.3f}")
    print(f"  expected calibration error (win prob): {ece:.4f}")
    save_csv("calibration.csv",
             ["fen", "true", "ply", "pieces", "p_win", "p_draw", "p_loss"],
             [(r[0], WDL_NAMES[r[1]], r[2], r[3], *np.round(preds_c[i], 4))
              for i, r in enumerate(rows_c)])

    # ---- Stockfish agreement ----------------------------------------------
    print(f"\n--- Stockfish agreement (depth {args.sf_depth}, "
          f"{args.sf_workers} engines) ---")
    sub_idx = rng.sample(range(len(rows_c)), min(args.sf, len(rows_c)))
    sf_rows = sf_analyse([rows_c[i] for i in sub_idx], args.sf_depth, args.sf_workers)
    fen_to_pred = {rows_c[i][0]: preds_c[i] for i in sub_idx}
    out_rows = []
    for r in sf_rows:
        p = fen_to_pred[r["fen"]]
        e_net = float(p[0] + 0.5 * p[1])
        out_rows.append((r["fen"], r["ply"], r["pieces"], r["cp"],
                         round(r["sf_e"], 4), round(e_net, 4),
                         round(abs(e_net - r["sf_e"]), 4),
                         *np.round(p, 4)))
    diffs = np.array([r[6] for r in out_rows])
    e_sf = np.array([r[4] for r in out_rows], dtype=np.float64)
    e_net_arr = np.array([r[5] for r in out_rows], dtype=np.float64)
    print(f"  n={len(out_rows):,}  MAE(expected score) = {diffs.mean():.4f}  "
          f"corr = {np.corrcoef(e_sf, e_net_arr)[0, 1]:.3f}")
    for lo, hi, label in ((25, 33, "opening-ish (25-32 pieces)"),
                          (13, 25, "middlegame (13-24 pieces)"),
                          (2, 13, "endgame (<=12 pieces)")):
        m = np.array([(lo <= r[2] < hi) for r in out_rows])
        if m.sum():
            print(f"    {label:<28s} n={m.sum():>5,}  MAE = {diffs[m].mean():.4f}")
    print("\n  worst 12 disagreements (net vs SF):")
    for r in sorted(out_rows, key=lambda r: -r[6])[:12]:
        print(f"    dE={r[6]:.2f}  net W/D/L {r[7]:.2f}/{r[8]:.2f}/{r[9]:.2f} "
              f"E={r[5]:.2f} | SF cp {r[3]:>6} E={r[4]:.2f} | pieces {r[2]:>2} "
              f"ply {r[1]:>3}\n      {r[0]}")
    save_csv("sf_agreement.csv",
             ["fen", "ply", "pieces", "sf_cp", "sf_e", "net_e", "abs_diff",
              "p_win", "p_draw", "p_loss"], out_rows)

    print(f"\nDone. CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()
