"""
Match runner on Kaggle: two opponents play a colour-balanced match, no GUI.

Each side is one of three opponent types, chosen by a spec string:
  * a net    — a weights *filename* (auto-located anywhere under /kaggle/input),
               played with MCTS (mode="mcts") or raw policy argmax (mode="raw").
  * stockfish — a fixed thinking time per move and an adjustable strength (skill
               level 0-20, or a UCI_Elo). Needs the stockfish binary (on Kaggle:
               `!apt-get install -y stockfish`; or set STOCKFISH_PATH).
  * random   — uniform random legal move; a sanity-check baseline.

This is the *Kaggle* tool (a 1000-sim x 100-game match is far too slow locally):
it mirrors play_notebook's zero-config discovery — no __file__, it walks the
mounts to find the repo, and nets are named by filename — so you never hard-code
a dataset path. The PGN is written under /kaggle/working (the mounts are
read-only), with the match's total score as %-escaped comment lines at the top.

Structure (like compare_nets.py): OPENINGS distinct random openings, each played
twice with the colours swapped (total games = 2 x OPENINGS). Colour reversal
cancels white bias; distinct openings (a repeat is rerolled) stop two fixed
opponents from replaying one identical game.

Usage on Kaggle (set Accelerator = GPU first):

    !pip install chess -q
    !apt-get install -y stockfish -q          # only if you use the stockfish side

    run_match(player_a="16_sl_best.weights.h5", player_b="sl_best.weights.h5")
    run_match(player_a="16_sl_best.weights.h5", player_b="stockfish",
                  sf_time=0.1, sf_level=5)     # net vs Stockfish, skill level 5
    run_match(player_a="16_sl_best.weights.h5", player_b="random", openings=25)

A player spec can also carry per-side overrides, e.g. "stockfish:level=8,time=0.2".
Locally it runs as a script too (paths resolve against the repo).
"""

import argparse
import glob
import math
import os
import random
import shutil
import sys
# On Kaggle
"""
!pip install chess -q
!apt-get install -y stockfish -q        # only if using the stockfish side
run_match(player_a="16_sl_best.weights.h5", player_b="stockfish", sf_time=0.1, sf_level=5)
run_match(player_a="16_sl_best.weights.h5", player_b="random", openings=25)

run_match(player_a="16_sl_best.weights.h5", player_b="stockfish", sf_time=0.1, sf_level=5)

"""
# ============================ CONFIG — edit, then run ============================
# Each player is: a weights filename (net), "stockfish", or "random".
PLAYER_A = "16_sl_best.weights.h5"
PLAYER_B = "sl_best.weights.h5"
MODE = "mcts"        # net move selection: "mcts" (search) | "raw" (policy argmax)
SIMS = 1000          # MCTS simulations per move (net, mode="mcts")
BATCH_SIZE = 16      # MCTS leaf batch size (net)
SF_TIME = 0.1        # stockfish: fixed seconds/move
SF_LEVEL = None      # stockfish: skill level 0-20 (None = full strength)
SF_ELO = None        # stockfish: UCI_Elo (overrides skill level; ~1320-3190)
OPENINGS = 50        # unique random openings; total games = 2 x this (colours swapped)
OPENING_PLIES = 3    # random plies used to seed each opening (a "3-move" opening)
MAX_PLIES = 400      # adjudicate a draw past this so a shuffling player can't hang
SEED = 7             # seeds the random openings + the random player
RENDER = False       # draw the SVG board in a notebook as each game plays
OUT_PGN = None       # None -> /kaggle/working/compare_nets_kaggle.pgn (or repo results/ locally)
# ================================================================================


# ---------------------------------------------------------------------------
# Path setup — locate the repo both locally and on Kaggle, with no __file__
# (so it works pasted into a cell). Mirrors play_notebook._find_repo_root.
# ---------------------------------------------------------------------------

def _find_repo_root() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.normpath(os.path.join(here, ".."))
        if os.path.isdir(os.path.join(cand, "reinforcement_learning")):
            return cand
    except NameError:
        pass  # no __file__ (pasted into a cell) — fall back to walking mounts
    for base in ("/kaggle/input", "/kaggle/working", os.getcwd()):
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if "reinforcement_learning" in dirnames:
                return dirpath
    raise RuntimeError(
        "Could not locate the repo root (the folder containing "
        "'reinforcement_learning/'). Attach a dataset that includes it.")


REPO_ROOT = _find_repo_root()
RL_DIR = os.path.join(REPO_ROOT, "reinforcement_learning")
for _p in (REPO_ROOT, RL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Converter opens "reinforcement_learning/move_lookup.json" relative to CWD.
os.chdir(REPO_ROOT)

import chess          # noqa: E402
import chess.engine   # noqa: E402
import chess.pgn      # noqa: E402
import chess.svg      # noqa: E402
import tensorflow as tf  # noqa: E402

from helpers.converter import Converter                              # noqa: E402
from monte_carlo_tree_search.nodes_and_edges_v2 import mirror_move_uci  # noqa: E402
from networks.big_network import BigNetwork                          # noqa: E402

_RESULT_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def _default_out_pgn() -> str:
    """Write into /kaggle/working on Kaggle (the mounts are read-only); locally
    fall back to the repo's results/ dir."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/compare_nets_kaggle.pgn"
    return os.path.join("supervised_learning", "results", "compare_nets_kaggle.pgn")


def _find_weights(spec: str) -> str:
    """Resolve a net given by filename (or any path). Tries it as-given and
    repo-relative, then searches recursively under the Kaggle mounts and the repo
    so a bare basename like 'sl_best.weights.h5' is found wherever it's mounted.
    If several match, the last (sorted) wins — pass more of the path to narrow it.
    """
    if os.path.exists(spec):
        return spec
    rel = os.path.join(REPO_ROOT, spec)
    if os.path.exists(rel):
        return rel
    seen: set[str] = set()
    for root in ("/kaggle/input", "/kaggle/working", REPO_ROOT, os.getcwd()):
        if not os.path.isdir(root):
            continue
        real = os.path.realpath(root)
        if real in seen:
            continue
        seen.add(real)
        hits = sorted(glob.glob(os.path.join(root, "**", spec), recursive=True))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        f"No weights matching {spec!r} under /kaggle/input, /kaggle/working, or "
        f"the repo. Check the filename / attach the dataset.")


def _find_stockfish() -> str:
    """Locate the stockfish binary: STOCKFISH_PATH, then PATH, then the usual apt
    install locations, then any uploaded under /kaggle/input."""
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.exists(env):
        return env
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    candidates = ["/usr/games/stockfish", "/usr/bin/stockfish",
                  "/usr/local/bin/stockfish"]
    for root in ("/kaggle/input", "/kaggle/working"):
        if os.path.isdir(root):
            candidates += sorted(glob.glob(os.path.join(root, "**", "stockfish*"),
                                           recursive=True))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Stockfish binary not found. On Kaggle run `!apt-get install -y stockfish` "
        "(or set the STOCKFISH_PATH env var to an uploaded binary).")


def load_net(path: str) -> BigNetwork:
    net = BigNetwork()
    net.load(path)
    return net


def net_name(path: str) -> str:
    """`.../16_sl_best.weights.h5` -> `16_sl_best`."""
    return os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]


# ---------------------------------------------------------------------------
# Move selection for a net (shared decode path with compare_nets.py).
# ---------------------------------------------------------------------------

def raw_move(net: BigNetwork, converter: Converter, board: chess.Board) -> chess.Move:
    """Greedy argmax over the legal-masked policy head (no search).

    The policy is in the side-to-move's *friendly* perspective, so a black move
    is mirrored back; a decoded back-rank pawn push gets its queen restored (the
    lookup stores queen promotions without the 'q'). Same as compare_nets.raw_move.
    """
    board_tensor = converter.board_to_input_tensor(board)
    policy, _value = net.predict(board_tensor)
    move_probs = converter.mask_illegal_moves(board, policy)
    best_index = int(tf.argmax(move_probs).numpy())
    move_uci = converter._get_move_from_index(best_index)
    if board.turn == chess.BLACK:
        move_uci = mirror_move_uci(move_uci)
    move = chess.Move.from_uci(move_uci)
    if (move.promotion is None
            and chess.square_rank(move.to_square) in (0, 7)
            and board.piece_type_at(move.from_square) == chess.PAWN):
        move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move


# ---------------------------------------------------------------------------
# Players. Each has: .name, .reset() (per-game state), .move(board), .close().
# ---------------------------------------------------------------------------

class NetPlayer:
    """A BigNetwork playing via MCTS ('mcts') or raw policy argmax ('raw')."""

    def __init__(self, net, converter, mode, sims, batch_size, name):
        self.net = net
        self.converter = converter
        self.mode = mode
        self.sims = sims
        self.batch_size = batch_size
        self.name = name

    def reset(self):
        # A fresh game must not reuse the previous game's MCTS tree (persisted on
        # the net instance across search calls), so clear it between games.
        self.net._search_mcts = None

    def move(self, board):
        if self.mode == "raw":
            return raw_move(self.net, self.converter, board)
        return self.net.search_for_best_move(
            board, num_simulations=self.sims, batch_size=self.batch_size)

    def close(self):
        pass


class StockfishPlayer:
    """Stockfish at a fixed thinking time per move and an adjustable strength."""

    def __init__(self, movetime, skill_level=None, elo=None, path=None):
        self.path = path or _find_stockfish()
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self.limit = chess.engine.Limit(time=movetime)
        options = {}
        if elo is not None:
            options["UCI_LimitStrength"] = True
            options["UCI_Elo"] = int(elo)
            tag = f"elo{int(elo)}"
        elif skill_level is not None:
            options["Skill Level"] = int(skill_level)
            tag = f"lvl{int(skill_level)}"
        else:
            tag = "full"
        if options:
            self.engine.configure(options)
        self.name = f"stockfish-{tag}-{movetime}s"

    def reset(self):
        pass

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


class RandomPlayer:
    """Uniform random legal move — a sanity-check baseline."""

    def __init__(self, rng, name="random"):
        self.rng = rng
        self.name = name

    def reset(self):
        pass

    def move(self, board):
        return self.rng.choice(list(board.legal_moves))

    def close(self):
        pass


def _parse_sf_overrides(spec: str, sf_time, sf_level, sf_elo):
    """Read per-side stockfish overrides from a spec like
    'stockfish:level=8,time=0.2' (or 'elo=1500'). Missing keys keep the defaults."""
    movetime, level, elo = sf_time, sf_level, sf_elo
    if ":" in spec:
        for kv in spec.split(":", 1)[1].split(","):
            if not kv.strip():
                continue
            key, _, val = kv.partition("=")
            key = key.strip().lower()
            if key in ("time", "movetime"):
                movetime = float(val)
            elif key in ("level", "skill"):
                level = int(val)
            elif key == "elo":
                elo = int(val)
            else:
                raise ValueError(f"unknown stockfish option {key!r} in {spec!r}")
    return movetime, level, elo


def make_player(spec, converter, net_cache, mode, sims, batch_size,
                sf_time, sf_level, sf_elo, seed):
    """Turn a spec string into a Player. 'random' | 'stockfish[:opts]' | a
    weights filename. Nets are cached by resolved path so the same file used on
    both sides is loaded once."""
    low = spec.strip().lower()
    if low in ("random", "rand"):
        return RandomPlayer(random.Random(seed))
    if low == "stockfish" or low.startswith("stockfish:") \
            or low == "sf" or low.startswith("sf:"):
        movetime, level, elo = _parse_sf_overrides(spec, sf_time, sf_level, sf_elo)
        return StockfishPlayer(movetime, skill_level=level, elo=elo)
    path = os.path.abspath(_find_weights(spec))
    if path not in net_cache:
        net_cache[path] = load_net(path)
    return NetPlayer(net_cache[path], converter, mode, sims, batch_size, net_name(path))


# ---------------------------------------------------------------------------
# Notebook rendering (same SVG board play_notebook's text fallback draws).
# No-ops gracefully outside IPython, so the script stays runnable as a .py.
# ---------------------------------------------------------------------------

def _render_board(board: chess.Board, caption: str = "") -> None:
    try:
        from IPython.display import SVG, clear_output, display
    except ImportError:
        return  # not in a notebook — nothing to render into
    clear_output(wait=True)
    last = board.peek() if board.move_stack else None
    check = board.king(board.turn) if board.is_check() else None
    display(SVG(chess.svg.board(board, lastmove=last, check=check, size=390)))
    if caption:
        print(caption)


# ---------------------------------------------------------------------------
# Openings.
# ---------------------------------------------------------------------------

def random_opening(rng: random.Random, plies: int) -> list[chess.Move]:
    board = chess.Board()
    moves = []
    for _ in range(plies):
        if board.is_game_over():
            break
        mv = rng.choice(list(board.legal_moves))
        moves.append(mv)
        board.push(mv)
    return moves


def unique_openings(rng: random.Random, count: int, plies: int) -> list[list[chess.Move]]:
    """`count` distinct random openings — a repeat is rerolled, so every opening
    (and therefore every colour-reversed pair) is genuinely different."""
    seen: set[tuple[str, ...]] = set()
    openings: list[list[chess.Move]] = []
    attempts = 0
    max_attempts = 200 * count + 1000  # generous; 3-ply openings have plenty of variety
    while len(openings) < count and attempts < max_attempts:
        attempts += 1
        opening = random_opening(rng, plies)
        key = tuple(m.uci() for m in opening)
        if key in seen:
            continue
        seen.add(key)
        openings.append(opening)
    if len(openings) < count:
        raise RuntimeError(
            f"Could only find {len(openings)} unique {plies}-ply openings "
            f"(wanted {count}); lower OPENINGS or raise OPENING_PLIES.")
    return openings


# ---------------------------------------------------------------------------
# One game.
# ---------------------------------------------------------------------------

def play_one_game(white, black, max_plies, opening, event, round_no, render):
    """Play a single game between two Players and return the python-chess Game."""
    white.reset()
    black.reset()

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = event
    game.headers["Round"] = str(round_no)
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    node = game

    for mv in opening:
        node = node.add_variation(mv)
        board.push(mv)

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mover = white if board.turn == chess.WHITE else black
        move = mover.move(board)
        node = node.add_variation(move)
        board.push(move)
        if render:
            side = white.name if board.turn == chess.BLACK else black.name  # who moved
            _render_board(board, f"{side} played {board.peek().uci()}  "
                                 f"(ply {board.ply()})")

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        outcome = board.outcome(claim_draw=True)
        reason = outcome.termination.name.lower().replace("_", " ") if outcome else "unknown"
    else:
        result = "1/2-1/2"
        reason = f"adjudicated draw ({max_plies}-ply cap)"
    game.headers["Result"] = result
    game.headers["Termination"] = reason
    return game


# ---------------------------------------------------------------------------
# Match summary.
# ---------------------------------------------------------------------------

def summarize(a_scores: list[float], name_a: str, name_b: str) -> str:
    """Total-score summary for player A (wins + 0.5*draws), with W-D-L, a 95% CI
    and an approx Elo diff. Mirrors compare_nets.summarize."""
    n = len(a_scores)
    wins = sum(1 for s in a_scores if s == 1.0)
    draws = sum(1 for s in a_scores if s == 0.5)
    losses = sum(1 for s in a_scores if s == 0.0)
    points = sum(a_scores)
    score = points / n if n else 0.0
    var = sum((s - score) ** 2 for s in a_scores) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else 0.0
    lo, hi = score - 1.96 * se, score + 1.96 * se
    if 0.0 < score < 1.0:
        elo_str = f"{-400.0 * math.log10(1.0 / score - 1.0):+.0f}"
    else:
        elo_str = "inf"
    verdict = ("A stronger" if lo > 0.5 else
               "B stronger" if hi < 0.5 else "not significant (CI spans 0.5)")
    return (
        f"{name_a} vs {name_b}: {n} games\n"
        f"  total score ({name_a}): {points:.1f} / {n}  ({score:.3f})\n"
        f"  W-D-L (for {name_a}): {wins}-{draws}-{losses}\n"
        f"  95% CI: {lo:.3f}..{hi:.3f}   approx Elo diff: {elo_str}\n"
        f"  -> {verdict}"
    )


def save_pgn(games, summary: str, path: str) -> None:
    """Write every game to `path`, with the match summary as %-escaped comment
    lines at the top (a leading '%' makes a line a PGN escape the parser skips,
    so the score is stored in the file without breaking game loading)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for line in summary.splitlines():
            f.write(f"% {line}\n")
        f.write("\n")
        for game in games:
            print(game, file=f)
            print(file=f)


# ---------------------------------------------------------------------------
# Driver: colour-balanced match, total score, PGN out.
# ---------------------------------------------------------------------------

def run_match(
    player_a: str = PLAYER_A,
    player_b: str = PLAYER_B,
    openings: int = OPENINGS,
    num_simulations: int = SIMS,
    mode: str = MODE,
    batch_size: int = BATCH_SIZE,
    sf_time: float = SF_TIME,
    sf_level: int | None = SF_LEVEL,
    sf_elo: int | None = SF_ELO,
    opening_plies: int = OPENING_PLIES,
    max_plies: int = MAX_PLIES,
    seed: int | None = SEED,
    render: bool = RENDER,
    out_pgn: str | None = OUT_PGN,
) -> list[chess.pgn.Game]:
    """Play a colour-balanced match between two opponents: `openings` unique
    openings, each twice with colours swapped (2 x openings games). Prints a short
    progress line per game and a total-score summary; writes all games + the
    summary to the PGN. Returns the list of finished Games.

    player_a / player_b: "random", "stockfish[:opts]", or a weights filename.
    """
    total = 2 * openings
    out_pgn = out_pgn or _default_out_pgn()
    converter = Converter()
    net_cache: dict[str, BigNetwork] = {}
    a = make_player(player_a, converter, net_cache, mode, num_simulations,
                    batch_size, sf_time, sf_level, sf_elo, seed)
    b = make_player(player_b, converter, net_cache, mode, num_simulations,
                    batch_size, sf_time, sf_level, sf_elo, seed)
    name_a, name_b = a.name, b.name
    print(f"A = {name_a}")
    print(f"B = {name_b}")
    print(f"Match: {openings} unique {opening_plies}-ply openings x 2 colours = "
          f"{total} games\n")

    event = f"{name_a} vs {name_b}"
    rng = random.Random(seed)
    opening_list = unique_openings(rng, openings, opening_plies)

    a_scores: list[float] = []
    games: list[chess.pgn.Game] = []
    try:
        for i, opening in enumerate(opening_list):
            for a_is_white in (True, False):
                white, black = (a, b) if a_is_white else (b, a)
                game = play_one_game(
                    white, black, max_plies, opening, event, i + 1, render)
                white_score = _RESULT_SCORE[game.headers["Result"]]
                a_scores.append(white_score if a_is_white else 1.0 - white_score)
                games.append(game)
                print(f"Game {len(games)}/{total}: {game.headers['Result']} "
                      f"({game.headers['Termination']})", flush=True)
    finally:
        # Always shut the stockfish subprocess(es) down, even on Ctrl-C / error.
        a.close()
        b.close()

    summary = summarize(a_scores, name_a, name_b)
    print("\n" + summary)

    save_pgn(games, summary, out_pgn)
    print(f"\nWrote {len(games)} games + summary -> {out_pgn}")
    return games


def main():
    ap = argparse.ArgumentParser(description="Colour-balanced net/stockfish/random match.")
    ap.add_argument("--player-a", "--net-a", dest="player_a", default=PLAYER_A,
                    help="net weights filename, 'stockfish[:opts]', or 'random'")
    ap.add_argument("--player-b", "--net-b", dest="player_b", default=PLAYER_B,
                    help="net weights filename, 'stockfish[:opts]', or 'random'")
    ap.add_argument("--openings", type=int, default=OPENINGS,
                    help="unique openings; total games = 2 x this")
    ap.add_argument("--mode", default=MODE, choices=["mcts", "raw"],
                    help="net move selection")
    ap.add_argument("--sims", type=int, default=SIMS, help="MCTS sims/move (net, mcts mode)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--sf-time", type=float, default=SF_TIME,
                    help="stockfish seconds/move")
    ap.add_argument("--sf-level", type=int, default=SF_LEVEL,
                    help="stockfish skill level 0-20")
    ap.add_argument("--sf-elo", type=int, default=SF_ELO,
                    help="stockfish UCI_Elo (overrides --sf-level)")
    ap.add_argument("--opening-plies", type=int, default=OPENING_PLIES)
    ap.add_argument("--max-plies", type=int, default=MAX_PLIES)
    ap.add_argument("--seed", type=int, default=SEED, help="seed openings + random player")
    ap.add_argument("--render", action="store_true",
                    help="draw the SVG board (only visible in a notebook)")
    ap.add_argument("--out", default=OUT_PGN, help="PGN output path (default: Kaggle working)")
    args = ap.parse_args()

    run_match(
        player_a=args.player_a,
        player_b=args.player_b,
        openings=args.openings,
        num_simulations=args.sims,
        mode=args.mode,
        batch_size=args.batch_size,
        sf_time=args.sf_time,
        sf_level=args.sf_level,
        sf_elo=args.sf_elo,
        opening_plies=args.opening_plies,
        max_plies=args.max_plies,
        seed=args.seed,
        render=args.render,
        out_pgn=args.out,
    )


if __name__ == "__main__":
    main()
