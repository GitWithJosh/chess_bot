"""
Play against the supervised BigNetwork *with MCTS*, inside a Jupyter/Kaggle notebook.

Renders the board as an SVG and reads your moves as UCI (e.g. e2e4, e7e8q). MCTS
runs on whatever device TensorFlow sees — on a GPU-enabled Kaggle kernel the
search is interactive (a few seconds/move); on CPU it is not.

Everything stays in python-chess land: we keep one chess.Board and push moves
onto it, so the board's move_stack feeds the network's 7-move history planes for
free (no custom-board bridge needed, unlike the pygame GUI).

Usage in a Kaggle cell (set Accelerator = GPU in notebook settings first):
    
    %run supervised_learning/play_notebook.py
    play(num_simulations=200, human_color="white")

Pass weights_path=... to pick a specific checkpoint; otherwise the newest
sl_*.weights.h5 under /kaggle/working, /kaggle/input, or the repo is used.
"""

import glob
import os
import sys


# ---------------------------------------------------------------------------
# Path setup — make the reinforcement_learning packages importable, both
# locally and on Kaggle (mirrors train_supervised.py).
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
        "'reinforcement_learning/'). Pass weights_path and set the CWD manually."
    )


REPO_ROOT = _find_repo_root()
RL_DIR = os.path.join(REPO_ROOT, "reinforcement_learning")
for _p in (REPO_ROOT, RL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Converter opens "reinforcement_learning/move_lookup.json" relative to CWD.
os.chdir(REPO_ROOT)

import chess
import chess.svg
from IPython.display import SVG, clear_output, display

from networks.big_network import BigNetwork


# ---------------------------------------------------------------------------
# Network loading (cached so repeated play() calls don't rebuild the graph)
# ---------------------------------------------------------------------------

_NET: BigNetwork | None = None


def _find_weights(explicit: str | None) -> str:
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(explicit)
    patterns = [
        "/kaggle/working/checkpoints/sl_best.weights.h5",
        "/kaggle/working/checkpoints/sl_*.weights.h5",
        "/kaggle/input/**/sl_*.weights.h5",
        os.path.join(REPO_ROOT, "supervised_learning", "checkpoints", "sl_*.weights.h5"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        "No sl_*.weights.h5 found under /kaggle/working, /kaggle/input, or the "
        "repo's supervised_learning/checkpoints/. Pass weights_path=..."
    )


def load_network(weights_path: str | None = None) -> BigNetwork:
    """Build BigNetwork and load weights once; cached in a module global."""
    global _NET
    path = _find_weights(weights_path)
    net = BigNetwork()
    net.load(path)
    _NET = net
    print(f"Loaded {os.path.basename(path)}")
    return net


# ---------------------------------------------------------------------------
# Rendering + interactive loop
# ---------------------------------------------------------------------------

def _render(board: chess.Board, orientation: bool, status: str = "") -> None:
    clear_output(wait=True)
    last = board.peek() if board.move_stack else None
    check = board.king(board.turn) if board.is_check() else None
    display(SVG(chess.svg.board(
        board, lastmove=last, check=check, orientation=orientation, size=420,
    )))
    if status:
        print(status)


def _parse_human_move(board: chess.Board, text: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        print("  not valid UCI (expected e.g. e2e4 or e7e8q)")
        return None
    if move in board.legal_moves:
        return move
    # Allow a promotion typed without the piece — default to queen.
    promo = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    if promo in board.legal_moves:
        return promo
    print("  illegal move")
    return None


def play(
    num_simulations: int = 200,
    human_color: str = "white",
    weights_path: str | None = None,
    batch_size: int = 16,
) -> None:
    """Interactive game vs. the network's MCTS.

    num_simulations: MCTS rollouts per move. Higher = stronger but slower.
    human_color: "white" or "black".
    batch_size: leaves evaluated per network call (virtual-loss batching).
        Bigger = better GPU utilization, up to a point; 16-32 is a good range.
    """
    net = _NET if _NET is not None else load_network(weights_path)
    human = chess.WHITE if human_color.lower().startswith("w") else chess.BLACK
    orientation = human
    board = chess.Board()

    while not board.is_game_over():
        to_move = "White" if board.turn else "Black"
        if board.turn == human:
            _render(board, orientation, f"Move {board.fullmove_number} — your turn ({to_move})")
            text = input("Your move (UCI, or 'quit'): ").strip()
            if text.lower() in ("quit", "q", "exit", ""):
                print("Aborted.")
                return
            move = _parse_human_move(board, text)
            if move is None:
                continue
            board.push(move)
        else:
            _render(board, orientation,
                    f"Move {board.fullmove_number} — engine thinking "
                    f"({num_simulations} sims)…")
            move = net.search_for_best_move(
                board, num_simulations=num_simulations, batch_size=batch_size)
            board.push(move)

    outcome = board.outcome()
    reason = outcome.termination.name.lower().replace("_", " ") if outcome else "unknown"
    _render(board, orientation, f"Game over: {board.result()}  ({reason})")
