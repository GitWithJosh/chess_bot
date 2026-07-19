"""
Play against the supervised BigNetwork *with MCTS*, inside a Jupyter/Kaggle notebook.

Renders a click-to-move chess board out of core ipywidgets and runs MCTS on
whatever device TensorFlow sees — on a GPU-enabled Kaggle kernel the search is
interactive (a few seconds/move); on CPU it is not.

Why click-to-move and not drag-and-drop? True piece-dragging needs a custom
front-end widget (e.g. anywidget), and Kaggle's notebook UI is a locked,
prebuilt JupyterLab app that refuses to load third-party widget modules
("No version of module anywidget is registered"). Core ipywidgets (Button,
GridBox, HTML) are baked into that front-end, so a button-grid board always
works — no extra install, no kernel restart. You click your piece (legal
targets light up), then click the destination.

Everything stays in python-chess land: we keep one chess.Board and push moves
onto it, so the board's move_stack feeds the network's 7-move history planes for
free. The UI is event-driven: play() displays the board and returns immediately;
each click runs through a callback, so it keeps working after the cell finishes
(standard ipywidgets behaviour).

Usage on Kaggle (set Accelerator = GPU first):

  1. Attach a dataset containing just the `reinforcement_learning/` folder and
     the weights file (e.g. sl_best.weights.h5) — the script auto-discovers both
     anywhere under /kaggle/input, so their exact paths don't matter.
  2. Paste this whole file into one notebook cell and run it (it needs no
     __file__ — it walks the mounts to find the repo). Then in the next cell:

        !pip install chess -q     # ipywidgets is preinstalled on Kaggle
        play(num_simulations=800, human_color="black")   # or "white"/"random"

Pass weights_path=... to pick a specific checkpoint; otherwise the newest
sl_*.weights.h5 under /kaggle/working, /kaggle/input, or the repo is used.

If ipywidgets can't render in your environment, play_text() is the UCI-input
fallback.
"""

import glob
import os
import random
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
import chess.pgn
import chess.svg
from IPython.display import HTML, SVG, clear_output, display

from networks.big_network import BigNetwork


# ---------------------------------------------------------------------------
# Network loading (cached so repeated play() calls don't rebuild the graph)
# ---------------------------------------------------------------------------

_NET: BigNetwork | None = None


def _resolve_color(human_color: str) -> bool:
    """Turn a colour name into a chess.WHITE/BLACK bool. Rejects anything
    unrecognised rather than guessing — a silent fallback would hand you the
    wrong side for a whole game."""
    name = human_color.strip().lower()
    if name == "random":
        name = random.choice(("white", "black"))
        print(f"You play {name}.")
    if name in ("white", "w"):
        return chess.WHITE
    if name in ("black", "b"):
        return chess.BLACK
    raise ValueError(
        f"human_color must be 'white', 'black' or 'random' (got {human_color!r})")


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
# Click-to-move board (core ipywidgets — works on Kaggle's locked front-end)
# ---------------------------------------------------------------------------

_LIGHT, _DARK = "#f0d9b5", "#b58863"   # base square colours
_SEL, _TGT, _LAST = "#f6f669", "#86c06c", "#cdd26a"  # selected / target / last move
_PROMO_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def _piece_class(piece: chess.Piece | None) -> str | None:
    """CSS class carrying a piece's background image (None for an empty square)."""
    if piece is None:
        return None
    s = piece.symbol()
    return "cbp-" + ("w" if s.isupper() else "b") + s.upper()


def _piece_css() -> str:
    """A <style> block giving each piece class a data-URI background image — the
    python-chess vector art, i.e. the same pieces the SVG board draws. Also
    squares off the buttons (no border-radius/border) so the grid looks clean."""
    import base64
    rules = [
        ".cbsq{background-size:84%;background-repeat:no-repeat;"
        "background-position:center;border-radius:0!important;border:0!important;"
        "box-shadow:none!important;}",
        ".cb-mono input,.cb-mono textarea{font-family:monospace;}",
    ]
    for sym in "PNBRQKpnbrqk":
        piece = chess.Piece.from_symbol(sym)
        svg = chess.svg.piece(piece)
        uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
        rules.append(f".{_piece_class(piece)}{{background-image:url({uri});}}")
    return "<style>" + "\n".join(rules) + "</style>"


def _lichess_analysis_url(board: chess.Board) -> str:
    """Lichess analysis-board link for a position (runs engine analysis in the
    browser). FEN spaces become underscores; the rank slashes are fine in the path."""
    return "https://lichess.org/analysis/" + board.fen().replace(" ", "_")


def _game_pgn(board: chess.Board) -> str:
    """PGN for the whole game so far — copy into https://lichess.org/paste to
    import and analyse the full game."""
    game = chess.pgn.Game.from_board(board)
    return game.accept(chess.pgn.StringExporter(
        headers=True, variations=False, comments=False))


def play(
    num_simulations: int = 200,
    human_color: str = "white",
    weights_path: str | None = None,
    batch_size: int = 16,
    square_px: int = 72,
):
    """Click-to-move game vs. the network's MCTS.

    Click your piece (legal targets light up green), then click the destination.
    Displays the board and returns immediately; the game advances via button
    callbacks as you click (so it keeps working after the cell finishes).

    num_simulations: MCTS rollouts per move. Higher = stronger but slower.
    human_color: "white", "black" or "random" (the board flips to your side).
    batch_size: leaves evaluated per network call (virtual-loss batching);
        16-32 is a good range on GPU.
    square_px: edge length of each board square in pixels (pieces scale with it).
    """
    try:
        import ipywidgets as W
    except ImportError as e:
        raise ImportError(
            "The board GUI needs ipywidgets (preinstalled on Kaggle). "
            "Use play_text() for the UCI-input fallback."
        ) from e

    net = _NET if _NET is not None else load_network(weights_path)
    human = _resolve_color(human_color)
    orient_white = human == chess.WHITE
    board = chess.Board()
    state = {"sel": None, "targets": {}, "busy": False}  # targets: to_sq -> [moves]

    # Build the 8x8 button grid in display order (white at bottom by default).
    buttons: dict[int, "W.Button"] = {}
    disp_ranks = range(7, -1, -1) if orient_white else range(0, 8)
    disp_files = range(0, 8) if orient_white else range(7, -1, -1)
    grid_children = []
    for rank in disp_ranks:
        for file in disp_files:
            sq = chess.square(file, rank)
            b = W.Button(layout=W.Layout(width=f"{square_px}px", height=f"{square_px}px",
                                         margin="0px", padding="0px"))
            b.add_class("cbsq")  # squares off corners + carries the piece image
            b._sq = sq  # type: ignore[attr-defined]
            buttons[sq] = b
            grid_children.append(b)

    grid = W.GridBox(grid_children, layout=W.Layout(
        grid_template_columns=f"repeat(8, {square_px}px)",
        grid_template_rows=f"repeat(8, {square_px}px)",
        grid_gap="0px", border="3px solid #444", width="max-content"))
    status = W.HTML()
    promo_row = W.HBox([])
    board_width = f"{8 * square_px}px"
    analysis_link = W.HTML()
    fen_field = W.Text(description="FEN", layout=W.Layout(width=board_width))
    fen_field.add_class("cb-mono")
    pgn_area = W.Textarea(
        placeholder="Full-game PGN appears here when the game ends "
                    "(paste into lichess.org/paste to analyse the whole game).",
        layout=W.Layout(width=board_width, height="120px"))
    pgn_area.add_class("cb-mono")
    box = W.VBox([grid, promo_row, status, analysis_link, fen_field, pgn_area])

    def set_status(text: str):
        status.value = f"<div style='margin-top:6px;font-family:sans-serif'>{text}</div>"

    def update_info():
        fen_field.value = board.fen()
        analysis_link.value = (
            f"<a href='{_lichess_analysis_url(board)}' target='_blank' "
            "style='font-family:sans-serif'>▶ Analyse this position on Lichess</a>")

    piece_cls: dict[int, str | None] = {sq: None for sq in buttons}  # current piece class

    def refresh_board():
        last = board.peek() if board.move_stack else None
        for sq, b in buttons.items():
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            col = _LIGHT if (file + rank) % 2 else _DARK
            if last is not None and sq in (last.from_square, last.to_square):
                col = _LAST
            if sq in state["targets"]:
                col = _TGT
            if sq == state["sel"]:
                col = _SEL
            new = _piece_class(board.piece_at(sq))
            with b.hold_sync():  # coalesce this square's colour + piece into one update
                b.style.button_color = col
                if new != piece_cls[sq]:
                    if piece_cls[sq]:
                        b.remove_class(piece_cls[sq])
                    if new:
                        b.add_class(new)
                    piece_cls[sq] = new
        update_info()

    def clear_selection():
        state["sel"] = None
        state["targets"] = {}

    def select(sq: int):
        piece = board.piece_at(sq)
        if piece is None or piece.color != human:
            return
        targets: dict[int, list] = {}
        for m in board.legal_moves:
            if m.from_square == sq:
                targets.setdefault(m.to_square, []).append(m)
        if not targets:
            return
        state["sel"], state["targets"] = sq, targets
        refresh_board()

    def finish():
        outcome = board.outcome()
        reason = outcome.termination.name.lower().replace("_", " ") if outcome else "unknown"
        clear_selection()
        refresh_board()
        pgn_area.value = _game_pgn(board)
        set_status(f"Game over: {board.result()}  ({reason})")

    def engine_turn():
        while not board.is_game_over() and board.turn != human:
            state["busy"] = True
            set_status(f"Move {board.fullmove_number} — engine thinking "
                       f"({num_simulations} sims)…")
            mv = net.search_for_best_move(
                board, num_simulations=num_simulations, batch_size=batch_size)
            board.push(mv)
        state["busy"] = False
        refresh_board()
        if board.is_game_over():
            finish()
        else:
            set_status(f"Move {board.fullmove_number} — your turn")

    def play_move(move: chess.Move):
        clear_selection()
        promo_row.children = []
        board.push(move)
        refresh_board()
        if board.is_game_over():
            finish()
        else:
            engine_turn()

    def ask_promotion(moves: list):
        by_piece = {m.promotion: m for m in moves}
        btns = []
        for piece_type in _PROMO_ORDER:
            if piece_type not in by_piece:
                continue
            pb = W.Button(layout=W.Layout(width=f"{square_px}px", height=f"{square_px}px"))
            pb.add_class("cbsq")
            pb.add_class(_piece_class(chess.Piece(piece_type, human)))
            pb.style.button_color = _LIGHT
            pb.on_click(lambda _b, mv=by_piece[piece_type]: play_move(mv))
            btns.append(pb)
        promo_row.children = btns
        set_status("Choose promotion piece")

    def on_click(b):
        if state["busy"] or board.is_game_over() or board.turn != human:
            return
        sq = b._sq
        if state["sel"] is None:
            select(sq)
        elif sq in state["targets"]:
            moves = state["targets"][sq]
            if len(moves) > 1:        # promotion — same to-square, 4 piece types
                ask_promotion(moves)
            else:
                play_move(moves[0])
        elif sq == state["sel"]:
            clear_selection()
            refresh_board()
        else:
            clear_selection()
            select(sq)

    # Wire the (now-defined) handler onto every button.
    for b in buttons.values():
        b.on_click(on_click)

    display(HTML(_piece_css()))  # register piece-image classes (once per call)
    display(box)
    refresh_board()
    if board.turn == human:
        set_status(f"Move {board.fullmove_number} — your turn")
    else:
        engine_turn()
    # Intentionally return nothing: returning `box` would make Jupyter display
    # the board a second time as the cell's output value.


# ---------------------------------------------------------------------------
# Text fallback (SVG render + UCI input) — for environments without ipywidgets.
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
    promo = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    if promo in board.legal_moves:
        return promo
    print("  illegal move")
    return None


def play_text(
    num_simulations: int = 200,
    human_color: str = "white",
    weights_path: str | None = None,
    batch_size: int = 16,
) -> None:
    """UCI-input fallback game vs. the network's MCTS."""
    net = _NET if _NET is not None else load_network(weights_path)
    human = _resolve_color(human_color)
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
