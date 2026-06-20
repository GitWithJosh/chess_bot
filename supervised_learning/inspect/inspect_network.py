"""
Interactive visual inspector for the supervised BigNetwork.

Run this file to open a GUI window with an editable chess board:

  * paste a FEN in the box at the bottom and click "Load" to set the position;
  * drag pieces around the board to edit it (right-click removes a piece) — the
    FEN box updates automatically and any arrows are cleared;
  * click "Analyze" to query the network + Stockfish on the current position.

After analysing, the board shows arrows for the most likely policy moves (arrow
colour red->green AND thickness scale with the move probability), and the right
panel shows:

  * the policy head's top-N moves in SAN (e.g. "Nf3") with probabilities,
    colour-matched to their arrows;
  * the value head's Win / Draw / Loss probabilities (side-to-move perspective),
    as text and a stacked bar;
  * Stockfish's evaluation and its top moves, for comparison.

    python supervised_learning/inspect_network.py

The same BigNetwork architecture/config as the RL + supervised training side is
imported, so weights from supervised_learning/checkpoints load directly.
"""

import glob
import io
import os
import shutil
import sys
import threading

# Quieten TensorFlow before it is imported (transitively, via BigNetwork).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ===========================================================================
# Config  — edit these
# ===========================================================================

# Preset positions: the board opens on the first, and the ◀/▶ buttons cycle
# through them. You can also just paste any FEN into the box at the bottom.
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "2b5/8/3kr3/2pp4/P7/1PP1PQ2/2NBBK2/6RR w - - 0 1",
    "2b5/6p1/p1qkrp1p/1ppp4/P7/1PP1PQ2/2NBBK2/6RR w - - 0 1",
]

# Weights. None -> sl_best.weights.h5 in supervised_learning/checkpoints,
# else the newest sl_*.weights.h5 there. Or set an explicit path.
WEIGHTS_PATH: str | None = None

# Stockfish. None -> auto-detect (PATH, then known local installs below).
# Set an explicit path to the executable to override.
STOCKFISH_PATH: str | None = None
STOCKFISH_DEPTH = 16        # search depth per position
STOCKFISH_MULTIPV = 5       # how many top lines to request (capped at legal moves)

# Display.
TOP_POLICY_MOVES = 10       # how many policy moves to list in the panel
NUM_ARROWS = 6              # how many policy moves to draw as arrows on the board
BOARD_PX = 560              # board render size (pixels); snapped to a multiple of 8

# Extra places to look for the Stockfish executable (Windows-friendly).
_STOCKFISH_FALLBACKS = [
    r"C:\Users\dhoff\OneDrive\Dokumente\Studium\Grundlagen Programmierung"
    r"\Semester 4\Personal\Reinforcement Learning\Chess\stockfish"
    r"\stockfish-windows-x86-64-avx2.exe",
    "stockfish",
    "stockfish-windows-x86-64-avx2.exe",
]

# ===========================================================================
# Path setup — make the reinforcement_learning packages importable and put the
# CWD where Converter expects reinforcement_learning/move_lookup.json.
# ===========================================================================

def _find_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", ".."))
    if os.path.isdir(os.path.join(cand, "reinforcement_learning")):
        return cand
    raise RuntimeError(
        "Could not locate the chess_bot repo root (the folder containing "
        "'reinforcement_learning/')."
    )


REPO_ROOT = _find_repo_root()
RL_DIR = os.path.join(REPO_ROOT, "reinforcement_learning")
for _p in (REPO_ROOT, RL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO_ROOT)

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
import chess
import chess.svg
import chess.engine
import cairosvg
from PIL import Image, ImageTk
import matplotlib
from matplotlib.colors import to_hex

from networks.big_network import BigNetwork      # noqa: E402
from helpers.converter import Converter          # noqa: E402

# WDL / arrow colours.
_COL_WIN = "#2ca02c"
_COL_DRAW = "#9aa0a6"
_COL_LOSS = "#d62728"
_CMAP = matplotlib.colormaps["RdYlGn"]  # 0 -> red (least likely shown), 1 -> green


def _prob_hex(t: float) -> str:
    return to_hex(_CMAP(max(0.0, min(1.0, t))))


# ===========================================================================
# Resource discovery
# ===========================================================================

def find_weights() -> str:
    if WEIGHTS_PATH:
        if os.path.exists(WEIGHTS_PATH):
            return WEIGHTS_PATH
        raise FileNotFoundError(f"WEIGHTS_PATH not found: {WEIGHTS_PATH}")
    ckpt_dir = os.path.join(REPO_ROOT, "supervised_learning", "checkpoints")
    best = os.path.join(ckpt_dir, "sl_best.weights.h5")
    if os.path.exists(best):
        return best
    hits = sorted(glob.glob(os.path.join(ckpt_dir, "sl_*.weights.h5")))
    if hits:
        return hits[-1]
    raise FileNotFoundError(
        f"No weights found. Looked for sl_best / sl_*.weights.h5 in {ckpt_dir}."
    )


def find_stockfish() -> str | None:
    candidates: list[str] = []
    if STOCKFISH_PATH:
        candidates.append(STOCKFISH_PATH)
    on_path = shutil.which("stockfish")
    if on_path:
        candidates.append(on_path)
    candidates.extend(_STOCKFISH_FALLBACKS)
    for c in candidates:
        if not c:
            continue
        if os.path.isfile(c):
            return c
        resolved = shutil.which(c)
        if resolved:
            return resolved
    return None


# ===========================================================================
# Inference
# ===========================================================================

def query_network(net: BigNetwork, converter: Converter, board: chess.Board):
    """Return (moves, wdl).

    moves : list of (chess.Move, san_str, probability) over the legal moves,
            sorted by probability descending. Probabilities are the policy head
            re-normalised over legal moves only. Empty if the position has no
            legal moves (or move generation fails on an illegal board).
    wdl   : np.array([win, draw, loss]) from the side-to-move's perspective.
    """
    tensor = converter.board_to_input_tensor(board)
    policy, wdl = net.predict(np.asarray(tensor, dtype=np.float32))

    moves: list[tuple[chess.Move, str, float]] = []
    try:
        if board.legal_moves.count() > 0:
            masked = np.asarray(converter.mask_illegal_moves(board, policy))
            for mv in board.legal_moves:
                # Mirror the move into the network's friendly perspective, as
                # Converter.mask_illegal_moves indexed it.
                uci = mv.uci()
                if board.turn == chess.BLACK:
                    uci = converter._mirror_move_uci(uci)
                if mv.promotion == chess.QUEEN:
                    uci = uci[:-1]  # queen promo has no suffix in the lookup
                idx = converter.index_lookup.get(uci)
                prob = float(masked[idx]) if idx is not None else 0.0
                moves.append((mv, board.san(mv), prob))
            moves.sort(key=lambda m: m[2], reverse=True)
    except Exception:
        moves = []  # illegal board (e.g. missing king) — show value head only

    return moves, np.asarray(wdl, dtype=np.float32)


def _fmt_score(score: chess.engine.Score | None) -> str:
    """Format a side-to-move-relative Score as '+1.23' or '#3' / '#-2'."""
    if score is None:
        return "n/a"
    if score.is_mate():
        return f"#{score.mate()}"
    cp = score.score()
    return "n/a" if cp is None else f"{cp / 100:+.2f}"


def query_stockfish(engine, board: chess.Board):
    """Return {"eval": Score|None, "lines": [(san, Score), ...], "error"?: str}."""
    if engine is None:
        return None
    if not board.is_valid():
        return {"eval": None, "lines": [], "error": "illegal position"}
    if board.is_game_over():
        return {"eval": None, "lines": []}
    try:
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=STOCKFISH_DEPTH),
            multipv=STOCKFISH_MULTIPV,
        )
    except chess.engine.EngineError as exc:
        return {"eval": None, "lines": [], "error": str(exc)}
    if isinstance(infos, dict):
        infos = [infos]

    lines = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        score = info["score"].pov(board.turn)
        lines.append((board.san(pv[0]), score))
    pos_eval = infos[0]["score"].pov(board.turn) if infos else None
    return {"eval": pos_eval, "lines": lines}


# ===========================================================================
# GUI
# ===========================================================================

class InspectorApp:
    """A tkinter window: editable board on the left, analysis panel on the right."""

    def __init__(self, root, net, converter, engine, weights_path, presets):
        self.root = root
        self.net = net
        self.converter = converter
        self.engine = engine
        self.weights_name = os.path.basename(weights_path)
        self.presets = presets or [chess.STARTING_FEN]
        self.preset_idx = 0

        self.board = chess.Board(self.presets[0])
        self.orientation = chess.WHITE
        self.results = None          # (moves, wdl, sf) for the shown board, or None
        self.analyzing = False
        self.drag = None             # dict while a piece is being dragged
        self._arrow_hex: dict[chess.Move, str] = {}
        self._imgs: dict = {}        # keep PhotoImage refs alive
        self._piece_cache: dict[str, ImageTk.PhotoImage] = {}

        self.sq = max(8, BOARD_PX // 8)
        self.bpx = self.sq * 8
        self.OX = 24                 # left margin (rank labels)
        self.OY = 6                  # top pad
        self.BLABEL = 22             # bottom strip (file labels)

        self._build_ui()
        self.refresh_all()
        self.set_status("Loaded. Drag pieces or paste a FEN, then click Analyze.")

    # -- UI construction --------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("BigNetwork inspector")
        self.mono = tkfont.Font(family="Consolas", size=11)
        self.mono_sm = tkfont.Font(family="Consolas", size=9)
        self.hdr = tkfont.Font(family="Consolas", size=12, weight="bold")

        top = tk.Frame(self.root)
        top.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(
            top,
            width=self.bpx + self.OX + 6,
            height=self.bpx + self.OY + self.BLABEL,
            highlightthickness=0, bg="#ffffff",
        )
        self.canvas.pack(side="left")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)

        right = tk.Frame(top)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.wdl_canvas = tk.Canvas(right, height=26, highlightthickness=0, bg="#ffffff")
        self.wdl_canvas.pack(side="bottom", fill="x", pady=(6, 0))
        self.wdl_canvas.bind("<Configure>", lambda _e: self.draw_wdl())
        tk.Label(right, text="value head: Win / Draw / Loss", font=self.mono_sm,
                 fg="#666").pack(side="bottom", anchor="w")

        txt_frame = tk.Frame(right)
        txt_frame.pack(side="top", fill="both", expand=True)
        self.text = tk.Text(txt_frame, width=42, height=30, font=self.mono,
                            wrap="none", borderwidth=0, padx=4, pady=2)
        scroll = ttk.Scrollbar(txt_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set, state="disabled")
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_config("hdr", font=self.hdr)
        self.text.tag_config("dim", foreground="#777")
        self.text.tag_config("loss", foreground=_COL_LOSS)

        # --- bottom controls ---
        bottom = tk.Frame(self.root)
        bottom.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        row1 = tk.Frame(bottom)
        row1.pack(side="top", fill="x", pady=2)
        tk.Label(row1, text="FEN:", font=self.mono).pack(side="left")
        self.fen_var = tk.StringVar()
        self.fen_entry = tk.Entry(row1, textvariable=self.fen_var, font=self.mono_sm)
        self.fen_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.fen_entry.bind("<Return>", lambda _e: self.load_fen())
        tk.Button(row1, text="Load", command=self.load_fen).pack(side="left", padx=2)
        self.btn_analyze = tk.Button(row1, text="Analyze", command=self.analyze,
                                     font=tkfont.Font(weight="bold"))
        self.btn_analyze.pack(side="left", padx=2)

        row2 = tk.Frame(bottom)
        row2.pack(side="top", fill="x", pady=2)
        tk.Button(row2, text="◀", width=3, command=lambda: self.cycle_preset(-1)).pack(side="left")
        tk.Button(row2, text="▶", width=3, command=lambda: self.cycle_preset(+1)).pack(side="left")
        tk.Button(row2, text="Start pos", command=self.load_startpos).pack(side="left", padx=4)
        tk.Button(row2, text="Empty", command=self.load_empty).pack(side="left")
        tk.Button(row2, text="Flip", command=self.flip).pack(side="left", padx=4)
        self.turn_btn = tk.Button(row2, text="Turn: White", command=self.toggle_turn)
        self.turn_btn.pack(side="left")
        self.status = tk.Label(row2, text="", font=self.mono_sm, fg="#444", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=8)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -- coordinate mapping ----------------------------------------------

    def xy_to_square(self, x: int, y: int):
        xb, yb = x - self.OX, y - self.OY
        if not (0 <= xb < self.bpx and 0 <= yb < self.bpx):
            return None
        col, row = int(xb // self.sq), int(yb // self.sq)
        if self.orientation == chess.WHITE:
            file, rank = col, 7 - row
        else:
            file, rank = 7 - col, row
        return chess.square(file, rank)

    def square_center(self, square: int) -> tuple[float, float]:
        file, rank = chess.square_file(square), chess.square_rank(square)
        if self.orientation == chess.WHITE:
            col, row = file, 7 - rank
        else:
            col, row = 7 - file, rank
        return self.OX + (col + 0.5) * self.sq, self.OY + (row + 0.5) * self.sq

    # -- rendering --------------------------------------------------------

    def _board_image(self, exclude_square=None) -> Image.Image:
        board = self.board
        if exclude_square is not None:
            board = board.copy(stack=False)
            board.remove_piece_at(exclude_square)
        try:
            check = board.king(board.turn) if board.is_check() else None
        except Exception:
            check = None
        svg = chess.svg.board(board, orientation=self.orientation,
                              coordinates=False, check=check, size=self.bpx)
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                               output_width=self.bpx, output_height=self.bpx)
        return Image.open(io.BytesIO(png)).convert("RGBA")

    def piece_sprite(self, symbol: str) -> ImageTk.PhotoImage:
        if symbol not in self._piece_cache:
            piece = chess.Piece.from_symbol(symbol)
            svg = chess.svg.piece(piece, size=self.sq)
            png = cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                                   output_width=self.sq, output_height=self.sq)
            img = Image.open(io.BytesIO(png)).convert("RGBA")
            self._piece_cache[symbol] = ImageTk.PhotoImage(img)
        return self._piece_cache[symbol]

    def draw_board(self, exclude_square=None) -> None:
        c = self.canvas
        c.delete("all")
        self._imgs["board"] = ImageTk.PhotoImage(self._board_image(exclude_square))
        c.create_image(self.OX, self.OY, image=self._imgs["board"], anchor="nw")
        self._draw_coords()

    def _draw_coords(self) -> None:
        c = self.canvas
        files = "abcdefgh" if self.orientation == chess.WHITE else "hgfedcba"
        ranks = "87654321" if self.orientation == chess.WHITE else "12345678"
        fy = self.OY + self.bpx + self.BLABEL / 2
        for col, ch in enumerate(files):
            c.create_text(self.OX + (col + 0.5) * self.sq, fy, text=ch,
                          font=self.mono_sm, fill="#444")
        for row, ch in enumerate(ranks):
            c.create_text(self.OX / 2, self.OY + (row + 0.5) * self.sq, text=ch,
                          font=self.mono_sm, fill="#444")

    def draw_arrows(self) -> None:
        self._arrow_hex = {}
        if not self.results:
            return
        moves = self.results[0]
        drawable = [m for m in moves if m[2] > 0][:NUM_ARROWS]
        pmax = drawable[0][2] if drawable else 1.0
        for mv, _san, p in drawable:
            self._arrow_hex[mv] = _prob_hex(p / pmax if pmax > 0 else 0.0)
        # Draw weakest first so the likeliest arrow ends up on top.
        for mv, _san, p in reversed(drawable):
            t = p / pmax if pmax > 0 else 0.0
            x0, y0 = self.square_center(mv.from_square)
            x1, y1 = self.square_center(mv.to_square)
            w = 3 + 9 * t
            head = 9 + 11 * t
            self.canvas.create_line(
                x0, y0, x1, y1, fill=self._arrow_hex[mv], width=w, arrow="last",
                arrowshape=(head * 1.6, head * 2.0, head * 0.9), capstyle=tk.ROUND,
            )

    def draw_wdl(self) -> None:
        c = self.wdl_canvas
        c.delete("all")
        if not self.results:
            return
        wdl = self.results[1]
        w = float(c.winfo_width()) or 320.0
        h = float(c.winfo_height()) or 26.0
        x = 0.0
        for frac, col in ((wdl[0], _COL_WIN), (wdl[1], _COL_DRAW), (wdl[2], _COL_LOSS)):
            seg = w * float(frac)
            c.create_rectangle(x, 2, x + seg, h - 2, fill=col, width=0)
            x += seg

    def update_panel(self) -> None:
        t = self.text
        t.config(state="normal")
        t.delete("1.0", "end")

        turn = "White" if self.board.turn == chess.WHITE else "Black"
        t.insert("end", f"weights: {self.weights_name}\n", ("dim",))
        t.insert("end", f"{turn} to move", ("hdr",))
        if not self.board.is_valid():
            t.insert("end", "   (illegal position)", ("loss",))
        t.insert("end", "\n\n")

        if self.results is None:
            t.insert("end", "Position edited — click Analyze.\n", ("dim",))
            t.config(state="disabled")
            return

        moves, wdl, sf = self.results

        t.insert("end", "Policy head — top moves\n", ("hdr",))
        if not moves:
            t.insert("end", "  (no legal moves)\n", ("loss",))
        for i, (mv, san, p) in enumerate(moves[:TOP_POLICY_MOVES], 1):
            tag = f"mv{i}"
            t.tag_config(tag, foreground=self._arrow_hex.get(mv, "#222222"))
            t.insert("end", f"{i:2d}. {san:<7s}{p * 100:5.1f}%\n", (tag,))

        t.insert("end", "\nValue head (side-to-move WDL)\n", ("hdr",))
        w, d, l = float(wdl[0]), float(wdl[1]), float(wdl[2])
        t.insert("end", f"Win {w*100:5.1f}%   Draw {d*100:5.1f}%   Loss {l*100:5.1f}%\n")

        t.insert("end", "\nStockfish\n", ("hdr",))
        if sf is None:
            t.insert("end", "  not found — set STOCKFISH_PATH\n", ("loss",))
        elif sf.get("error"):
            t.insert("end", f"  {sf['error']}\n", ("loss",))
        elif not sf["lines"]:
            t.insert("end", "  (no moves)\n", ("dim",))
        else:
            t.insert("end", f"depth {STOCKFISH_DEPTH}   eval {_fmt_score(sf['eval'])}\n")
            for i, (san, score) in enumerate(sf["lines"], 1):
                t.insert("end", f"{i:2d}. {san:<7s}{_fmt_score(score)}\n")

        t.config(state="disabled")

    def refresh_all(self) -> None:
        """Full redraw from current board/results state, and sync the FEN box."""
        self.draw_board()
        self.draw_arrows()
        self.update_panel()
        self.draw_wdl()
        self.fen_var.set(self.board.fen())
        self.turn_btn.config(text=f"Turn: {'White' if self.board.turn else 'Black'}")

    # -- drag & drop / editing -------------------------------------------

    def on_press(self, e) -> None:
        if self.analyzing:
            return
        sq = self.xy_to_square(e.x, e.y)
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if piece is None:
            return
        self.drag = {"from": sq, "piece": piece}
        # Redraw the board without the lifted piece and float a sprite of it.
        self.draw_board(exclude_square=sq)
        sprite = self.piece_sprite(piece.symbol())
        self._imgs["drag"] = sprite
        self.drag["item"] = self.canvas.create_image(e.x, e.y, image=sprite, anchor="center")

    def on_motion(self, e) -> None:
        if self.drag:
            self.canvas.coords(self.drag["item"], e.x, e.y)

    def on_release(self, e) -> None:
        if not self.drag:
            return
        frm, piece = self.drag["from"], self.drag["piece"]
        to = self.xy_to_square(e.x, e.y)
        self.drag = None
        if to is not None and to != frm:
            self.board.remove_piece_at(frm)
            self.board.set_piece_at(to, piece)
            self._sanitize()
            self.results = None
        self.refresh_all()

    def on_right_click(self, e) -> None:
        if self.analyzing:
            return
        sq = self.xy_to_square(e.x, e.y)
        if sq is None or self.board.piece_at(sq) is None:
            return
        self.board.remove_piece_at(sq)
        self._sanitize()
        self.results = None
        self.refresh_all()

    def _sanitize(self) -> None:
        """After a raw piece edit, drop stale en-passant / castling state."""
        self.board.ep_square = None
        self.board.castling_rights = self.board.clean_castling_rights()

    # -- buttons ----------------------------------------------------------

    def load_fen(self) -> None:
        fen = self.fen_var.get().strip()
        try:
            self.board = chess.Board(fen)
        except ValueError as exc:
            self.set_status(f"Invalid FEN: {exc}", error=True)
            return
        self.results = None
        self.refresh_all()
        self.set_status("FEN loaded. Click Analyze.")

    def load_startpos(self) -> None:
        self.board = chess.Board()
        self.results = None
        self.refresh_all()
        self.set_status("Start position. Click Analyze.")

    def load_empty(self) -> None:
        self.board = chess.Board(None)  # empty board, white to move
        self.results = None
        self.refresh_all()
        self.set_status("Empty board — drag pieces on, then Analyze.")

    def cycle_preset(self, delta: int) -> None:
        self.preset_idx = (self.preset_idx + delta) % len(self.presets)
        try:
            self.board = chess.Board(self.presets[self.preset_idx])
        except ValueError as exc:
            self.set_status(f"Bad preset FEN: {exc}", error=True)
            return
        self.results = None
        self.refresh_all()
        self.set_status(f"Preset {self.preset_idx + 1}/{len(self.presets)}. Click Analyze.")

    def flip(self) -> None:
        self.orientation = not self.orientation
        self.refresh_all()

    def toggle_turn(self) -> None:
        self.board.turn = not self.board.turn
        self.board.ep_square = None
        self.results = None
        self.refresh_all()

    # -- analysis (threaded) ---------------------------------------------

    def analyze(self) -> None:
        if self.analyzing:
            return
        self.analyzing = True
        self.btn_analyze.config(state="disabled")
        self.set_status("Analyzing…")
        fen = self.board.fen()
        board = self.board.copy()
        threading.Thread(target=self._worker, args=(board, fen), daemon=True).start()

    def _worker(self, board: chess.Board, fen: str) -> None:
        err = None
        res = None
        try:
            moves, wdl = query_network(self.net, self.converter, board)
            sf = query_stockfish(self.engine, board)
            res = (moves, wdl, sf)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        self.root.after(0, lambda: self._on_analyzed(fen, res, err))

    def _on_analyzed(self, fen: str, res, err) -> None:
        self.analyzing = False
        self.btn_analyze.config(state="normal")
        if fen != self.board.fen():
            self.set_status("Board changed during analysis — discarded. Click Analyze.")
            return
        if err is not None:
            self.results = None
            self.update_panel()
            self.set_status("Analyze failed: " + err, error=True)
            return
        self.results = res
        self.refresh_all()
        self.set_status("Done.")

    # -- misc -------------------------------------------------------------

    def set_status(self, msg: str, error: bool = False) -> None:
        self.status.config(text=msg, fg=(_COL_LOSS if error else "#444"))

    def on_close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
        self.root.destroy()


# ===========================================================================
# Main
# ===========================================================================

def _valid_presets(fens) -> list[str]:
    out = []
    for fen in fens:
        try:
            chess.Board(fen)
            out.append(fen)
        except ValueError as exc:
            print(f"Skipping invalid preset FEN ({exc}): {fen}")
    return out or [chess.STARTING_FEN]


def main() -> None:
    weights_path = find_weights()
    print(f"Loading network from {os.path.basename(weights_path)} … (takes a moment)")
    net = BigNetwork()
    net.load(weights_path)
    converter = Converter()
    # Warm up the inference graph (traces the tf.function) so the first Analyze
    # is not extra-slow and any error surfaces now rather than mid-click.
    net.predict(np.zeros(net.input_shape, dtype=np.float32))

    sf_path = find_stockfish()
    engine = None
    if sf_path:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            print(f"Stockfish: {sf_path}")
        except Exception as exc:
            print(f"Could not start Stockfish ({exc}). Continuing without it.")
    else:
        print("Stockfish not found — set STOCKFISH_PATH to enable its column.")

    root = tk.Tk()
    InspectorApp(root, net, converter, engine, weights_path, _valid_presets(FENS))
    root.mainloop()


if __name__ == "__main__":
    main()
