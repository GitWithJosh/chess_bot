"""
Interactive viewer for the supervised PGN games in supervised_data/.

Run this file to open a GUI window that:

  * draws a random game from the PGN batches (respecting the filters);
  * shows, along the top, which Batch the game came from, both players' ELO,
    the time control, the result, and the current move count;
  * renders the board and lets you step through the moves — click ◀ / ▶ (or the
    |◀ / ▶| jump buttons), or hit "Autoplay" to advance automatically every
    800 ms. The from/to squares of the last move played are highlighted.
  * On the right are filters: ELO range (both players must fall inside it), move
    count range, outcome and time control. "New random game" draws a fresh game
    matching whatever filters are set; with autoplay on, the next game loads
    automatically when one finishes (no end-of-game popup).

    python supervised_learning/visualize_games.py

Only the chess + rendering stack is needed here (chess, chess.svg, cairosvg,
PIL, tkinter) — no network / Stockfish, unlike inspect_network.py.
"""

import io
import os
import pickle
import random
import re
from collections import namedtuple

# ===========================================================================
# Config — edit these
# ===========================================================================

AUTOPLAY_DELAY_MS = 800     # default autoplay delay; the slider overrides this
BOARD_PX = 560              # board render size (pixels); snapped to a multiple of 8

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supervised_data")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".game_index.pkl")

# Highlight colour for the last-moved-from/to squares (chess.svg "lastmove").
_FILL_LASTMOVE = "#f6f669cc"

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import chess
import chess.svg
import cairosvg
from PIL import Image, ImageTk

# A single game's index entry. Lightweight (a plain tuple) so ~300k of them fit
# comfortably in memory; the movetext is re-read on demand from `offset`.
Game = namedtuple("Game", "batch path offset welo belo tc result plies")

_HEADER_RE = re.compile(rb'\[(\w+)\s+"(.*)"\]')
_MOVENUM_RE = re.compile(rb"\d+\.")

_OUTCOME_MAP = {
    "Any": None,
    "White win": "1-0",
    "Black win": "0-1",
    "Draw": "1/2-1/2",
}


# ===========================================================================
# Indexing
# ===========================================================================

def _index_file(path: str, batch: str) -> list[Game]:
    """Stream one PGN file, recording metadata + the byte offset of each game's
    movetext line. Cheap: no board parsing, just header scraping + a ply count.
    """
    games: list[Game] = []
    cur: dict[bytes, bytes] = {}
    pos = 0
    with open(path, "rb") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(b"["):
                m = _HEADER_RE.match(stripped)
                if m:
                    cur[m.group(1)] = m.group(2)
            elif stripped:
                # The only non-empty, non-header line is the movetext.
                plies = len(_MOVENUM_RE.findall(stripped))
                games.append(Game(
                    batch=batch,
                    path=path,
                    offset=pos,
                    welo=_to_int(cur.get(b"WhiteElo")),
                    belo=_to_int(cur.get(b"BlackElo")),
                    tc=cur.get(b"TimeControl", b"?").decode("ascii", "replace"),
                    result=cur.get(b"Result", b"*").decode("ascii", "replace"),
                    plies=plies,
                ))
                cur = {}
            pos += len(line)
    return games


def _to_int(b: bytes | None) -> int:
    if not b:
        return 0
    try:
        return int(b)
    except ValueError:
        return 0


def _cache_signature(paths: list[str]) -> list[tuple]:
    return [(os.path.basename(p), os.path.getsize(p), os.path.getmtime(p)) for p in paths]


def build_index() -> list[Game]:
    """Load the per-game index, using a pickle cache when the PGN files are
    unchanged since it was written."""
    paths = sorted(
        (os.path.join(DATA_DIR, n) for n in os.listdir(DATA_DIR) if n.lower().endswith(".pgn")),
        key=lambda p: _natural_key(os.path.basename(p)),
    )
    if not paths:
        raise FileNotFoundError(f"No .pgn files found in {DATA_DIR}")

    sig = _cache_signature(paths)
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("sig") == sig:
                print(f"Loaded index from cache: {len(cached['games'])} games.")
                return cached["games"]
        except Exception:
            pass  # stale/corrupt cache — rebuild

    games: list[Game] = []
    for i, path in enumerate(paths, 1):
        batch = os.path.splitext(os.path.basename(path))[0]
        n_before = len(games)
        games.extend(_index_file(path, batch))
        print(f"  [{i}/{len(paths)}] {batch}: {len(games) - n_before} games")
    print(f"Indexed {len(games)} games total.")

    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"sig": sig, "games": games}, f)
    except Exception as exc:
        print(f"(Could not write index cache: {exc})")
    return games


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def read_moves(game: Game) -> list[chess.Move]:
    """Re-read a game's movetext from disk and return its list of moves."""
    with open(game.path, "rb") as f:
        f.seek(game.offset)
        movetext = f.readline().decode("utf-8", "replace")

    board = chess.Board()
    moves: list[chess.Move] = []
    for tok in movetext.split():
        if not tok or tok[0].isdigit() or tok in ("*", "1-0", "0-1", "1/2-1/2"):
            continue  # move numbers ("12." / "12...") and the result token
        try:
            mv = board.parse_san(tok)
        except ValueError:
            continue
        board.push(mv)
        moves.append(mv)
    return moves


# ===========================================================================
# GUI
# ===========================================================================

class ViewerApp:
    def __init__(self, root: tk.Tk, games: list[Game]):
        self.root = root
        self.games = games
        self.time_controls = sorted({g.tc for g in games})

        # Global ranges for the ELO / move-count sliders (ignore missing ELOs).
        elos = [e for g in games for e in (g.welo, g.belo) if e > 0]
        self.elo_lo, self.elo_hi = (min(elos), max(elos)) if elos else (0, 4000)
        plies = [g.plies for g in games]
        self.ply_lo, self.ply_hi = (min(plies), max(plies)) if plies else (0, 0)

        self.game: Game | None = None
        self.moves: list[chess.Move] = []
        self.boards: list[chess.Board] = []   # boards[i] = position after i plies
        self.ply = 0
        self.orientation = chess.WHITE

        self.autoplay = False
        self._after_id: str | None = None
        self._sync_ply = False                # guards the move-slider feedback loop
        self._imgs: dict = {}                 # keep PhotoImage refs alive

        self.sq = max(8, BOARD_PX // 8)
        self.bpx = self.sq * 8
        self.OX = 24                          # left margin (rank labels)
        self.OY = 6                           # top pad
        self.BLABEL = 22                      # bottom strip (file labels)

        self._build_ui()
        self.new_random_game()

    # -- UI construction --------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("Supervised games viewer")
        self.mono = tkfont.Font(family="Consolas", size=11)
        self.mono_sm = tkfont.Font(family="Consolas", size=9)
        self.hdr = tkfont.Font(family="Consolas", size=13, weight="bold")

        # --- top info bar ---
        top_bar = tk.Frame(self.root, bg="#222")
        top_bar.pack(side="top", fill="x")
        self.info_var = tk.StringVar(value="")
        tk.Label(top_bar, textvariable=self.info_var, font=self.hdr, fg="#eee",
                 bg="#222", anchor="w", padx=10, pady=6).pack(side="left", fill="x", expand=True)

        # --- main area: board left, filters right ---
        main = tk.Frame(self.root)
        main.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(
            main,
            width=self.bpx + self.OX + 6,
            height=self.bpx + self.OY + self.BLABEL,
            highlightthickness=0, bg="#ffffff",
        )
        self.canvas.pack(side="left")

        self._build_filter_panel(main)

        # --- bottom navigation ---
        bottom = tk.Frame(self.root)
        bottom.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        prow = tk.Frame(bottom)
        prow.pack(side="top", fill="x", pady=(0, 4))
        tk.Label(prow, text="move", font=self.mono_sm).pack(side="left")
        self.ply_var = tk.IntVar(value=0)
        self.ply_scale = tk.Scale(prow, variable=self.ply_var, from_=0, to=1,
                                  resolution=1, orient="horizontal", showvalue=False,
                                  command=self._on_ply_slider)
        self.ply_scale.pack(side="left", fill="x", expand=True, padx=6)

        nav = tk.Frame(bottom)
        nav.pack(side="top")
        tk.Button(nav, text="|◀", width=4, command=lambda: self.goto(0)).pack(side="left", padx=2)
        tk.Button(nav, text="◀", width=4, command=lambda: self.step(-1)).pack(side="left", padx=2)
        self.autoplay_btn = tk.Button(nav, text="▶ Autoplay", width=12,
                                      command=self.toggle_autoplay,
                                      font=tkfont.Font(weight="bold"))
        self.autoplay_btn.pack(side="left", padx=8)
        tk.Button(nav, text="▶", width=4, command=lambda: self.step(+1)).pack(side="left", padx=2)
        tk.Button(nav, text="▶|", width=4, command=lambda: self.goto(len(self.moves))).pack(side="left", padx=2)
        tk.Button(nav, text="Flip", width=5, command=self.flip).pack(side="left", padx=(16, 2))

        tk.Label(nav, text="speed (ms)", font=self.mono_sm).pack(side="left", padx=(16, 2))
        self.speed_var = tk.IntVar(value=AUTOPLAY_DELAY_MS)
        tk.Scale(nav, variable=self.speed_var, from_=100, to=1000, resolution=100,
                 orient="horizontal", length=200, font=self.mono_sm,
                 showvalue=True).pack(side="left")

        self.status = tk.Label(bottom, text="", font=self.mono_sm, fg="#444", anchor="w")
        self.status.pack(side="top", fill="x", pady=(4, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Left>", lambda _e: self.step(-1))
        self.root.bind("<Right>", lambda _e: self.step(+1))
        self.root.bind("<space>", lambda _e: self.toggle_autoplay())

    def _build_filter_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(parent, padx=12)
        panel.pack(side="left", fill="both", expand=True)

        tk.Label(panel, text="Filters", font=self.hdr, anchor="w").pack(side="top", fill="x", pady=(0, 8))

        self.elo_min = tk.IntVar(value=self.elo_lo)
        self.elo_max = tk.IntVar(value=self.elo_hi)
        self.mv_min = tk.IntVar(value=self.ply_lo)
        self.mv_max = tk.IntVar(value=self.ply_hi)
        self.outcome = tk.StringVar(value="Any")
        self.tc_var = tk.StringVar(value="Any")

        self._slider_pair(panel, "ELO (both players)", self.elo_min, self.elo_max,
                          self.elo_lo, self.elo_hi, resolution=10)
        self._slider_pair(panel, "Move count (plies)", self.mv_min, self.mv_max,
                          self.ply_lo, self.ply_hi, resolution=1)

        self._dropdown_row(panel, "Outcome", self.outcome, list(_OUTCOME_MAP.keys()))
        self._dropdown_row(panel, "Time control", self.tc_var, ["Any"] + self.time_controls)

        tk.Button(panel, text="New random game", font=tkfont.Font(weight="bold"),
                  command=self.new_random_game).pack(side="top", fill="x", pady=(14, 4))

        self.match_lbl = tk.Label(panel, text="", font=self.mono_sm, fg="#666", anchor="w")
        self.match_lbl.pack(side="top", fill="x")

        hint = ("Filters apply to the next random game.\n"
                "Space = autoplay · ←/→ = step moves.")
        tk.Label(panel, text=hint, font=self.mono_sm, fg="#999",
                 anchor="w", justify="left").pack(side="bottom", fill="x", pady=(8, 0))

    def _slider_pair(self, parent, label, min_var, max_var, lo, hi, resolution) -> None:
        tk.Label(parent, text=label, font=self.mono, anchor="w").pack(side="top", fill="x")
        # Degenerate range (all games identical): a Scale needs from_ != to.
        if hi <= lo:
            hi = lo + resolution
        for tag, var in (("min", min_var), ("max", max_var)):
            row = tk.Frame(parent)
            row.pack(side="top", fill="x")
            tk.Label(row, text=tag, font=self.mono_sm, width=3, anchor="w").pack(side="left")
            tk.Scale(row, variable=var, from_=lo, to=hi, resolution=resolution,
                     orient="horizontal", font=self.mono_sm,
                     showvalue=True).pack(side="left", fill="x", expand=True)
        tk.Frame(parent, height=6).pack(side="top")

    def _dropdown_row(self, parent, label, var, values) -> None:
        row = tk.Frame(parent)
        row.pack(side="top", fill="x", pady=(0, 8))
        tk.Label(row, text=label, font=self.mono, width=14, anchor="w").pack(side="left")
        ttk.Combobox(row, textvariable=var, values=values, state="readonly",
                     font=self.mono_sm, width=14).pack(side="left", fill="x", expand=True)

    # -- coordinate mapping / rendering (mirrors inspect_network.py) -------

    def _board_image(self, board: chess.Board, lastmove) -> Image.Image:
        try:
            check = board.king(board.turn) if board.is_check() else None
        except Exception:
            check = None
        svg = chess.svg.board(
            board, orientation=self.orientation, coordinates=False,
            lastmove=lastmove, check=check, size=self.bpx,
            colors={"square light lastmove": _FILL_LASTMOVE,
                    "square dark lastmove": _FILL_LASTMOVE},
        )
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                               output_width=self.bpx, output_height=self.bpx)
        assert isinstance(png, bytes)
        return Image.open(io.BytesIO(png)).convert("RGBA")

    def draw_board(self) -> None:
        c = self.canvas
        c.delete("all")
        board = self.boards[self.ply]
        lastmove = self.moves[self.ply - 1] if self.ply > 0 else None
        self._imgs["board"] = ImageTk.PhotoImage(self._board_image(board, lastmove))
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

    def update_info(self) -> None:
        g = self.game
        if g is None:
            self.info_var.set("No game loaded")
            return
        self.info_var.set(
            f"{g.batch}    "
            f"White {g.welo}  vs  Black {g.belo}    "
            f"{g.tc}    {g.result}    "
            f"Move {self.ply} / {len(self.moves)}"
        )

    # -- filtering / game selection --------------------------------------

    def _filtered_games(self) -> list[Game]:
        emin, emax = sorted((self.elo_min.get(), self.elo_max.get()))
        mmin, mmax = sorted((self.mv_min.get(), self.mv_max.get()))
        want_result = _OUTCOME_MAP[self.outcome.get()]
        want_tc = self.tc_var.get()

        out = []
        for g in self.games:
            if g.welo < emin or g.belo < emin or g.welo > emax or g.belo > emax:
                continue
            if g.plies < mmin or g.plies > mmax:
                continue
            if want_result is not None and g.result != want_result:
                continue
            if want_tc != "Any" and g.tc != want_tc:
                continue
            out.append(g)
        return out

    def new_random_game(self) -> None:
        pool = self._filtered_games()
        self.match_lbl.config(text=f"{len(pool)} games match the filters")
        if not pool:
            self.set_status("No games match the current filters.", error=True)
            return
        self.load_game(random.choice(pool))

    def load_game(self, game: Game) -> None:
        moves = read_moves(game)
        board = chess.Board()
        boards = [board.copy(stack=False)]
        for mv in moves:
            board.push(mv)
            boards.append(board.copy(stack=False))

        self.game = game
        self.moves = moves
        self.boards = boards
        self.ply = 0
        self.refresh()
        self.set_status(f"Loaded {game.batch} game · {len(moves)} plies.")

    # -- navigation -------------------------------------------------------

    def goto(self, ply: int) -> None:
        if not self.moves and ply != 0:
            return
        self.ply = max(0, min(ply, len(self.moves)))
        self.refresh()

    def step(self, delta: int) -> None:
        self.goto(self.ply + delta)

    def _on_ply_slider(self, val) -> None:
        if self._sync_ply:
            return
        self.goto(int(float(val)))

    def flip(self) -> None:
        self.orientation = not self.orientation
        self.draw_board()

    def refresh(self) -> None:
        self.draw_board()
        self.update_info()
        # Keep the move slider in sync without re-triggering its callback.
        self._sync_ply = True
        self.ply_scale.config(to=max(1, len(self.moves)))
        self.ply_var.set(self.ply)
        self._sync_ply = False

    # -- autoplay ---------------------------------------------------------

    def toggle_autoplay(self) -> None:
        self.autoplay = not self.autoplay
        self.autoplay_btn.config(text="⏸ Pause" if self.autoplay else "▶ Autoplay")
        if self.autoplay:
            self._schedule_tick()
        else:
            self._cancel_tick()

    def _schedule_tick(self) -> None:
        self._after_id = self.root.after(self.speed_var.get(), self._autoplay_tick)

    def _cancel_tick(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _autoplay_tick(self) -> None:
        if not self.autoplay:
            return
        if self.ply < len(self.moves):
            self.goto(self.ply + 1)
        else:
            self.new_random_game()  # game finished — silently move on, no popup
        self._schedule_tick()

    # -- misc -------------------------------------------------------------

    def set_status(self, msg: str, error: bool = False) -> None:
        self.status.config(text=msg, fg=("#d62728" if error else "#444"))

    def on_close(self) -> None:
        self._cancel_tick()
        self.root.destroy()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("Building game index (first run scans the PGN files; later runs use a cache)…")
    games = build_index()
    if not games:
        raise SystemExit("No games found.")
    root = tk.Tk()
    ViewerApp(root, games)
    root.mainloop()


if __name__ == "__main__":
    main()
