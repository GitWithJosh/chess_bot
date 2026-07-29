"""Benchmark MCTS ``batch_size`` under an EQUAL WALL-CLOCK budget, scored
against hardcoded Stockfish evaluations.

Run on a **P100** kernel, not dual T4: TensorFlow uses one GPU by default and
single-stream MCTS cannot fill two, so the second T4 idles.

Why equal wall-clock and not equal sims
---------------------------------------
An earlier version gave every batch size the same ``num_simulations``. That is
structurally unfair: a large batch is ~8x faster, so in real play it would get
~8x the sims. Comparing at equal sims measures a scenario that never happens.
Here each (batch_size, position) gets the same number of SECONDS and runs as
many sims as it can. That answers the only question that matters: which batch
size produces the best move per unit of time?

Scoring
-------
Stockfish (depth 12, MultiPV = every legal move) annotated each position
offline; Kaggle has no Stockfish binary, hence the hardcoded table. EVERY legal
move is annotated, not a top-N: an arbitrary top-N forces you to invent a
penalty for moves that fall outside it, and "outside the top 10" means very
different things in a dead-drawn position (whole move list within 3cp) than in
a sharp one (top 10 spanning 400cp). With every move scored, the net's choice
always gets an exact centipawn value and an exact rank, and nothing is assumed.

The headline metric is **centipawn loss**: how much worse the chosen move is
than Stockfish's best, from the side-to-move's POV. Lower is better. Losses are
capped (see CP_CAP) so a single blunder into a mate cannot dominate the mean.

Caveats worth remembering when reading the output:
  * Depth 12 is shallow. It is a decent referee for blunders, not an oracle for
    close calls. Differences of a few cp between configs are noise.
  * Fresh MCTS per (batch_size, position): empty eval cache, no reused subtree.
    Real play keeps both warm across a game, so absolute sims/s here is
    pessimistic. The comparison between batch sizes stays fair.
  * Dirichlet noise is off, so each search is deterministic given the weights.

Usage on Kaggle (Accelerator = GPU P100):

    !pip install chess -q       # cell 1
    # paste this file into cell 2 — it self-runs.

Custom settings:

    main(batch_sizes=(1, 8, 16, 32, 64), seconds_per_move=3.0)
"""

import glob
import os
import statistics
import sys
import time


# ---------------------------------------------------------------------------
# Path setup — works locally and when pasted into a Kaggle cell (no __file__).
# ---------------------------------------------------------------------------

def _find_repo_root() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.normpath(os.path.join(here, ".."))
        if os.path.isdir(os.path.join(cand, "reinforcement_learning")):
            return cand
    except NameError:
        pass  # pasted into a cell — walk the mounts instead
    for base in ("/kaggle/input", "/kaggle/working", os.getcwd()):
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if "reinforcement_learning" in dirnames:
                return dirpath
    raise RuntimeError(
        "Could not locate the repo root (the folder containing "
        "'reinforcement_learning/')."
    )


REPO_ROOT = _find_repo_root()
RL_DIR = os.path.join(REPO_ROOT, "reinforcement_learning")
for _p in (REPO_ROOT, RL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO_ROOT)  # Converter opens move_lookup.json relative to CWD

import chess
import numpy as np

from helpers.converter import Converter
from monte_carlo_tree_search.mcts_v2 import MCTS
from monte_carlo_tree_search.nodes_and_edges_v2 import Node
from networks.big_network import BigNetwork


# ---------------------------------------------------------------------------
# Positions — spread across phases and branching factors, since the collision
# rate under virtual loss depends on how wide the tree is.
# ---------------------------------------------------------------------------

POSITIONS: list[tuple[str, str]] = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("open-italian", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("closed-french", "rnbqkb1r/pp3ppp/4pn2/2ppP3/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5"),
    ("ruy-lopez", "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
    ("queens-gambit", "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    ("sicilian-najdorf", "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("kings-indian", "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 6"),
    ("caro-kann", "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4"),
    ("sicilian-middlegame", "r1bqk2r/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w kq - 0 9"),
    ("tactical-kingside", "r2q1rk1/pp1nbppp/2p1bn2/3p4/3P1B2/2NBPN2/PPQ2PPP/R3K2R w KQ - 0 11"),
    ("queenless-complex", "r4rk1/1bp2ppp/p1np1n2/1p2p3/4P3/1BPP1N2/PP3PPP/R1B2RK1 w - - 0 13"),
    ("isolated-qp", "r1bq1rk1/pp2bppp/2n1pn2/8/2BP4/2N1BN2/PP3PPP/R2Q1RK1 w - - 0 11"),
    ("kingside-attack", "r1bq1rk1/pp3ppp/2n1pn2/2bp4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 10"),
    ("black-to-move-mg", "r2q1rk1/pp2bppp/2n1bn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 b - - 0 10"),
    ("heavy-pieces", "3r1rk1/pp3ppp/2p5/3q4/3P4/2P2Q2/PP3PPP/3R1RK1 w - - 0 20"),
    ("bishop-pair-mg", "r4rk1/pp2bppp/2n1pn2/q7/2BP4/2N1BN2/PP3PPP/R2Q1RK1 w - - 0 12"),
    ("rook-endgame", "8/5pk1/6p1/7p/5P1P/4R1P1/r5K1/8 w - - 0 40"),
    ("pawn-endgame", "8/8/4k3/8/3K4/8/4P3/8 w - - 0 50"),
    ("lucena-ish", "8/8/8/5K2/2k5/8/4P3/8 w - - 0 55"),
    ("bishop-vs-knight", "8/5pk1/6p1/7p/5P1P/4B1P1/5NK1/8 w - - 0 45"),
    ("queen-endgame", "8/5pk1/6p1/7p/5P1P/6P1/3Q2K1/6q1 w - - 0 42"),
    ("rook-pawns-eg", "8/pp3pk1/8/8/8/8/PP3PK1/2R2r2 w - - 0 35"),
    ("knight-endgame", "8/5pk1/6p1/7p/5P1P/4N1P1/5K2/8 w - - 0 44"),
]

# ---------------------------------------------------------------------------
# Stockfish annotation — depth 12, MultiPV = every legal move, best move first.
# cp is from the side-to-move's POV. Mates folded to +/-(30000 - 10*N).
# Generated offline; Kaggle has no Stockfish binary.
# ---------------------------------------------------------------------------

SF_DEPTH = 12

SF_EVALS: dict[str, str] = {
    "startpos":
        "e2e4:40,d2d4:29,g1f3:25,e2e3:19,c2c3:12,c2c4:10,g2g3:10,b1c3:9,a2a3:0,d2d3:-11,f2f4:-16,b2b4:-17,h2h3:-19,a2a4:-23,b2b3:-31,h2h4:-43,b1a3:-61,g1h3:-64,f2f3:-86,g2g4:-109",
    "open-italian":
        "f3g5:25,d2d3:24,d2d4:18,d1e2:-13,c4d5:-17,b1c3:-18,e1g1:-20,c4b5:-44,h2h3:-53,c2c3:-75,c4b3:-85,c4d3:-86,a2a3:-88,a2a4:-95,b2b3:-96,c4e2:-98,h2h4:-119,g2g3:-123,b2b4:-145,h1g1:-163,g2g4:-169,c4f1:-169,b1a3:-181,e1f1:-188,e1e2:-204,h1f1:-213,f3g1:-225,f3h4:-250,f3e5:-297,c4f7:-343,c4a6:-448,f3d4:-486,c4e6:-544",
    "closed-french":
        "e5f6:392,c3e2:37,g1f3:20,c1e3:16,c1g5:9,d4c5:7,f1b5:5,c3a4:3,c1f4:-6,f1e2:-7,f1d3:-12,f2f4:-16,c3b5:-26,g1e2:-31,h2h4:-32,a2a3:-34,a2a4:-39,h2h3:-44,d1e2:-48,d1d2:-49,a1b1:-58,c3b1:-58,d1d3:-61,g2g3:-65,c1d2:-75,b2b3:-76,f2f3:-80,g1h3:-82,g2g4:-98,d1f3:-135,e1e2:-150,b2b4:-152,c1h6:-160,f1a6:-164,f1c4:-229,e1d2:-295,c3d5:-411,c3e4:-525,d1h5:-632,d1g4:-683",
    "ruy-lopez":
        "e1g1:110,c2c3:108,d2d4:96,b1c3:85,d2d3:76,h2h3:68,a4b3:54,a2a3:51,d1e2:49,c2c4:46,a4c6:21,h2h4:8,g2g3:1,b1a3:-1,h1f1:-32,b2b4:-33,e1e2:-34,h1g1:-36,e1f1:-39,g2g4:-47,f3g1:-63,f3e5:-189,b2b3:-356,f3h4:-399,f3g5:-403,f3d4:-453,a4b5:-528",
    "queens-gambit":
        "b1c3:27,g1f3:24,g2g3:19,c4d5:18,e2e3:11,d1c2:8,a2a3:-4,c1f4:-6,h2h3:-7,c1d2:-9,b1d2:-25,a2a4:-27,c1e3:-31,d1b3:-32,g1h3:-36,h2h4:-37,d1d3:-37,b2b3:-38,d1d2:-41,d1a4:-43,b1a3:-46,c4c5:-57,f2f4:-57,g2g4:-57,e2e4:-73,f2f3:-79,e1d2:-113,b2b4:-137,c1g5:-514,c1h6:-574",
    "sicilian-najdorf":
        "c1g5:48,c1e3:48,f2f3:42,f2f4:39,a2a3:38,f1e2:37,h2h3:36,d4b3:29,h2h4:26,f1c4:25,d1d3:20,h1g1:20,d1e2:20,g2g3:17,f1d3:16,a2a4:16,d4f3:12,c1d2:6,d1f3:-1,a1b1:-11,b2b3:-13,d4e2:-15,d1d2:-25,b2b4:-27,d4f5:-28,e1e2:-62,e1d2:-64,g2g4:-87,c3d5:-134,c3b1:-135,c3e2:-155,e4e5:-161,c3a4:-185,d4b5:-296,c1f4:-307,f1b5:-345,f1a6:-391,c3b5:-408,c1h6:-437,d4c6:-461,d4e6:-477,d1h5:-582,d1g4:-637",
    "kings-indian":
        "c1e3:84,h2h3:81,e4e5:74,e1g1:68,c1g5:66,d1b3:65,c1f4:64,h2h4:58,b2b4:57,d1c2:56,d1d3:51,d1a4:49,a1b1:49,a2a3:45,c1d2:44,d4d5:41,a2a4:41,f3g5:38,g2g3:38,f3d2:38,e1f1:35,d1d2:34,e2f1:26,e2d3:26,f3h4:19,b2b3:18,f3g1:17,h1g1:-6,h1f1:-12,e1d2:-22,c4c5:-80,c3b5:-119,c3a4:-123,c3d5:-126,g2g4:-128,c3b1:-147,f3e5:-420,c1h6:-493",
    "caro-kann":
        "c4d5:36,g1f3:35,e2e3:33,d1b3:12,a2a4:7,g2g3:6,d1d3:-1,c1d2:-4,h2h3:-8,d1a4:-10,c1g5:-29,c1f4:-29,h2h4:-29,d1c2:-33,a2a3:-34,d1d2:-37,f2f4:-38,b2b3:-40,c4c5:-50,c1e3:-53,e2e4:-53,f2f3:-58,g1h3:-67,a1b1:-81,c3b1:-83,c3a4:-135,e1d2:-138,g2g4:-150,b2b4:-182,c3d5:-423,c1h6:-457,c3b5:-475,c3e4:-604",
    "sicilian-middlegame":
        "c1d2:38,a2a3:36,d1e2:31,e3e4:30,h2h3:27,f1e1:26,b2b3:22,d1b3:21,d3c2:18,a1b1:18,f3e5:16,d3b5:12,d3b1:9,d1d2:9,c3e2:8,g2g3:8,d1e1:5,f3d2:5,a2a4:4,c3a4:4,f3g5:2,f3e1:1,c3b5:-1,d1c2:-1,c3b1:-3,g1h1:-3,f3h4:-4,h2h4:-4,d1a4:-4,d3e2:-6,b2b4:-92,g2g4:-156,c3d5:-407,d3c4:-452,d3h7:-479,d3a6:-508,d3f5:-558,d3e4:-573,d3g6:-601,c3e4:-613",
    "tactical-kingside":
        "f3g5:91,a1d1:85,h2h4:49,h2h3:43,a2a3:42,a1c1:42,e1g1:40,a1b1:40,a2a4:38,f4g3:32,e1c1:32,f4g5:23,c3a4:23,c2b3:22,f3e5:22,c2b1:20,f3d2:13,b2b3:12,c3d1:11,f3g1:10,c3e2:10,c3b1:3,c2d2:3,g2g3:3,h1g1:-1,f4e5:-1,d3f1:-6,c2d1:-7,e1f1:-8,c2c1:-10,d3f5:-15,d3e2:-18,c2a4:-19,c2e2:-20,f3h4:-21,e1e2:-31,e1d1:-34,h1f1:-35,e1d2:-41,b2b4:-47,e3e4:-93,g2g4:-120,d3h7:-346,f4h6:-346,c3b5:-398,c3d5:-400,f4c7:-440,f4d6:-461,d3a6:-477,d3c4:-491,f4b8:-508,d3b5:-516,d3g6:-552,d3e4:-655,c3e4:-655",
    "queenless-complex":
        "f3h4:60,c1e3:42,f1d1:41,c1g5:40,a2a4:40,f1e1:40,b3c2:37,h2h3:36,c1d2:34,b3d1:31,h2h4:30,a2a3:29,f3e1:28,g2g3:27,a1b1:17,f3g5:14,f3d2:12,g1h1:9,c3c4:-7,d3d4:-65,g2g4:-76,b3d5:-115,f3e5:-376,c1h6:-427,b3f7:-462,c1f4:-498,f3d4:-519,b3c4:-566,b3e6:-575,b3a4:-611",
    "isolated-qp":
        "a2a3:40,f1e1:34,f3e5:33,d1e2:33,a1c1:23,h2h3:20,c4b3:15,c4b5:11,a2a4:6,c4e2:5,h2h4:4,e3f4:3,d1c2:-1,d1b1:-1,a1b1:-6,d1d2:-7,g2g3:-13,d1e1:-15,g1h1:-15,d4d5:-16,c4d3:-19,f3g5:-20,b2b3:-21,e3g5:-24,d1a4:-24,c3e2:-26,d1c1:-26,d1d3:-26,e3c1:-27,f3h4:-27,c3a4:-30,f3e1:-33,c3b1:-36,c3b5:-41,d1b3:-46,e3d2:-91,f3d2:-93,b2b4:-123,g2g4:-150,e3h6:-378,c4a6:-487,c3e4:-492,c4e6:-493,c3d5:-575,c4d5:-589",
    "kingside-attack":
        "d4c5:493,d3h7:66,c3d5:38,a2a3:34,h2h3:29,f1e1:28,c1d2:25,d1e2:23,b2b3:20,d3c2:16,a1b1:16,f3e5:16,d3b1:13,f3g5:12,a2a4:11,g1h1:10,c3b5:8,d1b3:8,f3e1:7,f3d2:4,f3h4:4,d1e1:1,d1d2:-2,d3f5:-5,h2h4:-5,c3e2:-7,d3b5:-8,d1c2:-11,d1a4:-12,c3b1:-12,g2g3:-12,g2g4:-14,d3a6:-15,d3e2:-18,c3a4:-18,d3c4:-22,e3e4:-24,b2b4:-95,d3g6:-116,d3e4:-450,c3e4:-555",
    "black-to-move-mg":
        "e7d6:-106,h7h6:-115,a8c8:-116,e6g4:-121,a7a6:-124,f8e8:-128,d8d7:-131,d8c8:-132,g8h8:-133,e7b4:-136,d8b6:-139,g7g6:-140,c6b4:-147,a8b8:-151,d8c7:-152,a7a5:-155,d8e8:-157,d8b8:-157,f6g4:-161,e6d7:-164,f6d7:-168,f6h5:-168,f6e8:-169,d8d6:-171,d8a5:-177,e6c8:-177,c6b8:-182,b7b6:-183,c6a5:-188,h7h5:-211,b7b5:-271,f6e4:-280,g7g5:-332,e6h3:-404,e7c5:-485,e7a3:-525,c6e5:-565,e6f5:-596,c6d4:-631",
    "heavy-pieces":
        "f3d5:182,d1a1:86,b2b3:80,a2a3:71,b2b4:62,a2a4:53,f3d3:50,d1e1:34,f3e2:15,h2h3:10,f1e1:7,h2h4:7,d1d2:5,f3g3:3,f3e3:2,g1h1:0,d1c1:-2,d1d3:-7,f3g4:-8,d1b1:-10,f3f4:-12,f3h3:-14,c3c4:-26,f3f6:-531,f3f7:-620,f3e4:-639,g2g3:-662,f3h5:-673,f3f5:-674,g2g4:-753",
    "bishop-pair-mg":
        "d1e2:603,f1e1:599,d1b3:592,a1c1:590,c4b3:586,h2h3:575,c4b5:574,a2a3:574,d1b1:570,c3e2:568,d1d2:565,a2a4:565,d1e1:565,d1c2:561,d1d3:556,f3d2:552,e3d2:544,c4d3:544,g1h1:540,a1b1:539,c3b5:534,d1c1:531,f3e1:523,c4e2:520,e3c1:517,g2g3:514,d1a4:511,c3b1:511,e3f4:503,f3h4:502,d4d5:483,e3g5:482,f3g5:468,f3e5:464,h2h4:452,b2b4:423,g2g4:347,c4e6:269,c3a4:211,e3h6:88,c3e4:22,b2b3:16,c3d5:15,c4d5:-10,c4a6:-30",
    "rook-endgame":
        "g2h1:10,g2h3:6,g2g1:6,g2f1:4,g2f3:3,e3e2:-622",
    "pawn-endgame":
        "d4e4:542,e2e3:0,e2e4:0,d4d3:0,d4c3:0,d4c4:0,d4e3:0,d4c5:0",
    "lucena-ish":
        "f5e4:701,f5e5:684,e2e3:680,e2e4:679,f5f4:669,f5g4:655,f5g5:655,f5g6:0,f5e6:0,f5f6:0",
    "bishop-vs-knight":
        "g2f3:665,e3d4:648,f2e4:644,e3c5:644,e3b6:644,e3c1:638,g2g1:632,f2h3:632,g2h1:623,e3d2:623,e3a7:621,f2d3:620,f2h1:619,g2h3:614,g2f1:602,f4f5:601,g2h2:573,f2d1:572,f2g4:535,g3g4:514",
    "queen-endgame":
        "g2g1:604,g2f3:-21,g2h3:-24",
    "rook-pawns-eg":
        "g2f1:675,c1f1:633,c1c4:16,c1c7:6,c1c3:0,c1c2:-1,c1c5:-2,c1c8:-2,c1c6:-298,g2f3:-639,f2f4:-642,b2b4:-642,c1e1:-659,a2a3:-663,g2h3:-664,c1a1:-668,c1d1:-669,f2f3:-669,b2b3:-670,g2h2:-670,a2a4:-673,c1b1:-675,g2g3:-677",
    "knight-endgame":
        "f2f3:592,e3d5:587,f2e2:575,e3g2:568,f2g2:563,e3c4:562,e3f1:556,f2g1:550,f2e1:540,f2f1:537,e3d1:530,e3c2:525,f4f5:480,g3g4:475,e3g4:77,e3f5:26",
}

# Cap on per-move centipawn loss. A search that hangs a queen into mate would
# otherwise contribute ~30000 and single-handedly decide the mean. 1000cp (a
# queen) is already "catastrophic"; distinguishing degrees of catastrophe adds
# noise, not signal.
CP_CAP = 1000


def _parse_sf(blob: str) -> list[tuple[str, int]]:
    out = []
    for part in blob.split(","):
        uci, cp = part.rsplit(":", 1)
        out.append((uci, int(cp)))
    return out


SF: dict[str, list[tuple[str, int]]] = {k: _parse_sf(v) for k, v in SF_EVALS.items()}
SF_RANK: dict[str, dict[str, int]] = {
    name: {uci: i + 1 for i, (uci, _) in enumerate(moves)} for name, moves in SF.items()
}
SF_CP: dict[str, dict[str, int]] = {
    name: dict(moves) for name, moves in SF.items()
}


def _score_move(position: str, uci: str) -> tuple[int, int, int]:
    """(rank, cp_loss_capped, n_legal) for a chosen move.

    Every legal move is annotated, so a lookup miss means the move list and the
    position disagree — a bug, not a "garbage move". Surfaced loudly.
    """
    ranks, cps = SF_RANK[position], SF_CP[position]
    if uci not in ranks:
        raise KeyError(
            f"{position}: move {uci} is not in the Stockfish annotation. The "
            f"hardcoded table is stale or the FEN changed."
        )
    best_cp = SF[position][0][1]
    loss = min(best_cp - cps[uci], CP_CAP)
    return ranks[uci], loss, len(ranks)


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
        os.path.join(REPO_ROOT, "*.weights.h5"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[-1]
    raise FileNotFoundError("No sl_*.weights.h5 found. Pass weights_path=...")


def _report_device() -> None:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("!! No GPU visible — these timings will not transfer to a GPU run.")
        return
    print(f"GPUs visible to TensorFlow: {len(gpus)}")
    for g in gpus:
        try:
            det = tf.config.experimental.get_device_details(g)
            print(f"  {g.name}  {det.get('device_name', '?')}")
        except Exception:
            print(f"  {g.name}")
    if len(gpus) > 1:
        print("  NOTE: TF uses GPU:0 only. A single P100 beats dual T4 here.")


# ---------------------------------------------------------------------------
# Phase A — raw network throughput, no MCTS
# ---------------------------------------------------------------------------

def bench_raw_inference(
    net: BigNetwork,
    converter: Converter,
    batch_sizes: tuple[int, ...],
    repeats: int = 30,
) -> dict[int, float]:
    """Median seconds per predict_batch call, per batch size."""
    pool = [converter.board_to_input_tensor(chess.Board(fen)) for _, fen in POSITIONS]
    while len(pool) < max(batch_sizes):
        pool.append(pool[len(pool) % len(POSITIONS)])

    per_call: dict[int, float] = {}
    print("\n--- Phase A: raw network throughput (no MCTS) ---")
    print(f"{'batch':>6} {'ms/call':>10} {'pos/sec':>10} {'speedup':>9}")
    base = None
    for bs in batch_sizes:
        x = np.asarray(pool[:bs], dtype=np.float32)
        for _ in range(5):           # warm up this shape
            net.predict_batch(x)
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            net.predict_batch(x)
            samples.append(time.perf_counter() - t0)
        dt = float(np.median(samples))  # median: one host hiccup shouldn't count
        per_call[bs] = dt
        pps = bs / dt
        if base is None:
            base = pps
        print(f"{bs:>6} {dt * 1e3:>10.2f} {pps:>10.1f} {pps / base:>8.2f}x")
    return per_call


# ---------------------------------------------------------------------------
# Phase B — equal wall-clock MCTS
# ---------------------------------------------------------------------------

def _search_for_seconds(
    mcts: MCTS, root: Node, batch_size: int, seconds: float
) -> tuple[int, float]:
    """Run batched MCTS on ``root`` until the time budget is spent.

    Driven one batch-round at a time (num_simulations = batch_size per call) so
    the time check lands between rounds and never splits a batch. This is
    behaviourally identical to one long search: search_batched only expands the
    root on the first call, add_noise is off, and the tree/cache persist in the
    MCTS object across calls.

    Returns (sims_run, elapsed_seconds).
    """
    mcts.num_simulations = batch_size
    sims = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if root.is_terminal:
            break
        mcts.search_batched(root, add_noise=False, batch_size=batch_size)
        sims += batch_size
    return sims, time.perf_counter() - t0


def bench_mcts_timed(
    net: BigNetwork,
    converter: Converter,
    batch_sizes: tuple[int, ...],
    seconds_per_move: float,
    fpu_reduction: float | None,
    c_puct: float,
    verbose: bool = True,
) -> dict:
    """Give every (batch_size, position) the same wall-clock budget."""
    results: dict = {}
    print(f"\n--- Phase B: {seconds_per_move}s/move, {len(POSITIONS)} positions, "
          f"scored vs Stockfish depth {SF_DEPTH} ---")

    for bs in batch_sizes:
        rows = []
        t_config = time.perf_counter()
        for name, fen in POSITIONS:
            board = chess.Board(fen)
            # Fresh MCTS: empty cache, no reused subtree, so no config inherits
            # a warm cache from an earlier one.
            mcts = MCTS(
                network=net,
                converter=converter,
                num_simulations=bs,
                c_puct=c_puct,
                fpu_reduction=fpu_reduction,
            )
            root = Node(board.copy())

            sims, elapsed = _search_for_seconds(mcts, root, bs, seconds_per_move)
            move = mcts.get_best_move(root, temperature=0)
            rank, loss, n_legal = _score_move(name, move.uci())
            top_n = max(e.N for e in root.edges)
            rows.append({
                "position": name,
                "seconds": elapsed,
                "sims": sims,
                "move": move.uci(),
                "rank": rank,
                "cp_loss": loss,
                "n_legal": n_legal,
                "evals": mcts.cache_misses,
                "top_visit_frac": top_n / max(1, root.total_visits),
            })
            if verbose:
                print(f"  bs={bs:<3} {name:<20} sims={sims:<6} {move.uci():<6} "
                      f"rank={rank:>2}/{n_legal:<3} cp_loss={loss:>5}")
        results[bs] = rows
        print(f"  bs={bs:<3} config total: {time.perf_counter() - t_config:.1f}s")
    return results


def summarize(results: dict, seconds_per_move: float) -> None:
    batch_sizes = sorted(results)
    n_pos = len(POSITIONS)

    print("\n" + "=" * 86)
    print(f"SUMMARY — {seconds_per_move}s/move, {n_pos} positions, "
          f"Stockfish depth {SF_DEPTH}, cp_loss capped at {CP_CAP}")
    print("=" * 86)
    print(f"{'batch':>6} {'sims':>7} {'sims/s':>8} {'mean cp':>9} {'med cp':>7} "
          f"{'top1':>6} {'top3':>6} {'top10':>6} {'mean rk':>8} {'topvis%':>8}")

    for bs in batch_sizes:
        rows = results[bs]
        sims = sum(r["sims"] for r in rows) / n_pos
        sims_s = sum(r["sims"] for r in rows) / sum(r["seconds"] for r in rows)
        losses = [r["cp_loss"] for r in rows]
        mean_cp = sum(losses) / n_pos
        med_cp = statistics.median(losses)
        top1 = sum(r["rank"] == 1 for r in rows)
        top3 = sum(r["rank"] <= 3 for r in rows)
        top10 = sum(r["rank"] <= 10 for r in rows)
        mean_rk = sum(r["rank"] for r in rows) / n_pos
        topvis = sum(r["top_visit_frac"] for r in rows) / n_pos * 100
        print(f"{bs:>6} {sims:>7.0f} {sims_s:>8.0f} {mean_cp:>9.1f} {med_cp:>7.0f} "
              f"{top1:>3}/{n_pos:<2} {top3:>3}/{n_pos:<2} {top10:>3}/{n_pos:<2} "
              f"{mean_rk:>8.1f} {topvis:>8.1f}")

    print("\nHow to read this:")
    print("  Every batch size got the SAME seconds per position, so this is a")
    print("  like-for-like comparison of move quality per unit time.")
    print("  mean cp  — average centipawn loss vs Stockfish's best move. THE")
    print("             headline number. Lower is better.")
    print("  med cp   — median; if it is far below the mean, a few blunders are")
    print("             driving the average rather than a broad quality gap.")
    print("  sims     — how many simulations fit in the budget. More is only")
    print("             better if cp_loss actually improves with it.")
    print("  topvis%  — visit share of the chosen move. Low = virtual loss is")
    print("             smearing visits instead of converging.")
    print(f"\nDepth {SF_DEPTH} is a shallow referee: trust it on blunders, not on")
    print("a few cp between configs. If two batch sizes land within ~10 mean cp")
    print("of each other, treat them as tied and pick the simpler/faster one.")

    # Positions where configs disagree are where the signal is; the rest is
    # shared baseline that tells you nothing about batch_size.
    print("\nPositions where the batch sizes disagree:")
    any_disagree = False
    for i, (name, _) in enumerate(POSITIONS):
        moves = {bs: results[bs][i]["move"] for bs in batch_sizes}
        if len(set(moves.values())) == 1:
            continue
        any_disagree = True
        parts = " ".join(
            f"bs{bs}={moves[bs]}(r{results[bs][i]['rank']},"
            f"{results[bs][i]['cp_loss']}cp)" for bs in batch_sizes
        )
        print(f"  {name:<20} {parts}")
    if not any_disagree:
        print("  (none — batch_size had no effect on the chosen move anywhere)")


def main(
    batch_sizes: tuple[int, ...] = (1, 8, 16, 32, 64),
    seconds_per_move: float = 3.0,
    weights_path: str | None = None,
    fpu_reduction: float | None = 0.3,
    c_puct: float = 1.5,
    raw_repeats: int = 30,
    verbose: bool = True,
) -> dict:
    """Run both phases and print the summary. Returns the Phase B raw results."""
    _report_device()

    path = _find_weights(weights_path)
    net = BigNetwork()
    net.load(path)
    print(f"Loaded {os.path.basename(path)}")
    converter = Converter()

    # Trace the tf.function before timing anything (first call builds the graph).
    net.predict_batch(np.zeros((1, 8, 8, 20), dtype=np.float32))

    bench_raw_inference(net, converter, batch_sizes, repeats=raw_repeats)

    budget = len(POSITIONS) * len(batch_sizes) * seconds_per_move
    print(f"\nPhase B will take ~{budget / 60:.1f} min "
          f"({len(POSITIONS)} positions x {len(batch_sizes)} configs "
          f"x {seconds_per_move}s, fixed by construction)")

    results = bench_mcts_timed(
        net, converter, batch_sizes, seconds_per_move, fpu_reduction, c_puct,
        verbose=verbose,
    )
    summarize(results, seconds_per_move)
    return results


if __name__ == "__main__":
    main()
