"""
Head-to-head comparison of two supervised nets.

Because the data and the training script changed, comparing validation loss
between an old and a new checkpoint is no longer apples-to-apples. This plays
the two nets against each other instead and reports a match score.

Two move-selection modes:
    raw  : argmax of the (legal-masked) policy head, no search. The most
           sensitive probe of what the network actually learned — MCTS tends to
           paper over raw policy/value weaknesses.
    mcts : 200-sim MCTS (the deployment setting, matching play_notebook).

`search_for_best_move` is deterministic (no root noise, temperature 0), so two
fixed nets would otherwise replay one identical game forever. We seed each game
from a short random opening and play that same opening twice with colours
reversed, which both diversifies the games and cancels first-move/colour bias.

TF-heavy (it runs the nets). For the *where are they weak* breakdown, feed the
PGN this writes to phase_acpl.py.

Edit the CONFIG block below, then just run the file (no arguments):
    python supervised_learning/compare_nets.py
"""

import math
import os
import random
import sys

# ============================ CONFIG — edit, then run ============================
NET_A = "supervised_learning/checkpoints/sl_best.weights.h5"   # usually the newer net
NET_B = "supervised_learning/checkpoints/old_sl_best.weights.h5"  # the baseline
MODE = "raw"        # "raw" | "mcts" | "both"
SIMS = 150           # MCTS simulations per move
BATCH_SIZE = 16      # MCTS leaf batch size
OPENINGS = {         # distinct random openings per mode; total games = 2 x this
    "raw": 50,      # raw policy is one forward pass per move -> cheap, play lots
    "mcts": 8,      # mcts is ~200 sims per move -> expensive, keep modest (esp. CPU)
}
OPENING_PLIES = 3    # random plies used to seed each opening
SEED = 7
OUT_DIR = os.path.join("supervised_learning", "results")
MAX_PLIES = 400      # adjudicate a draw past this so a shuffling net can't hang the match
# ================================================================================

# --- make the reinforcement_learning packages importable (mirrors play_notebook) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
RL_DIR = os.path.join(REPO_ROOT, "reinforcement_learning")
for _p in (REPO_ROOT, RL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Converter opens "reinforcement_learning/move_lookup.json" relative to CWD.
os.chdir(REPO_ROOT)

import chess
import chess.pgn
import tensorflow as tf

from helpers.converter import Converter
from monte_carlo_tree_search.nodes_and_edges_v2 import mirror_move_uci
from networks.big_network import BigNetwork


def load_net(path: str) -> BigNetwork:
    net = BigNetwork()
    net.load(path)
    return net


def raw_move(net: BigNetwork, converter: Converter, board: chess.Board) -> chess.Move:
    """Greedy argmax over the legal-masked policy head (no search).

    Same decode path as evaluation_against_other_engine.py: the policy is in the
    side-to-move's *friendly* perspective, so a black move is mirrored back.
    """
    board_tensor = converter.board_to_input_tensor(board)
    policy, _value = net.predict(board_tensor)
    move_probs = converter.mask_illegal_moves(board, policy)
    best_index = int(tf.argmax(move_probs).numpy())
    move_uci = converter._get_move_from_index(best_index)
    if board.turn == chess.BLACK:
        move_uci = mirror_move_uci(move_uci)
    move = chess.Move.from_uci(move_uci)
    # Queen promotions are stored in the lookup without the 'q' suffix, so a
    # decoded pawn push to the back rank comes back with promotion=None. Restore
    # the queen, else board.san() writes an illegal bare 'b1' that breaks PGN parsing.
    if (move.promotion is None
            and chess.square_rank(move.to_square) in (0, 7)
            and board.piece_type_at(move.from_square) == chess.PAWN):
        move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move


def select_move(net, converter, board, mode, sims, batch_size):
    if mode == "raw":
        return raw_move(net, converter, board)
    return net.search_for_best_move(board, num_simulations=sims, batch_size=batch_size)


def random_opening(rng: random.Random, plies: int) -> list[chess.Move]:
    """A short random legal opening, shared by both colour assignments."""
    board = chess.Board()
    moves = []
    for _ in range(plies):
        if board.is_game_over():
            break
        mv = rng.choice(list(board.legal_moves))
        moves.append(mv)
        board.push(mv)
    return moves


def play_game(white_net, black_net, white_name, black_name, converter,
              mode, sims, batch_size, opening, event) -> tuple[float, chess.pgn.Game]:
    """Play one game. Returns (white_score in {1,0.5,0}, pgn game)."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = event
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    node = game

    for mv in opening:  # record the seeded opening as part of the movetext
        node = node.add_variation(mv)
        board.push(mv)

    while not board.is_game_over() and board.ply() < MAX_PLIES:
        net = white_net if board.turn == chess.WHITE else black_net
        move = select_move(net, converter, board, mode, sims, batch_size)
        node = node.add_variation(move)
        board.push(move)

    result = board.result(claim_draw=True) if board.is_game_over() else "1/2-1/2"
    game.headers["Result"] = result
    white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
    return white_score, game


def run_match(net_a, net_b, name_a, name_b, converter, mode, sims, batch_size,
              openings, opening_plies, seed):
    """Each opening is played twice with colours reversed. Returns (a_scores, games)."""
    rng = random.Random(seed)
    a_scores: list[float] = []
    games: list[chess.pgn.Game] = []
    event = f"{name_a} vs {name_b} [{mode}]"

    for i in range(openings):
        opening = random_opening(rng, opening_plies)
        for a_is_white in (True, False):
            if a_is_white:
                w_net, b_net, w_name, b_name = net_a, net_b, name_a, name_b
            else:
                w_net, b_net, w_name, b_name = net_b, net_a, name_b, name_a

            w_score, game = play_game(
                w_net, b_net, w_name, b_name, converter,
                mode, sims, batch_size, opening, event,
            )
            a_score = w_score if a_is_white else 1.0 - w_score
            a_scores.append(a_score)
            games.append(game)
            print(f"  [{mode}] opening {i + 1}/{openings} "
                  f"({'A=white' if a_is_white else 'A=black'}): "
                  f"{game.headers['Result']}  (A scored {a_score})", flush=True)

    return a_scores, games


def summarize(a_scores: list[float], name_a: str, name_b: str) -> str:
    n = len(a_scores)
    wins = sum(1 for s in a_scores if s == 1.0)
    draws = sum(1 for s in a_scores if s == 0.5)
    losses = sum(1 for s in a_scores if s == 0.0)
    score = sum(a_scores) / n
    # std error of the mean game score, then a normal-approx 95% CI
    mean = score
    var = sum((s - mean) ** 2 for s in a_scores) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else 0.0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    if 0.0 < score < 1.0:
        elo = -400.0 * math.log10(1.0 / score - 1.0)
        elo_str = f"{elo:+.0f}"
    else:
        elo_str = "inf"
    return (
        f"{name_a} vs {name_b}: {n} games\n"
        f"  W-D-L (for {name_a}): {wins}-{draws}-{losses}\n"
        f"  score: {score:.3f}  (95% CI {lo:.3f}..{hi:.3f})\n"
        f"  approx Elo diff: {elo_str}\n"
        f"  -> {'A stronger' if lo > 0.5 else 'B stronger' if hi < 0.5 else 'not significant (CI spans 0.5)'}"
    )


def save_pgn(games, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for game in games:
            print(game, file=f)
            print(file=f)
    print(f"  wrote {path}")


def net_name(path: str) -> str:
    return os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]


def main():
    name_a, name_b = net_name(NET_A), net_name(NET_B)
    print(f"Loading A={name_a}  ({NET_A})")
    net_a = load_net(NET_A)
    print(f"Loading B={name_b}  ({NET_B})")
    net_b = load_net(NET_B)
    converter = Converter()

    modes = ["raw", "mcts"] if MODE == "both" else [MODE]
    for mode in modes:
        openings = OPENINGS[mode]
        print(f"\n=== Match ({mode}{', ' + str(SIMS) + ' sims' if mode == 'mcts' else ''}, "
              f"{openings} openings -> {2 * openings} games) ===")
        a_scores, games = run_match(
            net_a, net_b, name_a, name_b, converter, mode, SIMS,
            BATCH_SIZE, openings, OPENING_PLIES, SEED,
        )
        print("\n" + summarize(a_scores, name_a, name_b))
        save_pgn(games, os.path.join(OUT_DIR, f"match_{name_a}_vs_{name_b}_{mode}.pgn"))


if __name__ == "__main__":
    main()
