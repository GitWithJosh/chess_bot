"""Tests for evaluation modules: RandomMover and EvaluationAgainstOtherEngine."""

import chess
import numpy as np
import pytest

from reinforcement_learning.evaluation.random_move_opponent import RandomMover
from reinforcement_learning.evaluation.engine_opponent import EngineOpponent
from reinforcement_learning.evaluation.evaluation_against_other_engine import (
    EvaluationAgainstOtherEngine,
)
from reinforcement_learning.helpers.converter import Converter


# ---------------------------------------------------------------------------
# EngineOpponent ABC
# ---------------------------------------------------------------------------

class TestEngineOpponent:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            EngineOpponent()

    def test_random_mover_is_subclass(self):
        assert issubclass(RandomMover, EngineOpponent)


# ---------------------------------------------------------------------------
# RandomMover
# ---------------------------------------------------------------------------

class TestRandomMover:
    def test_returns_legal_move(self):
        mover = RandomMover()
        board = chess.Board()
        move = mover.choose_move(board)
        assert move in board.legal_moves

    def test_returns_move_object(self):
        mover = RandomMover()
        board = chess.Board()
        move = mover.choose_move(board)
        assert isinstance(move, chess.Move)

    def test_works_for_various_positions(self):
        mover = RandomMover()
        board = chess.Board()
        # Play a few moves
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        board.push_uci("g1f3")
        move = mover.choose_move(board)
        assert move in board.legal_moves

    def test_different_moves_over_many_calls(self):
        """RandomMover should not always return the same move."""
        mover = RandomMover()
        board = chess.Board()
        moves = set()
        for _ in range(50):
            move = mover.choose_move(board)
            moves.add(move)
        # With 20 legal moves, we should see variety
        assert len(moves) > 1


# ---------------------------------------------------------------------------
# EvaluationAgainstOtherEngine
# ---------------------------------------------------------------------------

class MockNetworkForEval:
    """A mock network for evaluation tests."""

    def __init__(self):
        self.converter = Converter()

    def predict(self, board_tensor):
        policy = np.ones(1858, dtype=np.float32) / 1858
        value = np.array([0.0])
        return policy, value

    def search_for_best_move(self, board, num_simulations):
        # Just return the first legal move
        return list(board.legal_moves)[0]


class TestEvaluationAgainstOtherEngine:
    def test_initialization(self):
        network = MockNetworkForEval()
        opponent = RandomMover()
        converter = Converter()

        evaluation = EvaluationAgainstOtherEngine(
            amount_of_games=2,
            network=network,
            other_engine=opponent,
            converter=converter,
            num_simulations=2,
        )

        assert evaluation.amount_of_games == 2
        assert evaluation.results == {"network_wins": 0, "draws": 0, "network_losses": 0}

    def test_play_games_updates_results(self):
        network = MockNetworkForEval()
        opponent = RandomMover()
        converter = Converter()

        evaluation = EvaluationAgainstOtherEngine(
            amount_of_games=2,
            network=network,
            other_engine=opponent,
            converter=converter,
            num_simulations=2,
        )

        evaluation.play_games()
        total = (
            evaluation.results["network_wins"]
            + evaluation.results["draws"]
            + evaluation.results["network_losses"]
        )
        assert total == 2

    def test_play_games_stores_pgn_games(self):
        network = MockNetworkForEval()
        opponent = RandomMover()
        converter = Converter()

        evaluation = EvaluationAgainstOtherEngine(
            amount_of_games=2,
            network=network,
            other_engine=opponent,
            converter=converter,
            num_simulations=2,
        )

        evaluation.play_games()
        assert len(evaluation.games) == 2

    def test_save_pgn(self, tmp_path):
        network = MockNetworkForEval()
        opponent = RandomMover()
        converter = Converter()

        evaluation = EvaluationAgainstOtherEngine(
            amount_of_games=1,
            network=network,
            other_engine=opponent,
            converter=converter,
            num_simulations=2,
        )

        evaluation.play_games()
        pgn_path = str(tmp_path / "test.pgn")
        evaluation.save_pgn(pgn_path)

        import os
        assert os.path.exists(pgn_path)
        with open(pgn_path) as f:
            content = f.read()
        assert "Event" in content

    def test_half_games_as_white_half_as_black(self):
        """Network should play half games as white, half as black."""
        network = MockNetworkForEval()
        opponent = RandomMover()
        converter = Converter()

        evaluation = EvaluationAgainstOtherEngine(
            amount_of_games=4,
            network=network,
            other_engine=opponent,
            converter=converter,
            num_simulations=2,
        )

        evaluation.play_games()

        white_games = sum(1 for g in evaluation.games if g.headers["White"] == "Network")
        black_games = sum(1 for g in evaluation.games if g.headers["Black"] == "Network")
        assert white_games == 2
        assert black_games == 2

    def test_play_game_without_mcts(self):
        """Test raw network output (no search) mode."""
        network = MockNetworkForEval()
        opponent = RandomMover()
        converter = Converter()

        evaluation = EvaluationAgainstOtherEngine(
            amount_of_games=1,
            network=network,
            other_engine=opponent,
            converter=converter,
            num_simulations=None,  # No MCTS
        )

        evaluation.play_games()
        total = (
            evaluation.results["network_wins"]
            + evaluation.results["draws"]
            + evaluation.results["network_losses"]
        )
        assert total == 1