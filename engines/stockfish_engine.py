"""Stockfish chess engine wrapper."""

import subprocess
import os
import queue
import threading
import time
from board.board import BoardState
from move_generation.move import Move
from move_generation.generator import MoveGenerator
from engines.engine import ChessEngine


class StockfishEngine(ChessEngine):
    """Interface to Stockfish UCI chess engine."""

    STDOUT_QUEUE_MAXSIZE = 1000
    QUEUE_PUT_TIMEOUT = 0.1
    MIN_POLL_INTERVAL = 0.01
    READER_JOIN_TIMEOUT = 0.2

    def __init__(self, depth=10, elo=None):
        """Initialize Stockfish engine.

        Args:
            depth: Search depth (higher = stronger)
            elo: ELO rating to limit strength (1320-3190), None = full strength
        """
        self.depth = depth
        self.elo = elo
        self.process = None
        self._stdout_queue = queue.Queue(maxsize=self.STDOUT_QUEUE_MAXSIZE)
        self._stdout_reader_thread = None
        self._stop_reader = threading.Event()

        try:
            self.process = subprocess.Popen(
                ['stockfish'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._stdout_reader_thread = threading.Thread(
                target=self._read_stdout,
                daemon=True,
            )
            self._stdout_reader_thread.start()
            self._send_command('uci')
            self._wait_for_response('uciok')
            self._configure()
        except FileNotFoundError:
            raise RuntimeError(
                "Stockfish not found. Install with: brew install stockfish"
            )

    def _configure(self):
        """Configure Stockfish settings."""
        if self.elo is not None:
            self._send_command('setoption name UCI_LimitStrength value true')
            self._send_command(f'setoption name UCI_Elo value {self.elo}')
        else:
            self._send_command('setoption name UCI_LimitStrength value false')

    def _send_command(self, cmd: str):
        """Send command to Stockfish."""
        if self.process:
            self.process.stdin.write(cmd + '\n')
            self.process.stdin.flush()

    def _read_stdout(self):
        """Continuously read Stockfish stdout into a queue."""
        if not self.process or not self.process.stdout:
            return
        while not self._stop_reader.is_set():
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            while not self._stop_reader.is_set():
                try:
                    self._stdout_queue.put(line, timeout=self.QUEUE_PUT_TIMEOUT)
                    break
                except queue.Full:
                    continue

    def _wait_for_response(self, marker: str, timeout=5) -> list[str]:
        """Read lines until marker found or timeout expires."""
        lines = []
        if not self.process:
            return lines

        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_timeout = min(remaining, self.MIN_POLL_INTERVAL)

            try:
                line = self._stdout_queue.get(timeout=wait_timeout)
            except queue.Empty:
                continue

            lines.append(line)
            if marker in line:
                break

        return lines

    def get_best_move(self, board_state: BoardState) -> Move | None:
        """Get best move using Stockfish."""
        if not self.process:
            return None

        legal_moves = MoveGenerator(board_state).get_legal_moves()
        if not legal_moves:
            return None

        fen = board_state.to_fen()
        self._send_command(f'position fen {fen}')
        self._send_command(f'go depth {self.depth}')

        best_move_uci = None
        for line in self._wait_for_response('bestmove'):
            if line.startswith('bestmove'):
                parts = line.split()
                if len(parts) >= 2:
                    best_move_uci = parts[1]
                break

        if not best_move_uci or best_move_uci == '(none)':
            return None

        return self._uci_to_move(best_move_uci, board_state)

    def _uci_to_move(self, uci: str, board_state: BoardState) -> Move | None:
        """Convert UCI notation (e2e4) to Move object."""
        if len(uci) < 4:
            return None

        from utils.coordinates import algebraic_to_indices

        try:
            from_row, from_col = algebraic_to_indices(uci[:2])
            to_row, to_col = algebraic_to_indices(uci[2:4])

            promotion = None
            if len(uci) > 4:
                # Convert single character to full piece name
                piece_map = {'q': 'queen', 'r': 'rook', 'b': 'bishop', 'n': 'knight'}
                promotion = piece_map.get(uci[4], uci[4])

            return Move(from_row, from_col, to_row, to_col, promotion)
        except (ValueError, IndexError):
            return None

    def name(self) -> str:
        """Return engine name with strength."""
        elo_str = f" ({self.elo})" if self.elo else ""
        return f"Stockfish{elo_str}"

    def __del__(self):
        """Cleanup Stockfish process."""
        if self.process:
            try:
                self._send_command('quit')
                self.process.wait(timeout=1)
            except:
                self.process.terminate()
        self._stop_reader.set()
        if self._stdout_reader_thread:
            self._stdout_reader_thread.join(timeout=self.READER_JOIN_TIMEOUT)
