# chess-engine

A fully playable chess engine with a pluggable backend interface — designed from the ground up to support custom AI agents without touching game logic or GUI.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-008000?style=flat-square&logoColor=white)
![Stockfish](https://img.shields.io/badge/Stockfish-UCI-8B4513?style=flat-square&logoColor=white)

## Overview

The engine is structured as a strict dependency hierarchy — GUI knows about Game, Game knows about the engine interface, but nothing flows the other way. Move generation, board representation, game rules, and rendering are all separate modules with no cross-dependencies. Any engine that implements `get_best_move(board_state) → Move` slots in without modifications to the rest of the stack. Stockfish is included as a reference engine via the UCI protocol, configurable from 800 to 2400 ELO.

## Architecture

```
main.py
  └── gui/gui.py          — rendering, input handling
        └── game/game.py  — rules, state, move history
              ├── move_generation/generator.py  — legal move generation
              │     └── board/board.py           — board representation
              └── engines/*                      — pluggable engine interface
```

| Module | Responsibility |
|---|---|
| `board/board.py` | Immutable `BoardState` with deepcopy for safe move simulation |
| `move_generation/generator.py` | Pseudo-legal generation + legality filtering, zero-recursion check detection |
| `game/game.py` | Move execution, castling, en passant, promotion, draw conditions |
| `engines/engine.py` | Abstract `ChessEngine` base class |
| `engines/stockfish_engine.py` | UCI wrapper with ELO/depth configuration |
| `gui/menu.py` | Game setup — mode, engine, color selection |
| `gui/gui.py` | Pygame rendering at 30 FPS, PNG piece graphics |

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew install stockfish   # macOS; adjust for your OS
python main.py
```

The menu will appear. Choose Human vs Human or Human vs Bot, select difficulty (800 / 1600 / 2400 ELO), and pick your color.

## Adding a Custom Engine

```python
from engines.engine import ChessEngine
from move_generation.move import Move

class MyEngine(ChessEngine):
    def get_best_move(self, board_state) -> Move | None:
        # board_state is read-only — use MoveGenerator to enumerate moves
        return move

    def name(self) -> str:
        return "MyEngine"
```

Pass it to `ChessGUI(game, white_engine, black_engine)` — nothing else needs to change.

## Project Structure

```
chess-engine/
├── board/
│   └── board.py                  # BoardState — immutable, copyable
├── move_generation/
│   ├── move.py                   # Move dataclass (from/to + promotion)
│   └── generator.py              # Legal move generation
├── game/
│   └── game.py                   # Game rules, status, move history
├── engines/
│   ├── engine.py                 # ChessEngine abstract base
│   ├── random_engine.py
│   ├── human_engine.py
│   └── stockfish_engine.py
├── gui/
│   ├── gui.py                    # Game screen
│   └── menu.py                   # Setup menu
├── utils/
│   ├── coordinates.py            # Algebraic notation ↔ board indices
│   └── fen.py                    # FEN parsing and generation
├── pieces-basic-png/             # Piece graphics
└── main.py
```

## Chess Rules Coverage

Castling, en passant, pawn promotion, check detection, checkmate, stalemate, threefold repetition, 50-move rule, and insufficient material are all handled.
