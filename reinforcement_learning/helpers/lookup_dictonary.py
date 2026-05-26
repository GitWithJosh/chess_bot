"""
Generate a lookup dictionary of all legal queen moves across all 64 squares.

Approach:
- Place a lone white queen on each square of an otherwise empty board.
- Collect all legal moves (UCI strings like "a1a2") reported by python-chess.
- Store them in a flat dict: {int_key: "uci_string", ...}
  Keys are assigned sequentially (0, 1, 2, …) in square order (a1→h8).

Dependencies:
    pip install chess
"""

import chess
import json


def build_move_lookup() -> dict[int, str]:
    """
    Returns a dict mapping a sequential integer key to a UCI move string.
    Moves are grouped by source square (a1=0 … h8=63), then ordered as
    python-chess yields them (roughly by target square index).
    """
    lookup: dict[int, str] = {}
    key = 0

    # ALL possible queen moves
    for square in chess.SQUARES:
        board = chess.Board(fen=None)
        board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
        board.turn = chess.WHITE

        for move in board.generate_pseudo_legal_moves():
            lookup[key] = move.uci()
            key += 1
    
    print("Key after Queens:", key)
    # ALL possible knight moves
    for square in chess.SQUARES:
        board = chess.Board(fen=None)
        board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
        board.turn = chess.WHITE

        for move in board.generate_pseudo_legal_moves():
            lookup[key] = move.uci()
            key += 1
    print("Key after Knights:", key)
    ### Place white pawn on the seventh rank
    ### Black bishops on the eigth rank but not on the same file as the pawn 
    ### Allows capturing and underpromotion for the lookup dictionary

    seventh_rank = chess.SquareSet(chess.BB_RANK_7)
    eigth_rank = chess.SquareSet(chess.BB_RANK_8)
    for i, square_white in enumerate(seventh_rank):
        board = chess.Board(fen=None)
        board.set_piece_at(square_white, chess.Piece(chess.PAWN, chess.WHITE))

        for j, square in enumerate(eigth_rank):
            if not j == i:
                board.set_piece_at(square, chess.Piece(chess.BISHOP, chess.BLACK))
        board.turn = chess.WHITE

        for move in board.generate_pseudo_legal_moves():

            move_uci = move.uci()
            if not move_uci.endswith("q"):
                lookup[key] = move_uci
                key += 1     

    print("Key after Pawns", key)
    ###
    ### Same for the second rank and black pieces
    ###
    # first_rank = chess.SquareSet(chess.BB_RANK_1)
    # second_rank = chess.SquareSet(chess.BB_RANK_2)
    # print(second_rank)
    # for i, square_primary in enumerate(second_rank):
    #     board = chess.Board(fen=None)
    #     board.set_piece_at(square_primary, chess.Piece(chess.PAWN, chess.BLACK))
    #     for j, square_secondary in enumerate(first_rank):
    #         if not j == i:
    #             board.set_piece_at(square_secondary, chess.Piece(chess.BISHOP, chess.WHITE))

    #     print("=========================================")
    #     print(board)
    #     print("=========================================")

    #     board.turn = chess.BLACK
    #     for move in board.generate_pseudo_legal_moves():
    #         lookup[key] = move.uci()
    #         key += 1

    return lookup


def main():
    lookup = build_move_lookup()

    print(f"Total moves in lookup: {len(lookup)}\n")

    # Show first 10 entries as a sanity check
    print("First 10 entries:")
    for k in range(min(10, len(lookup))):
        print(f"  {k}: '{lookup[k]}'")

    # Show last 10 entries as a sanity check
    print("Last 10 entries:")
    for k in range(len(lookup)-10, len(lookup)):
        print(f'{k}: {lookup[k]}')

    print("\n\n")

    # Show entries for the a1 queen as a block
    print("\nAll moves with source square a1:")
    a1_moves = [v for v in lookup.values() if v.startswith("a1")]
    print(f"  {a1_moves}")

    # Check duplicates
    list = []
    for key, item in lookup.items():
        list.append(item)
    
    as_set = set(list)
    print("Length as list:", len(list))
    print("Length of set:", len(as_set))
    

    with open("move_lookup.json", "w") as f:
        json.dump(lookup, f)
    print("\nSaved to queen_move_lookup.json")

    return lookup

if __name__ == "__main__":
    main()

