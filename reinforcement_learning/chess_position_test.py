import chess
import chess.svg

# Stellung aufbauen:
# Weißer Bauer auf d7, kann nach c8 und e8 schlagen
# Schwarze Figuren auf c8 (Turm) und e8 (Turm)
# Könige müssen auf dem Brett sein

board = chess.Board(fen=None)  # Leeres Brett

# Könige setzen
board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
board.set_piece_at(chess.H6, chess.Piece(chess.KING, chess.BLACK))

# Weißer Bauer auf d7 (7. Reihe)
board.set_piece_at(chess.D7, chess.Piece(chess.PAWN, chess.WHITE))

# Zwei schwarze Figuren auf der Grundreihe (8. Reihe),
# die der Bauer diagonal schlagen kann
board.set_piece_at(chess.C8, chess.Piece(chess.ROOK, chess.BLACK))
board.set_piece_at(chess.E8, chess.Piece(chess.ROOK, chess.BLACK))

# Weiß ist am Zug
print(board)
print(chess.SquareSet(chess.BB_RANK_1))
print(board.turn)
print("=========")
board = board.mirror()
print(board)
print(chess.SquareSet(chess.BB_RANK_1))
print(board.turn)

# print("=" * 50)
# print("Stellung (FEN):")
# print(board.fen())
# print("=" * 50)
# print()
# print(board)
# print()

# # Alle legalen Züge des Bauern auf d7 anzeigen
# pawn_moves = [m for m in board.legal_moves if m.from_square == chess.D7]

# print(f"Legale Züge des Bauern auf d7 ({len(pawn_moves)} Züge):")
# print("-" * 50)

# for move in pawn_moves:
#     target = chess.square_name(move.to_square)
#     captured = board.piece_at(move.to_square)
#     promo = chess.piece_name(move.promotion) if move.promotion else None

#     if captured:
#         cap_info = f"  ← schlägt {chess.piece_name(captured.piece_type)} auf {target}!"
#     else:
#         cap_info = f"  ← Vorwärtszug (keine Schlagmöglichkeit)"

#     print(f"  {board.san(move):10s} → {target} (Umwandlung: {promo}){cap_info}")

# print()

# # Schlagzüge hervorheben
# captures = [m for m in pawn_moves if board.is_capture(m)]
# print(f"Davon Schlagzüge mit Umwandlung: {len(captures)}")
# for move in captures:
#     print(f"  {board.san(move)}")

# print()
# print(board.legal_moves)