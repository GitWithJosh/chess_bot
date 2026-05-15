"""Pygame-based chess GUI - Complete redesign with promotion, game over screen, and improved UI."""

import os
import sys
import pygame
from typing import Optional
from board.board import BoardState
from move_generation.move import Move
from move_generation.generator import MoveGenerator
from game.game import Game, GameStatus
from engines.engine import ChessEngine
from utils.coordinates import indices_to_algebraic


# Color scheme
COLOR_BG = (45, 45, 45)
COLOR_BOARD_LIGHT = (240, 217, 181)
COLOR_BOARD_DARK = (181, 136, 99)
COLOR_HIGHLIGHT = (186, 202, 43)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (100, 160, 210)
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_SECONDARY = (200, 200, 200)
COLOR_PANEL = (60, 60, 60)
COLOR_PANEL_BORDER = (120, 120, 120)


class PromotionDialog:
    """Pawn promotion selection dialog."""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.result = None

    def show(self) -> str:
        """Show promotion dialog and return promotion piece (q/r/b/n)."""
        overlay = pygame.Surface(self.screen.get_size())
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))

        pieces = [('Queen', 'queen'), ('Rook', 'rook'), ('Bishop', 'bishop'), ('Knight', 'knight')]
        dialog_width = 300
        dialog_height = 250
        x = (self.screen.get_width() - dialog_width) // 2
        y = (self.screen.get_height() - dialog_height) // 2

        buttons = []
        for i, (name, char) in enumerate(pieces):
            btn_y = y + 80 + i * 40
            btn = pygame.Rect(x + 30, btn_y, 240, 35)
            buttons.append((btn, name, char))

        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'queen'  # Default to queen
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for btn, name, char in buttons:
                        if btn.collidepoint(event.pos):
                            return char
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return 'queen'
                    elif event.key == pygame.K_r:
                        return 'rook'
                    elif event.key == pygame.K_b:
                        return 'bishop'
                    elif event.key == pygame.K_n:
                        return 'knight'

            self.screen.blit(overlay, (0, 0))

            pygame.draw.rect(self.screen, COLOR_PANEL, (x, y, dialog_width, dialog_height))
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (x, y, dialog_width, dialog_height), 2)

            title = self.font_large.render("Promote Pawn", True, COLOR_TEXT_PRIMARY)
            self.screen.blit(title, (x + 60, y + 15))

            for btn, name, char in buttons:
                pygame.draw.rect(self.screen, COLOR_BUTTON, btn)
                text = self.font_small.render(f"{name} ({char.upper()})", True, COLOR_TEXT_PRIMARY)
                self.screen.blit(text, (btn.x + 30, btn.y + 8))

            pygame.display.flip()
            clock.tick(30)


class GameOverScreen:
    """Game over screen with replay/menu options."""

    def __init__(self, screen: pygame.Surface, game: Game):
        self.screen = screen
        self.game = game
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

    def show(self) -> str:
        """Show game over screen. Returns 'replay', 'menu', or 'close'."""
        overlay = pygame.Surface(self.screen.get_size())
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))

        dialog_width = 500
        dialog_height = 350
        x = (self.screen.get_width() - dialog_width) // 2
        y = (self.screen.get_height() - dialog_height) // 2

        replay_btn = pygame.Rect(x + 30, y + 210, 140, 50)
        menu_btn = pygame.Rect(x + 180, y + 210, 140, 50)
        close_btn = pygame.Rect(x + 330, y + 210, 140, 50)

        clock = pygame.time.Clock()
        start_time = pygame.time.get_ticks()
        delay_ms = 800  # 0.8 second delay

        # Get game status message
        status = self.game.get_status()
        if status == GameStatus.CHECKMATE:
            winner = 'BLACK' if self.game.board.active_color == 'white' else 'WHITE'
            message = f"CHECKMATE!\n{winner} WINS!"
        elif status == GameStatus.STALEMATE:
            message = "STALEMATE\nGAME DRAW"
        else:
            message = f"GAME OVER\n{status.value.upper()}"

        while True:
            elapsed = pygame.time.get_ticks() - start_time
            if elapsed < delay_ms:
                # Still in delay period, just render
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return 'menu'
            else:
                # Delay over, handle input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return 'menu'
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if replay_btn.collidepoint(event.pos):
                            return 'replay'
                        elif menu_btn.collidepoint(event.pos):
                            return 'menu'
                        elif close_btn.collidepoint(event.pos):
                            return 'close'
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            return 'replay'
                        elif event.key == pygame.K_m:
                            return 'menu'
                        elif event.key == pygame.K_ESCAPE:
                            return 'close'

            self.screen.blit(overlay, (0, 0))

            pygame.draw.rect(self.screen, COLOR_PANEL, (x, y, dialog_width, dialog_height))
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (x, y, dialog_width, dialog_height), 3)

            msg_lines = message.split('\n')
            for i, line in enumerate(msg_lines):
                text = self.font_large.render(line, True, (255, 100, 100))
                self.screen.blit(text, (x + 80, y + 40 + i * 50))

            # Only show buttons after delay
            if elapsed >= delay_ms:
                pygame.draw.rect(self.screen, COLOR_BUTTON, replay_btn)
                pygame.draw.rect(self.screen, COLOR_BUTTON, menu_btn)
                pygame.draw.rect(self.screen, COLOR_BUTTON, close_btn)

                replay_text = self.font_small.render("Replay (R)", True, COLOR_TEXT_PRIMARY)
                menu_text = self.font_small.render("Menu (M)", True, COLOR_TEXT_PRIMARY)
                close_text = self.font_small.render("Close (Esc)", True, COLOR_TEXT_PRIMARY)

                self.screen.blit(replay_text, (replay_btn.x + 15, replay_btn.y + 8))
                self.screen.blit(menu_text, (menu_btn.x + 20, menu_btn.y + 8))
                self.screen.blit(close_text, (close_btn.x + 15, close_btn.y + 8))
            else:
                # Show waiting message during delay
                wait_text = self.font_small.render("Processing...", True, COLOR_TEXT_SECONDARY)
                self.screen.blit(wait_text, (x + 150, y + 200))

            pygame.display.flip()
            clock.tick(30)


class ConfirmDialog:
    """Confirmation dialog."""

    def __init__(self, screen: pygame.Surface, title: str, message: str):
        self.screen = screen
        self.title = title
        self.message = message
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

    def show(self) -> str | None:
        """Show dialog. Returns 'yes' or 'no'."""
        overlay = pygame.Surface(self.screen.get_size())
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))

        dialog_width = 400
        dialog_height = 200
        x = (self.screen.get_width() - dialog_width) // 2
        y = (self.screen.get_height() - dialog_height) // 2

        yes_btn = pygame.Rect(x + 50, y + 130, 140, 40)
        no_btn = pygame.Rect(x + 210, y + 130, 140, 40)

        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if yes_btn.collidepoint(event.pos):
                        return "yes"
                    elif no_btn.collidepoint(event.pos):
                        return "no"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        return "yes"
                    elif event.key == pygame.K_n:
                        return "no"

            self.screen.blit(overlay, (0, 0))

            pygame.draw.rect(self.screen, COLOR_PANEL, (x, y, dialog_width, dialog_height))
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (x, y, dialog_width, dialog_height), 2)

            title_surf = self.font_medium.render(self.title, True, COLOR_TEXT_PRIMARY)
            self.screen.blit(title_surf, (x + 20, y + 20))

            msg_surf = self.font_small.render(self.message, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(msg_surf, (x + 20, y + 80))

            pygame.draw.rect(self.screen, COLOR_BUTTON, yes_btn)
            pygame.draw.rect(self.screen, COLOR_BUTTON, no_btn)

            yes_text = self.font_small.render("Yes (Y)", True, COLOR_TEXT_PRIMARY)
            no_text = self.font_small.render("No (N)", True, COLOR_TEXT_PRIMARY)

            self.screen.blit(yes_text, (yes_btn.x + 30, yes_btn.y + 8))
            self.screen.blit(no_text, (no_btn.x + 35, no_btn.y + 8))

            pygame.display.flip()
            clock.tick(30)


class ChessGUI:
    """Pygame GUI for chess game."""

    WINDOW_WIDTH = 1100
    WINDOW_HEIGHT = 800
    BOARD_SIZE = 8
    SQUARE_SIZE = 80
    BOARD_START_X = 50
    BOARD_START_Y = 50

    # Move history panel - smaller and to the right
    HISTORY_START_X = 750
    HISTORY_START_Y = 150
    HISTORY_WIDTH = 300
    HISTORY_HEIGHT = 600

    def __init__(self, game: Game, white_engine: ChessEngine, black_engine: ChessEngine):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Chess")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_move = pygame.font.Font(None, 22)  # Larger for move history
        self.font_small = pygame.font.Font(None, 20)

        self.game = game
        self.white_engine = white_engine
        self.black_engine = black_engine

        self.selected_square: Optional[tuple[int, int]] = None
        self.legal_moves_from_selected: list[Move] = []
        self.last_move: Optional[Move] = None
        self.running = True
        self.show_menu = False

        self.dragging = False
        self.drag_from_square: Optional[tuple[int, int]] = None

        self.history_scroll = 0
        self.history_moves_per_page = 13  # More compact
        self.game_over_shown = False
        self.board_flipped = False

        self.piece_images = self._load_piece_images()
        self.menu_button_img = self._load_button_image("menu.png", (40, 40))
        self.flip_button_img = self._load_button_image("right-left.png", (40, 40))

    def _load_piece_images(self) -> dict:
        """Load PNG piece images."""
        pieces = {}
        pieces_dir = os.path.join(os.path.dirname(__file__), "..", "pieces-basic-png")

        piece_files = {
            ('white', 'pawn'): 'white-pawn.png',
            ('white', 'knight'): 'white-knight.png',
            ('white', 'bishop'): 'white-bishop.png',
            ('white', 'rook'): 'white-rook.png',
            ('white', 'queen'): 'white-queen.png',
            ('white', 'king'): 'white-king.png',
            ('black', 'pawn'): 'black-pawn.png',
            ('black', 'knight'): 'black-knight.png',
            ('black', 'bishop'): 'black-bishop.png',
            ('black', 'rook'): 'black-rook.png',
            ('black', 'queen'): 'black-queen.png',
            ('black', 'king'): 'black-king.png',
        }

        for piece_key, filename in piece_files.items():
            path = os.path.join(pieces_dir, filename)
            if os.path.exists(path):
                img = pygame.image.load(path)
                pieces[piece_key] = pygame.transform.scale(img, (self.SQUARE_SIZE - 10, self.SQUARE_SIZE - 10))

        return pieces

    def _load_button_image(self, filename: str, size: tuple) -> pygame.Surface | None:
        """Load button image."""
        path = os.path.join(os.path.dirname(__file__), "..", "pieces-basic-png", filename)
        try:
            if os.path.exists(path):
                img = pygame.image.load(path)
                return pygame.transform.scale(img, size)
        except:
            pass
        return None

    def run(self) -> bool:
        """Main game loop."""
        while self.running:
            status = self.game.get_status()

            # Check for game over
            if status != GameStatus.ONGOING and not self.game_over_shown:
                self.game_over_shown = True
                game_over = GameOverScreen(self.screen, self.game)
                result = game_over.show()
                if result == 'replay':
                    return False  # Request new game
                elif result == 'menu':
                    return True  # Back to menu
                # If 'close', continue the loop to render the board

            self._handle_events()
            self._update_game()
            self._render()
            self.clock.tick(30)

        return self.show_menu

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_down(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._handle_mouse_up(event.pos)

            elif event.type == pygame.MOUSEWHEEL:
                self.history_scroll += event.y
                max_scroll = max(0, len(self.game.move_history) - self.history_moves_per_page)
                self.history_scroll = max(0, min(self.history_scroll, max_scroll))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._request_menu()

    def _handle_mouse_down(self, pos: tuple[int, int]):
        """Handle mouse down."""
        # Menu button (far right, top)
        if 1030 <= pos[0] <= 1080 and 30 <= pos[1] <= 80:
            self._request_menu()
            return

        # Flip board button (far right, below menu)
        if 1030 <= pos[0] <= 1080 and 90 <= pos[1] <= 140:
            self.board_flipped = not self.board_flipped
            return

        col = (pos[0] - self.BOARD_START_X) // self.SQUARE_SIZE
        row = (pos[1] - self.BOARD_START_Y) // self.SQUARE_SIZE

        # Convert coordinates if board is flipped
        if self.board_flipped:
            row = 7 - row
            col = 7 - col

        if not (0 <= row < 8 and 0 <= col < 8):
            return

        piece = self.game.board.get_piece(row, col)

        if piece and piece[0] == self.game.board.active_color:
            self.dragging = True
            self.drag_from_square = (row, col)
            self._update_selected_square(row, col)
        elif self.selected_square and (row, col) in [(m.to_row, m.to_col) for m in self.legal_moves_from_selected]:
            self._try_make_move(self.selected_square[0], self.selected_square[1], row, col)
        else:
            self.selected_square = None
            self.legal_moves_from_selected = []

    def _handle_mouse_up(self, pos: tuple[int, int]):
        """Handle mouse up."""
        if not self.dragging or not self.drag_from_square:
            self.dragging = False
            return

        col = (pos[0] - self.BOARD_START_X) // self.SQUARE_SIZE
        row = (pos[1] - self.BOARD_START_Y) // self.SQUARE_SIZE

        # Convert coordinates if board is flipped
        if self.board_flipped:
            row = 7 - row
            col = 7 - col

        self.dragging = False

        if not (0 <= row < 8 and 0 <= col < 8):
            return

        if (row, col) == self.drag_from_square:
            return

        self._try_make_move(self.drag_from_square[0], self.drag_from_square[1], row, col)

    def _try_make_move(self, from_row: int, from_col: int, to_row: int, to_col: int):
        """Try to make a move, handling promotion if needed."""
        # Find any legal move matching from/to squares (ignoring promotion_piece)
        legal_move = None
        for lm in self.legal_moves_from_selected:
            if lm.from_row == from_row and lm.from_col == from_col and lm.to_row == to_row and lm.to_col == to_col:
                legal_move = lm
                break

        if legal_move is None:
            piece = self.game.board.get_piece(to_row, to_col)
            if piece and piece[0] == self.game.board.active_color:
                self._update_selected_square(to_row, to_col)
            else:
                self.selected_square = None
                self.legal_moves_from_selected = []
            return

        if legal_move.promotion_piece:
            # Need to show promotion dialog
            promo_dialog = PromotionDialog(self.screen)
            promotion_piece = promo_dialog.show()
            legal_move = Move(from_row, from_col, to_row, to_col, promotion_piece)

        if self.game.make_move(legal_move):
            self.last_move = legal_move
            self.selected_square = None
            self.legal_moves_from_selected = []

    def _update_selected_square(self, row: int, col: int):
        """Update selected square and legal moves."""
        self.selected_square = (row, col)
        gen = MoveGenerator(self.game.board)
        self.legal_moves_from_selected = [
            m for m in gen.get_legal_moves()
            if m.from_row == row and m.from_col == col
        ]

    def _request_menu(self):
        """Request return to menu."""
        dialog = ConfirmDialog(self.screen, "Return to Menu?", "Your game will be lost.")
        result = dialog.show()
        if result == "yes":
            self.show_menu = True
            self.running = False

    def _update_game(self):
        """Update game state."""
        engine = (
            self.white_engine
            if self.game.board.active_color == 'white'
            else self.black_engine
        )

        move = engine.get_best_move(self.game.board)
        if move:
            if self.game.make_move(move):
                self.last_move = move
                self.selected_square = None
                self.legal_moves_from_selected = []

    def _draw_coordinates(self):
        """Draw file (a-h) and rank (1-8) labels around the board."""
        for i in range(8):
            # File labels below the board
            x = self.BOARD_START_X + i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
            y = self.BOARD_START_Y + 8 * self.SQUARE_SIZE + 12
            file_label = chr(ord('h') - i) if self.board_flipped else chr(ord('a') + i)
            text = self.font_small.render(file_label, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(text, text.get_rect(center=(x, y)))

            # Rank labels to the left of the board
            x = self.BOARD_START_X - 15
            y = self.BOARD_START_Y + i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
            rank_label = str(i + 1) if self.board_flipped else str(8 - i)
            text = self.font_small.render(rank_label, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(text, text.get_rect(center=(x, y)))

    def _render(self):
        """Render board and UI."""
        self.screen.fill(COLOR_BG)
        self._draw_board()
        self._draw_coordinates()
        self._draw_pieces()
        self._draw_move_history()
        self._draw_ui()
        pygame.display.flip()

    def _draw_board(self):
        """Draw chessboard."""
        for row in range(8):
            for col in range(8):
                display_row = (7 - row) if self.board_flipped else row
                display_col = (7 - col) if self.board_flipped else col

                x = self.BOARD_START_X + display_col * self.SQUARE_SIZE
                y = self.BOARD_START_Y + display_row * self.SQUARE_SIZE

                is_light = (row + col) % 2 == 0
                color = COLOR_BOARD_LIGHT if is_light else COLOR_BOARD_DARK

                if self.last_move:
                    if ((row, col) == (self.last_move.from_row, self.last_move.from_col) or
                        (row, col) == (self.last_move.to_row, self.last_move.to_col)):
                        color = (200, 190, 100)

                pygame.draw.rect(self.screen, color, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE))

                if self.selected_square == (row, col):
                    pygame.draw.rect(self.screen, COLOR_HIGHLIGHT, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE), 3)

                for move in self.legal_moves_from_selected:
                    if (move.to_row, move.to_col) == (row, col):
                        pygame.draw.circle(
                            self.screen,
                            COLOR_HIGHLIGHT,
                            (x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2),
                            8,
                        )

    def _draw_pieces(self):
        """Draw chess pieces."""
        for row in range(8):
            for col in range(8):
                if self.dragging and (row, col) == self.drag_from_square:
                    continue

                piece = self.game.board.get_piece(row, col)
                if piece:
                    display_row = (7 - row) if self.board_flipped else row
                    display_col = (7 - col) if self.board_flipped else col

                    x = self.BOARD_START_X + display_col * self.SQUARE_SIZE + 5
                    y = self.BOARD_START_Y + display_row * self.SQUARE_SIZE + 5
                    if piece in self.piece_images:
                        self.screen.blit(self.piece_images[piece], (x, y))

        if self.dragging and self.drag_from_square:
            piece = self.game.board.get_piece(self.drag_from_square[0], self.drag_from_square[1])
            if piece and piece in self.piece_images:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                x = mouse_x - self.SQUARE_SIZE // 2
                y = mouse_y - self.SQUARE_SIZE // 2
                self.screen.blit(self.piece_images[piece], (x, y))

    def _draw_move_history(self):
        """Draw move history panel."""
        # Panel background
        pygame.draw.rect(
            self.screen,
            COLOR_PANEL,
            (self.HISTORY_START_X, self.HISTORY_START_Y, self.HISTORY_WIDTH, self.HISTORY_HEIGHT),
        )
        pygame.draw.rect(
            self.screen,
            COLOR_PANEL_BORDER,
            (self.HISTORY_START_X, self.HISTORY_START_Y, self.HISTORY_WIDTH, self.HISTORY_HEIGHT),
            2,
        )

        # Title
        title = self.font_medium.render("Moves", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title, (self.HISTORY_START_X + 10, self.HISTORY_START_Y + 10))

        # Moves - display pairs (white and black) side-by-side
        moves = self.game.move_history
        y_offset = self.HISTORY_START_Y + 50

        start_idx = self.history_scroll
        end_idx = min(start_idx + self.history_moves_per_page * 2, len(moves))

        i = start_idx
        while i < end_idx:
            move_num = i // 2 + 1

            # White move
            if i < len(moves):
                white_move = moves[i]
                white_text = f"{move_num}. {white_move.from_square}-{white_move.to_square}"
                white_surf = self.font_move.render(white_text, True, COLOR_TEXT_PRIMARY)
                self.screen.blit(white_surf, (self.HISTORY_START_X + 10, y_offset))

            # Black move (on the same line, offset to the right)
            if i + 1 < len(moves):
                black_move = moves[i + 1]
                black_text = f"{black_move.from_square}-{black_move.to_square}"
                black_surf = self.font_move.render(black_text, True, COLOR_TEXT_SECONDARY)
                self.screen.blit(black_surf, (self.HISTORY_START_X + 160, y_offset))

            y_offset += 22
            i += 2

    def _draw_ui(self):
        """Draw UI elements."""
        status = self.game.get_status()

        status_text = f"Turn: {self.game.board.active_color.upper()}"
        if status != GameStatus.ONGOING:
            status_text = "Game Over"

        status_surf = self.font_medium.render(status_text, True, COLOR_TEXT_PRIMARY)
        self.screen.blit(status_surf, (self.BOARD_START_X + 700, 30))

        white_eng = self.white_engine.name()
        black_eng = self.black_engine.name()
        white_text = self.font_small.render(f"White: {white_eng}", True, COLOR_TEXT_SECONDARY)
        black_text = self.font_small.render(f"Black: {black_eng}", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(white_text, (self.BOARD_START_X + 700, 70))
        self.screen.blit(black_text, (self.BOARD_START_X + 700, 95))

        # Menu button (far right, top)
        pygame.draw.rect(self.screen, COLOR_PANEL, (1030, 30, 50, 50))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (1030, 30, 50, 50), 2)
        if self.menu_button_img:
            self.screen.blit(self.menu_button_img, (1035, 35))
        else:
            menu_text = self.font_small.render("☰", True, COLOR_TEXT_PRIMARY)
            self.screen.blit(menu_text, (1042, 38))

        # Flip board button (far right, below menu)
        pygame.draw.rect(self.screen, COLOR_PANEL, (1030, 90, 50, 50))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (1030, 90, 50, 50), 2)
        if self.flip_button_img:
            self.screen.blit(self.flip_button_img, (1035, 95))
        else:
            flip_text = self.font_small.render("⟲", True, COLOR_TEXT_PRIMARY)
            self.screen.blit(flip_text, (1040, 100))
