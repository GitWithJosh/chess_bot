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
from engines.human_engine import HumanInputEngine
from utils.coordinates import indices_to_algebraic
from gui.assets import load_image, load_icon, ui_font

try:
    # Only used to name moves in SAN. The game itself runs on this repo's own
    # move generator, so a missing python-chess costs the notation, nothing more.
    import chess
except ImportError:
    chess = None


# Short names for the draw conditions. The enum values are long snake_case and
# do not fit the game over box at its largest font.
DRAW_LABELS = {
    GameStatus.THREEFOLD: "DRAW BY REPETITION",
    GameStatus.FIFTY_MOVE: "50-MOVE RULE",
    GameStatus.INSUFFICIENT: "INSUFFICIENT MATERIAL",
}

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

# Win / draw / loss meter, the same three colours the Kaggle notebook uses.
COLOR_WDL_WIN = (46, 158, 91)
COLOR_WDL_DRAW = (154, 154, 154)
COLOR_WDL_LOSS = (209, 73, 91)


class PromotionDialog:
    """Pawn promotion selection dialog."""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_large = ui_font(24, bold=True)
        self.font_small = ui_font(15)
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
        self.font_large = ui_font(30, bold=True)
        self.font_medium = ui_font(22, bold=True)
        self.font_small = ui_font(16)

    def _fit(self, line: str, max_width: int) -> pygame.Surface:
        """Render `line` at the largest of our sizes that fits `max_width`."""
        for font in (self.font_large, self.font_medium, self.font_small):
            surf = font.render(line, True, (255, 100, 100))
            if surf.get_width() <= max_width:
                return surf
        return surf

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

        # Get game status message. Second line is spelled out rather than taken
        # from the enum, whose values are long snake_case and overflowed the box.
        status = self.game.get_status()
        if status == GameStatus.CHECKMATE:
            winner = 'BLACK' if self.game.board.active_color == 'white' else 'WHITE'
            message = f"CHECKMATE!\n{winner} WINS!"
        elif status == GameStatus.STALEMATE:
            message = "STALEMATE\nGAME DRAW"
        else:
            message = f"GAME OVER\n{DRAW_LABELS.get(status, status.value.replace('_', ' ').upper())}"

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
                text = self._fit(line, dialog_width - 40)
                # Centred, so a long line stays inside the box either way.
                self.screen.blit(
                    text,
                    (x + (dialog_width - text.get_width()) // 2, y + 40 + i * 50),
                )

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
        self.font_medium = ui_font(20, bold=True)
        self.font_small = ui_font(16)

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

    # Right-hand column, stacked top to bottom: the win/draw/loss meters, the
    # search readout, then whatever height is left goes to the move history.
    PANEL_X = 750
    PANEL_W = 300
    # Starts below the menu and flip buttons, which sit at y 30-80 and 90-140 and
    # reach into this column's x range.
    PANEL_TOP = 150
    PANEL_BOTTOM = 770
    PANEL_GAP = 10

    WDL_BAR_H = 18            # the coloured bar itself
    WDL_CAPTION_GAP = 5       # caption baseline to the top of the bar
    WDL_PCT_GAP = 3           # bar to the percentages under it
    WDL_BLOCK_GAP = 8         # between one meter and the next

    THINK_ROWS = 5
    THINK_HEADER_H = 46       # title and column headings
    THINK_ROW_H = 21

    HISTORY_START_X = 750
    HISTORY_WIDTH = 300
    HISTORY_HEADER_H = 64     # title, plus room for the "more above" hint
    HISTORY_ROW_H = 24
    HISTORY_PAD_B = 10

    # Delay between engine moves in engine-vs-engine mode (milliseconds)
    ENGINE_MOVE_DELAY_MS = 700

    def __init__(self, game: Game, white_engine: ChessEngine, black_engine: ChessEngine):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Chess")
        self.clock = pygame.time.Clock()
        self.font_large = ui_font(30, bold=True)
        self.font_medium = ui_font(22, bold=True)
        self.font_move = ui_font(16)
        self.font_small = ui_font(16)

        self.game = game
        self.white_engine = white_engine
        self.black_engine = black_engine

        # Networks whose value head can be read. One of them against a human, two
        # of them against each other, none in a human-only game.
        self.net_engines = [
            (colour, eng)
            for colour, eng in (('white', white_engine), ('black', black_engine))
            if hasattr(eng, "evaluate_wdl")
        ]
        # The search readout only makes sense for a single engine. With two of
        # them the numbers would swap sides every ply and read as noise.
        self.think_engine = (
            self.net_engines[0][1] if len(self.net_engines) == 1 else None
        )
        self.human_colour = (
            'white' if isinstance(white_engine, HumanInputEngine)
            else 'black' if isinstance(black_engine, HumanInputEngine)
            else None
        )

        self.selected_square: Optional[tuple[int, int]] = None
        self.legal_moves_from_selected: list[Move] = []
        self.last_move: Optional[Move] = None
        self.running = True
        self.show_menu = False

        self.dragging = False
        self.drag_from_square: Optional[tuple[int, int]] = None

        # First visible row, one row per move pair. Following means the list
        # sticks to the newest move until the user scrolls away from the bottom.
        self.history_scroll = 0
        self.history_follow = True

        # Reviewing an earlier position: the number of plies the board shows,
        # or None while it shows the live game. The game itself keeps running.
        self.review_ply: Optional[int] = None
        self._review_board: Optional[tuple[int, BoardState]] = None
        self._san_cache: list[str] = []

        self.game_over_shown = False
        self.board_flipped = False
        self.last_engine_polled_position: Optional[str] = None

        # In engine-vs-engine mode we space moves out so they're watchable.
        self.bot_vs_bot = (
            not isinstance(white_engine, HumanInputEngine)
            and not isinstance(black_engine, HumanInputEngine)
        )
        self.last_engine_move_time = 0

        self.piece_images = self._load_piece_images()
        self.menu_button_img = self._load_button_image("menu.png", (40, 40))
        self.flip_button_img = self._load_button_image("right-left.png", (40, 40))

    def _load_piece_images(self) -> dict:
        """Load PNG piece images."""
        pieces = {}

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

        size = (self.SQUARE_SIZE - 10, self.SQUARE_SIZE - 10)
        for piece_key, filename in piece_files.items():
            img = load_image(filename, size)
            if img is not None:
                pieces[piece_key] = img

        return pieces

    def _load_button_image(self, filename: str, size: tuple) -> pygame.Surface | None:
        """Load a UI icon, tinted so it reads against the dark panels."""
        return load_icon(filename, size)

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
            # Only let the engines move while the game is actually running.
            # _handle_events may have just cleared self.running, and the status
            # may have become final, in which case asking an engine for another
            # move would keep the game going past its own end.
            if self.running and self.game.get_status() == GameStatus.ONGOING:
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
                # event.y is positive when scrolling up, and history_scroll is
                # the index of the first visible row, so it has to go the other
                # way or the list moves against the wheel.
                self._scroll_history(-event.y)

            elif event.type == pygame.KEYDOWN:
                # Left and right step through the game, up and down scroll the
                # list without moving the board.
                if event.key == pygame.K_ESCAPE:
                    self._request_menu()
                elif event.key == pygame.K_LEFT:
                    self._set_review(self._view_index() - 1)
                elif event.key == pygame.K_RIGHT:
                    self._set_review(self._view_index() + 1)
                elif event.key == pygame.K_HOME:
                    self._set_review(0)
                elif event.key == pygame.K_END:
                    self._set_review(len(self.game.move_history))
                elif event.key == pygame.K_UP:
                    self._scroll_history(-1)
                elif event.key == pygame.K_DOWN:
                    self._scroll_history(1)

    def _is_human_turn(self) -> bool:
        """Return True if the active player is a human."""
        engine = self.white_engine if self.game.board.active_color == 'white' else self.black_engine
        return isinstance(engine, HumanInputEngine)

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

        # A move in the list jumps to the position it produced. Clicking the
        # newest one lands on the live game, which is also how you leave review.
        for rect, ply in self._history_cell_rects():
            if rect.collidepoint(pos):
                self._set_review(ply + 1)
                return

        if self._is_reviewing():
            # Anywhere else returns to the live position, so a game can never be
            # left stranded in review with no obvious way back.
            self._set_review(len(self.game.move_history))
            return

        if not self._is_human_turn():
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
        if self._is_human_turn():
            return

        position_key = self.game.board.to_fen()
        if self.last_engine_polled_position == position_key:
            return

        # In bot-vs-bot mode, leave a small gap between moves so the game is
        # watchable instead of flashing to the end instantly.
        if self.bot_vs_bot:
            now = pygame.time.get_ticks()
            if now - self.last_engine_move_time < self.ENGINE_MOVE_DELAY_MS:
                return

        engine = self.white_engine if self.game.board.active_color == 'white' else self.black_engine
        move = engine.get_best_move(self.game.board, self.game.move_history)
        if move and self.game.make_move(move):
            self.last_move = move
            self.selected_square = None
            self.legal_moves_from_selected = []
            self.last_engine_move_time = pygame.time.get_ticks()

        self.last_engine_polled_position = position_key

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
        self._draw_wdl_bars()
        self._draw_think_table()
        self._draw_move_history()
        self._draw_ui()
        pygame.display.flip()

    def _draw_board(self):
        """Draw chessboard."""
        last_move = self._display_last_move()
        for row in range(8):
            for col in range(8):
                display_row = (7 - row) if self.board_flipped else row
                display_col = (7 - col) if self.board_flipped else col

                x = self.BOARD_START_X + display_col * self.SQUARE_SIZE
                y = self.BOARD_START_Y + display_row * self.SQUARE_SIZE

                is_light = (row + col) % 2 == 0
                color = COLOR_BOARD_LIGHT if is_light else COLOR_BOARD_DARK

                if last_move:
                    if ((row, col) == (last_move.from_row, last_move.from_col) or
                        (row, col) == (last_move.to_row, last_move.to_col)):
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
        board = self._display_board()
        for row in range(8):
            for col in range(8):
                if self.dragging and (row, col) == self.drag_from_square:
                    continue

                piece = board.get_piece(row, col)
                if piece:
                    display_row = (7 - row) if self.board_flipped else row
                    display_col = (7 - col) if self.board_flipped else col

                    x = self.BOARD_START_X + display_col * self.SQUARE_SIZE + 5
                    y = self.BOARD_START_Y + display_row * self.SQUARE_SIZE + 5
                    if piece in self.piece_images:
                        self.screen.blit(self.piece_images[piece], (x, y))

        if self.dragging and self.drag_from_square:
            piece = board.get_piece(self.drag_from_square[0], self.drag_from_square[1])
            if piece and piece in self.piece_images:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                x = mouse_x - self.SQUARE_SIZE // 2
                y = mouse_y - self.SQUARE_SIZE // 2
                self.screen.blit(self.piece_images[piece], (x, y))

    def _wdl_for(self, engine, colour: str) -> tuple[float, float, float]:
        """The engine's win/draw/loss, restated for `colour` rather than for
        whoever happens to be on move."""
        win, draw, loss = engine.evaluate_wdl(self.game.board, self.game.move_history)
        if self.game.board.active_color != colour:
            win, loss = loss, win
        return win, draw, loss

    def _draw_wdl_bars(self):
        """One meter per network, green for a win, grey a draw, red a loss.

        Against a human the single bar is stated for the human, so it reads as
        your own chances. Two networks are both stated for White, so the bars can
        be compared directly and you can see where they disagree.
        """
        if not self.net_engines:
            return

        y = self.PANEL_TOP
        for colour, engine in self.net_engines:
            subject = self.human_colour or 'white'
            win, draw, loss = self._wdl_for(engine, subject)

            if self.human_colour:
                caption = "The net rates your chances"
            else:
                caption = f"{engine.name()} ({colour.capitalize()})"
            label = self.font_small.render(caption, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(label, (self.PANEL_X, y))

            bar_y = y + self.font_small.get_height() + self.WDL_CAPTION_GAP
            x = self.PANEL_X
            # Rounding each segment separately can leave a gap, so the last one
            # takes whatever width remains.
            widths = [int(self.PANEL_W * win), int(self.PANEL_W * draw)]
            widths.append(self.PANEL_W - widths[0] - widths[1])
            for w, colr in zip(widths,
                               (COLOR_WDL_WIN, COLOR_WDL_DRAW, COLOR_WDL_LOSS)):
                if w > 0:
                    pygame.draw.rect(self.screen, colr, (x, bar_y, w, self.WDL_BAR_H))
                x += w
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER,
                             (self.PANEL_X, bar_y, self.PANEL_W, self.WDL_BAR_H), 1)

            pct = (f"{win * 100:.0f}% win   {draw * 100:.0f}% draw   "
                   f"{loss * 100:.0f}% loss")
            if not self.human_colour:
                pct += "  (White)"
            pct_surf = self.font_small.render(pct, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(pct_surf,
                             (self.PANEL_X, bar_y + self.WDL_BAR_H + self.WDL_PCT_GAP))

            y += self._wdl_block_h()

    def _draw_think_table(self):
        """The engine's top moves, read straight off its search tree.

        Sims is how often a move was tried, which is what the choice is actually
        made on. Policy is the network's opinion before any search, its first
        instinct. Value is what the search concluded, positive when the engine
        likes the position. In policy mode there is no tree, so only the priors
        are real and the rest is left blank.
        """
        engine = self.think_engine
        if engine is None:
            return

        top = self.PANEL_TOP
        if self.net_engines:
            top += self._wdl_block_total() + self.PANEL_GAP

        title = self.font_medium.render("Engine", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title, (self.PANEL_X, top))

        cols = (0, 118, 172, 222, self.PANEL_W)   # move, sims, share, policy, value
        heads = ("Move", "Sims", "Share", "Policy", "Value")
        head_y = top + 28
        for i, head in enumerate(heads):
            surf = self.font_small.render(head, True, COLOR_TEXT_SECONDARY)
            x = self.PANEL_X + cols[i]
            if i:                       # numeric columns are right aligned
                x = self.PANEL_X + cols[i] - surf.get_width()
            self.screen.blit(surf, (x, head_y))

        lines = getattr(engine, "last_lines", [])
        if not lines:
            hint = self.font_small.render("no move yet", True, COLOR_TEXT_SECONDARY)
            self.screen.blit(hint, (self.PANEL_X, head_y + self.THINK_ROW_H))
            return

        searched = getattr(engine, "mode", "mcts") == "mcts"
        y = top + self.THINK_HEADER_H
        for line in lines[:self.THINK_ROWS]:
            colour = COLOR_TEXT_PRIMARY if line["chosen"] else COLOR_TEXT_SECONDARY
            cells = [
                line["san"] + (" <" if line["chosen"] else ""),
                str(line["n"]) if searched else "-",
                f"{line['share'] * 100:.0f}%" if searched else "-",
                f"{line['p'] * 100:.1f}%",
                f"{line['q']:+.2f}" if searched else "-",
            ]
            for i, cell in enumerate(cells):
                surf = self.font_move.render(cell, True, colour)
                x = self.PANEL_X + cols[i]
                if i:
                    x = self.PANEL_X + cols[i] - surf.get_width()
                self.screen.blit(surf, (x, y))
            y += self.THINK_ROW_H

    def _wdl_block_h(self) -> int:
        """Caption, bar, percentages. Measured from the font so the caption
        cannot end up sitting on top of the bar when the font changes."""
        line = self.font_small.get_height()
        return (line + self.WDL_CAPTION_GAP + self.WDL_BAR_H
                + self.WDL_PCT_GAP + line + self.WDL_BLOCK_GAP)

    def _wdl_block_total(self) -> int:
        return len(self.net_engines) * self._wdl_block_h()

    def _think_block_h(self) -> int:
        if self.think_engine is None:
            return 0
        return self.THINK_HEADER_H + self.THINK_ROWS * self.THINK_ROW_H

    def _history_top(self) -> int:
        """Whatever is left under the meters and the search readout."""
        y = self.PANEL_TOP
        if self.net_engines:
            y += self._wdl_block_total() + self.PANEL_GAP
        think = self._think_block_h()
        if think:
            y += think + self.PANEL_GAP
        return y

    def _history_height(self) -> int:
        return max(self.HISTORY_HEADER_H + self.HISTORY_ROW_H,
                   self.PANEL_BOTTOM - self._history_top())

    def _history_rows_per_page(self) -> int:
        usable = self._history_height() - self.HISTORY_HEADER_H - self.HISTORY_PAD_B
        return max(1, usable // self.HISTORY_ROW_H)

    def _history_total_rows(self) -> int:
        return (len(self.game.move_history) + 1) // 2

    def _history_max_scroll(self) -> int:
        return max(0, self._history_total_rows() - self._history_rows_per_page())

    def _clamp_history_scroll(self):
        """Keep the scroll in range, and pin it to the bottom while following."""
        max_scroll = self._history_max_scroll()
        if self.history_follow:
            self.history_scroll = max_scroll
        else:
            self.history_scroll = max(0, min(self.history_scroll, max_scroll))

    def _scroll_history(self, rows: int):
        """Move the list by `rows`, positive downwards. Reaching the bottom
        re-arms following, so the list resumes tracking the game."""
        max_scroll = self._history_max_scroll()
        self.history_scroll = max(0, min(self.history_scroll + rows, max_scroll))
        self.history_follow = self.history_scroll >= max_scroll

    def _scroll_to_ply(self, ply: int):
        """Scroll the least amount that brings `ply` into view."""
        if ply < 0:
            return
        row = ply // 2
        page = self._history_rows_per_page()
        if row < self.history_scroll:
            self.history_scroll = row
        elif row >= self.history_scroll + page:
            self.history_scroll = row - page + 1
        self.history_follow = self.history_scroll >= self._history_max_scroll()

    # ---------------------------------------------------------------- notation

    def _san_history(self) -> list[str]:
        """The move history in SAN, named once per move and remembered.

        SAN needs the position a move was played from, which position_history
        already keeps, so each name costs one FEN parse the first time the move
        appears and nothing on the frames after that.
        """
        moves = self.game.move_history
        if len(self._san_cache) > len(moves):
            self._san_cache = []          # a new game in the same window
        while len(self._san_cache) < len(moves):
            self._san_cache.append(self._name_move(len(self._san_cache)))
        return self._san_cache

    def _name_move(self, i: int) -> str:
        move = self.game.move_history[i]
        plain = f"{move.from_square}-{move.to_square}"
        if chess is None or i >= len(self.game.position_history):
            return plain
        try:
            board = chess.Board(self.game.position_history[i])
            return board.san(board.parse_uci(move.to_uci()))
        except (ValueError, AssertionError, IndexError):
            return plain                  # never let notation break the board

    # ----------------------------------------------------------------- review

    def _view_index(self) -> int:
        """How many plies of the game the board is currently showing."""
        if self.review_ply is None:
            return len(self.game.move_history)
        return max(0, min(self.review_ply, len(self.game.move_history)))

    def _is_reviewing(self) -> bool:
        return self.review_ply is not None

    def _set_review(self, index: int):
        """Show the position after `index` plies. The live game is the last one,
        and landing there leaves review mode rather than pinning to a position
        the engines are about to move on from."""
        total = len(self.game.move_history)
        index = max(0, min(index, total))
        self.review_ply = None if index == total else index
        self.selected_square = None
        self.legal_moves_from_selected = []
        self.dragging = False
        self.drag_from_square = None
        if self.review_ply is not None:
            self._scroll_to_ply(index - 1)

    def _display_board(self) -> BoardState:
        """The board to draw: the live one, or a reviewed position rebuilt from
        its FEN and kept until the reviewed ply changes."""
        if not self._is_reviewing():
            return self.game.board
        index = self._view_index()
        if self._review_board is None or self._review_board[0] != index:
            self._review_board = (
                index, BoardState.from_fen(self.game.position_history[index])
            )
        return self._review_board[1]

    def _display_last_move(self) -> Optional[Move]:
        """The move to highlight on the board, i.e. the one that led here."""
        if not self._is_reviewing():
            return self.last_move
        index = self._view_index()
        return self.game.move_history[index - 1] if index else None

    def _history_cell_rects(self) -> list[tuple[pygame.Rect, int]]:
        """(rect, ply) for every move on screen.

        Drawing and hit-testing both read this, so a click can never land on a
        different move than the one under the cursor.
        """
        self._clamp_history_scroll()
        moves = len(self.game.move_history)
        white_x = self.HISTORY_START_X + 62
        black_x = self.HISTORY_START_X + int(self.HISTORY_WIDTH * 0.60)
        right = self.HISTORY_START_X + self.HISTORY_WIDTH - 8
        columns = ((white_x, black_x - 6), (black_x, right))

        y = self._history_top() + self.HISTORY_HEADER_H
        cells = []
        start = self.history_scroll
        end = min(start + self._history_rows_per_page(), self._history_total_rows())
        for row in range(start, end):
            for col, (x, x_end) in enumerate(columns):
                ply = row * 2 + col
                if ply < moves:
                    cells.append(
                        (pygame.Rect(x - 4, y - 2, x_end - x + 4, self.HISTORY_ROW_H),
                         ply)
                    )
            y += self.HISTORY_ROW_H
        return cells

    def _draw_move_history(self):
        """Draw move history panel."""
        top = self._history_top()
        # Panel background
        pygame.draw.rect(
            self.screen,
            COLOR_PANEL,
            (self.HISTORY_START_X, top, self.HISTORY_WIDTH, self._history_height()),
        )
        pygame.draw.rect(
            self.screen,
            COLOR_PANEL_BORDER,
            (self.HISTORY_START_X, top, self.HISTORY_WIDTH, self._history_height()),
            2,
        )

        # Title
        title = self.font_medium.render("Moves", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title, (self.HISTORY_START_X + 10, top + 10))

        # Moves in SAN, pairs side by side, one row per full move. The move that
        # produced the position on the board is boxed, so stepping through the
        # game shows where you are.
        san = self._san_history()
        num_x = self.HISTORY_START_X + 14
        current = self._view_index() - 1

        numbered = set()
        for rect, ply in self._history_cell_rects():
            row = ply // 2
            if row not in numbered:
                numbered.add(row)
                num_surf = self.font_move.render(f"{row + 1}.", True, COLOR_TEXT_SECONDARY)
                self.screen.blit(num_surf, (num_x, rect.y + 2))

            if ply == current:
                pygame.draw.rect(self.screen, COLOR_BUTTON, rect)
                colour = COLOR_TEXT_PRIMARY
            else:
                colour = COLOR_TEXT_PRIMARY if ply % 2 == 0 else COLOR_TEXT_SECONDARY
            self.screen.blit(self.font_move.render(san[ply], True, colour),
                             (rect.x + 4, rect.y + 2))

        # Tell the reader there is more above, since the list follows the game.
        start_row = self.history_scroll
        if start_row > 0:
            more = self.font_small.render(f"{start_row} more above", True, COLOR_TEXT_SECONDARY)
            self.screen.blit(more, (num_x, top + self.HISTORY_HEADER_H - 22))

    def _draw_ui(self):
        """Draw UI elements."""
        status = self.game.get_status()

        status_text = f"Turn: {self.game.board.active_color.upper()}"
        if status != GameStatus.ONGOING:
            status_text = "Game Over"
        if self._is_reviewing():
            status_text = f"Review {self._view_index()}/{len(self.game.move_history)}"

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
