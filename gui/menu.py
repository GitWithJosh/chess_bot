"""Menu screen with consistent color scheme and styling."""

import pygame
from dataclasses import dataclass
from enum import Enum
import os

# Consistent color scheme
COLOR_BG = (45, 45, 45)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (100, 160, 210)
COLOR_BUTTON_ACTIVE = (120, 180, 240)
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_SECONDARY = (200, 200, 200)
COLOR_PANEL = (60, 60, 60)
COLOR_PANEL_BORDER = (120, 120, 120)


class GameMode(Enum):
    HUMAN_VS_HUMAN = "human_vs_human"
    HUMAN_VS_ENGINE = "human_vs_engine"
    ENGINE_VS_ENGINE = "engine_vs_engine"


@dataclass
class GameConfig:
    """Game configuration from menu."""
    mode: GameMode
    white_engine_name: str | None = None
    black_engine_name: str | None = None


class MenuScreen:
    """Menu screen with professional styling."""

    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Chess - Game Setup")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 56)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.config = None
        self.running = True
        self.current_screen = "mode_selection"
        self.selected_mode = None
        self.engine_selection = "opponent"
        self.selected_opponent_engine = None
        self.engines = {
            "Random": "random",
            "Stockfish (Easy)": "stockfish_800",
            "Stockfish (Medium)": "stockfish_1600",
            "Stockfish (Hard)": "stockfish_2400",
        }
        self.selected_engines = {"opponent": "random", "white": "random", "black": "random"}
        self.hovered_engine = None

        self.back_button_img = self._load_button_image("back-button.png", (50, 50))

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

    def run(self) -> GameConfig | None:
        """Run menu loop."""
        while self.running and not self.config:
            self._handle_events()
            self._render()
            self.clock.tick(30)

        pygame.quit()
        return self.config

    def _handle_events(self):
        """Handle menu input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(event.pos)

            elif event.type == pygame.MOUSEMOTION:
                self._update_hover(event.pos)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.current_screen != "mode_selection":
                        self.current_screen = "mode_selection"
                    else:
                        self.running = False

    def _update_hover(self, pos: tuple[int, int]):
        """Update hovered engine."""
        self.hovered_engine = None
        if self.current_screen == "engine_selection":
            engine_names = list(self.engines.keys())
            for i, name in enumerate(engine_names):
                y = 150 + i * 70
                if 100 <= pos[0] <= 700 and y <= pos[1] <= y + 60:
                    self.hovered_engine = i
        elif self.current_screen == "color_selection":
            colors = [("White", "white"), ("Black", "black"), ("Random", "random")]
            for i, (name, _) in enumerate(colors):
                y = 150 + i * 70
                if 100 <= pos[0] <= 700 and y <= pos[1] <= y + 60:
                    self.hovered_engine = i

    def _handle_click(self, pos: tuple[int, int]):
        """Handle mouse click."""
        if self.current_screen == "mode_selection":
            self._handle_mode_selection_click(pos)
        elif self.current_screen == "engine_selection":
            self._handle_engine_selection_click(pos)
        elif self.current_screen == "color_selection":
            self._handle_color_selection_click(pos)

    def _handle_mode_selection_click(self, pos: tuple[int, int]):
        """Handle mode selection clicks."""
        # Human vs Human (100, 140, 600, 80)
        if 100 <= pos[0] <= 700 and 140 <= pos[1] <= 220:
            self.config = GameConfig(mode=GameMode.HUMAN_VS_HUMAN)
            self.running = False
            return

        # Human vs Engine (100, 270, 600, 80)
        if 100 <= pos[0] <= 700 and 270 <= pos[1] <= 350:
            self.selected_mode = GameMode.HUMAN_VS_ENGINE
            self.current_screen = "engine_selection"
            self.engine_selection = "opponent"
            return

        # Engine vs Engine (100, 400, 600, 80)
        if 100 <= pos[0] <= 700 and 400 <= pos[1] <= 480:
            self.selected_mode = GameMode.ENGINE_VS_ENGINE
            self.current_screen = "engine_selection"
            self.engine_selection = "white"
            return

    def _handle_engine_selection_click(self, pos: tuple[int, int]):
        """Handle engine selection clicks."""
        # Back button (30, 30, 50, 50)
        if 30 <= pos[0] <= 80 and 30 <= pos[1] <= 80:
            self.current_screen = "mode_selection"
            return

        # Engine options
        engine_names = list(self.engines.keys())
        for i, name in enumerate(engine_names):
            y = 150 + i * 70
            if 100 <= pos[0] <= 700 and y <= pos[1] <= y + 60:
                self.selected_engines[self.engine_selection] = self.engines[name]

                if self.engine_selection == "opponent":
                    # Move to color selection screen
                    self.selected_opponent_engine = name
                    self.current_screen = "color_selection"
                elif self.engine_selection == "white":
                    self.engine_selection = "black"
                else:
                    white_name = self._get_engine_display_name(self.selected_engines["white"])
                    black_name = self._get_engine_display_name(self.selected_engines["black"])
                    self.config = GameConfig(
                        mode=GameMode.ENGINE_VS_ENGINE,
                        white_engine_name=white_name,
                        black_engine_name=black_name,
                    )
                    self.running = False
                return

    def _handle_color_selection_click(self, pos: tuple[int, int]):
        """Handle color selection clicks."""
        # Back button (30, 30, 50, 50)
        if 30 <= pos[0] <= 80 and 30 <= pos[1] <= 80:
            self.current_screen = "engine_selection"
            return

        # Color options: White, Black, Random
        colors = [("White", "white"), ("Black", "black"), ("Random", "random")]
        for i, (name, color) in enumerate(colors):
            y = 150 + i * 70
            if 100 <= pos[0] <= 700 and y <= pos[1] <= y + 60:
                opponent_name = self.selected_opponent_engine
                if color == "white":
                    self.config = GameConfig(
                        mode=GameMode.HUMAN_VS_ENGINE,
                        white_engine_name="human",
                        black_engine_name=opponent_name,
                    )
                elif color == "black":
                    self.config = GameConfig(
                        mode=GameMode.HUMAN_VS_ENGINE,
                        white_engine_name=opponent_name,
                        black_engine_name="human",
                    )
                else:  # Random
                    import random
                    if random.random() < 0.5:
                        self.config = GameConfig(
                            mode=GameMode.HUMAN_VS_ENGINE,
                            white_engine_name="human",
                            black_engine_name=opponent_name,
                        )
                    else:
                        self.config = GameConfig(
                            mode=GameMode.HUMAN_VS_ENGINE,
                            white_engine_name=opponent_name,
                            black_engine_name="human",
                        )
                self.running = False
                return

    def _get_engine_display_name(self, engine_id: str) -> str:
        """Get engine display name."""
        for name, eid in self.engines.items():
            if eid == engine_id:
                return name
        return "Random"

    def _render(self):
        """Render menu."""
        self.screen.fill(COLOR_BG)

        if self.current_screen == "mode_selection":
            self._render_mode_selection()
        elif self.current_screen == "engine_selection":
            self._render_engine_selection()
        elif self.current_screen == "color_selection":
            self._render_color_selection()

        pygame.display.flip()

    def _render_mode_selection(self):
        """Render mode selection screen."""
        title = self.font_large.render("Chess", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title, (self.WINDOW_WIDTH // 2 - title.get_width() // 2, 30))

        # Human vs Human
        pygame.draw.rect(self.screen, COLOR_BUTTON, (100, 140, 600, 80))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (100, 140, 600, 80), 2)
        text = self.font_medium.render("Human vs Human", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(text, (160, 165))

        # Human vs Engine
        pygame.draw.rect(self.screen, COLOR_BUTTON, (100, 270, 600, 80))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (100, 270, 600, 80), 2)
        text = self.font_medium.render("Human vs Engine", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(text, (170, 295))

        # Engine vs Engine
        pygame.draw.rect(self.screen, COLOR_BUTTON, (100, 400, 600, 80))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (100, 400, 600, 80), 2)
        text = self.font_medium.render("Engine vs Engine", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(text, (170, 425))

    def _render_engine_selection(self):
        """Render engine selection screen."""
        if self.engine_selection == "opponent":
            title_text = "Select Opponent"
        elif self.engine_selection == "white":
            title_text = "Select White Engine"
        else:
            title_text = "Select Black Engine"

        title = self.font_large.render(title_text, True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title, (self.WINDOW_WIDTH // 2 - title.get_width() // 2, 30))

        # Back button
        pygame.draw.rect(self.screen, COLOR_PANEL, (30, 30, 50, 50))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (30, 30, 50, 50), 2)
        if self.back_button_img:
            self.screen.blit(self.back_button_img, (30, 30))
        else:
            text = self.font_small.render("←", True, COLOR_TEXT_PRIMARY)
            self.screen.blit(text, (42, 38))

        # Engine options
        engine_names = list(self.engines.keys())
        for i, (name, eid) in enumerate(self.engines.items()):
            y = 150 + i * 70
            is_selected = self.selected_engines[self.engine_selection] == eid
            is_hovered = self.hovered_engine == i

            if is_selected:
                color = COLOR_BUTTON_ACTIVE
            elif is_hovered:
                color = COLOR_BUTTON_HOVER
            else:
                color = COLOR_BUTTON

            pygame.draw.rect(self.screen, color, (100, y, 600, 60))
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (100, y, 600, 60), 2)
            text = self.font_small.render(name, True, COLOR_TEXT_PRIMARY)
            self.screen.blit(text, (130, y + 18))

    def _render_color_selection(self):
        """Render color selection screen."""
        opponent_name = self.selected_opponent_engine or "Unknown"
        title_text = f"Play as... ({opponent_name})"
        title = self.font_large.render(title_text, True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title, (self.WINDOW_WIDTH // 2 - title.get_width() // 2, 30))

        # Back button
        pygame.draw.rect(self.screen, COLOR_PANEL, (30, 30, 50, 50))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (30, 30, 50, 50), 2)
        if self.back_button_img:
            self.screen.blit(self.back_button_img, (30, 30))
        else:
            text = self.font_small.render("←", True, COLOR_TEXT_PRIMARY)
            self.screen.blit(text, (42, 38))

        # Color options
        colors = [("White", "white"), ("Black", "black"), ("Random", "random")]
        for i, (name, color_id) in enumerate(colors):
            y = 150 + i * 70
            is_hovered = self.hovered_engine == i

            if is_hovered:
                color = COLOR_BUTTON_HOVER
            else:
                color = COLOR_BUTTON

            pygame.draw.rect(self.screen, color, (100, y, 600, 60))
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (100, y, 600, 60), 2)
            text = self.font_small.render(name, True, COLOR_TEXT_PRIMARY)
            self.screen.blit(text, (130, y + 18))
