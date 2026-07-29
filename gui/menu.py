"""Menu screen with consistent color scheme and styling."""

import pygame
from dataclasses import dataclass, field
from enum import Enum
import os

from engines import net_catalog

# Consistent color scheme
COLOR_BG = (45, 45, 45)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (100, 160, 210)
COLOR_BUTTON_ACTIVE = (120, 180, 240)
COLOR_BUTTON_DISABLED = (60, 70, 78)
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_SECONDARY = (200, 200, 200)
COLOR_TEXT_DISABLED = (130, 135, 140)
COLOR_PANEL = (60, 60, 60)
COLOR_PANEL_BORDER = (120, 120, 120)
COLOR_INPUT_BG = (35, 35, 35)
COLOR_INPUT_FOCUS = (120, 180, 240)

# Row geometry, shared by rendering and hit testing so the two cannot drift.
ROW_X, ROW_W, ROW_H = 100, 600, 60
ROW_TOP, ROW_STEP = 150, 70

BACK_RECT = (30, 30, 50, 50)


class GameMode(Enum):
    HUMAN_VS_HUMAN = "human_vs_human"
    HUMAN_VS_ENGINE = "human_vs_engine"
    ENGINE_VS_ENGINE = "engine_vs_engine"


@dataclass
class SearchSettings:
    """How a network should pick its move."""
    mode: str = "mcts"      # "policy" or "mcts"
    sims: int = 200         # ignored when mode is "policy"


@dataclass
class GameConfig:
    """Game configuration from menu."""
    mode: GameMode
    white_engine_name: str | None = None
    black_engine_name: str | None = None
    white_search: SearchSettings = field(default_factory=SearchSettings)
    black_search: SearchSettings = field(default_factory=SearchSettings)


def _row_rect(i: int) -> tuple[int, int, int, int]:
    return (ROW_X, ROW_TOP + i * ROW_STEP, ROW_W, ROW_H)


def _in_rect(pos: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y, w, h = rect
    return x <= pos[0] <= x + w and y <= pos[1] <= y + h


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

        # The five trained networks. Ones whose weights are not in Weights/ are
        # still listed, greyed out, so it is obvious what is missing rather than
        # the menu silently being short.
        self.engine_names = list(net_catalog.NETS.keys())
        self.selected_engines = {"opponent": None, "white": None, "black": None}
        self.search = {
            "opponent": SearchSettings(),
            "white": SearchSettings(),
            "black": SearchSettings(),
        }
        self.hovered_engine = None
        self.sims_text = "200"
        self.sims_focused = False

        self.back_button_img = self._load_button_image("back-button.png", (50, 50))

    def _load_button_image(self, filename: str, size: tuple) -> pygame.Surface | None:
        """Load button image."""
        path = os.path.join(os.path.dirname(__file__), "..", "pieces-basic-png", filename)
        try:
            if os.path.exists(path):
                img = pygame.image.load(path)
                return pygame.transform.scale(img, size)
        except Exception:
            pass
        return None

    def run(self) -> GameConfig | None:
        """Run menu loop."""
        while self.running and not self.config:
            self._handle_events()
            self._render()
            self.clock.tick(30)

        return self.config

    # ------------------------------------------------------------------ input

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
                if self.current_screen == "search_selection" and self.sims_focused:
                    if self._handle_sims_key(event):
                        continue
                if event.key == pygame.K_ESCAPE:
                    if self.current_screen != "mode_selection":
                        self._go_back()
                    else:
                        self.running = False

    def _handle_sims_key(self, event) -> bool:
        """Digits, backspace and Enter for the simulation count. True if consumed."""
        if event.key == pygame.K_BACKSPACE:
            self.sims_text = self.sims_text[:-1]
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.sims_focused = False
            return True
        if event.unicode.isdigit() and len(self.sims_text) < 5:
            self.sims_text += event.unicode
            return True
        return False

    def _update_hover(self, pos: tuple[int, int]):
        """Update hovered row."""
        self.hovered_engine = None
        if self.current_screen == "engine_selection":
            for i in range(len(self.engine_names)):
                if _in_rect(pos, _row_rect(i)):
                    self.hovered_engine = i
        elif self.current_screen == "color_selection":
            for i in range(3):
                if _in_rect(pos, _row_rect(i)):
                    self.hovered_engine = i

    def _handle_click(self, pos: tuple[int, int]):
        """Handle mouse click."""
        if self.current_screen == "mode_selection":
            self._handle_mode_selection_click(pos)
            return

        if _in_rect(pos, BACK_RECT):
            self._go_back()
            return

        if self.current_screen == "engine_selection":
            self._handle_engine_selection_click(pos)
        elif self.current_screen == "search_selection":
            self._handle_search_selection_click(pos)
        elif self.current_screen == "color_selection":
            self._handle_color_selection_click(pos)

    def _go_back(self):
        if self.current_screen == "search_selection":
            self.current_screen = "engine_selection"
        elif self.current_screen == "color_selection":
            self.current_screen = "search_selection"
        else:
            self.current_screen = "mode_selection"

    def _handle_mode_selection_click(self, pos: tuple[int, int]):
        """Handle mode selection clicks."""
        if _in_rect(pos, (100, 140, 600, 80)):
            self.config = GameConfig(
                mode=GameMode.HUMAN_VS_HUMAN,
                white_engine_name="human",
                black_engine_name="human",
            )
            self.running = False
            return

        if _in_rect(pos, (100, 270, 600, 80)):
            self.selected_mode = GameMode.HUMAN_VS_ENGINE
            self.current_screen = "engine_selection"
            self.engine_selection = "opponent"
            return

        if _in_rect(pos, (100, 400, 600, 80)):
            self.selected_mode = GameMode.ENGINE_VS_ENGINE
            self.current_screen = "engine_selection"
            self.engine_selection = "white"
            return

    def _handle_engine_selection_click(self, pos: tuple[int, int]):
        """Handle engine selection clicks. Unavailable nets are inert."""
        for i, name in enumerate(self.engine_names):
            if not _in_rect(pos, _row_rect(i)):
                continue
            if not net_catalog.is_available(name):
                return
            self.selected_engines[self.engine_selection] = name
            if self.engine_selection == "opponent":
                self.selected_opponent_engine = name
            current = self.search[self.engine_selection]
            self.sims_text = str(current.sims)
            self.current_screen = "search_selection"
            return

    def _handle_search_selection_click(self, pos: tuple[int, int]):
        """Policy / MCTS choice, the simulation field and Continue."""
        slot = self.engine_selection
        settings = self.search[slot]

        if _in_rect(pos, (100, 140, 600, 70)):
            settings.mode = "policy"
            self.sims_focused = False
            return

        if _in_rect(pos, (100, 225, 600, 70)):
            settings.mode = "mcts"
            return

        if _in_rect(pos, (390, 320, 150, 46)):
            self.sims_focused = settings.mode == "mcts"
            return

        if _in_rect(pos, (100, 460, 600, 70)):
            self.sims_focused = False
            settings.sims = self._parsed_sims()
            self.sims_text = str(settings.sims)
            self._advance_from_search()
            return

        self.sims_focused = False

    def _parsed_sims(self) -> int:
        try:
            return max(1, min(100000, int(self.sims_text)))
        except ValueError:
            return 200

    def _advance_from_search(self):
        """Where to go once a net's search settings are set."""
        if self.engine_selection == "opponent":
            self.current_screen = "color_selection"
        elif self.engine_selection == "white":
            self.engine_selection = "black"
            self.current_screen = "engine_selection"
        else:
            self.config = GameConfig(
                mode=GameMode.ENGINE_VS_ENGINE,
                white_engine_name=self.selected_engines["white"],
                black_engine_name=self.selected_engines["black"],
                white_search=self.search["white"],
                black_search=self.search["black"],
            )
            self.running = False

    def _handle_color_selection_click(self, pos: tuple[int, int]):
        """Handle color selection clicks."""
        colors = ["white", "black", "random"]
        for i, color in enumerate(colors):
            if not _in_rect(pos, _row_rect(i)):
                continue

            opponent = self.selected_opponent_engine
            settings = self.search["opponent"]
            if color == "random":
                import random
                color = "white" if random.random() < 0.5 else "black"

            if color == "white":
                self.config = GameConfig(
                    mode=GameMode.HUMAN_VS_ENGINE,
                    white_engine_name="human",
                    black_engine_name=opponent,
                    black_search=settings,
                )
            else:
                self.config = GameConfig(
                    mode=GameMode.HUMAN_VS_ENGINE,
                    white_engine_name=opponent,
                    black_engine_name="human",
                    white_search=settings,
                )
            self.running = False
            return

    def _get_engine_display_name(self, engine_id: str | None) -> str:
        """Get engine display name."""
        if engine_id and engine_id in net_catalog.NETS:
            return str(net_catalog.NETS[engine_id]["label"])
        return "Human"

    # --------------------------------------------------------------- rendering

    def _render(self):
        """Render menu."""
        self.screen.fill(COLOR_BG)

        if self.current_screen == "mode_selection":
            self._render_mode_selection()
        elif self.current_screen == "engine_selection":
            self._render_engine_selection()
        elif self.current_screen == "search_selection":
            self._render_search_selection()
        elif self.current_screen == "color_selection":
            self._render_color_selection()

        pygame.display.flip()

    def _title(self, text: str):
        surf = self.font_large.render(text, True, COLOR_TEXT_PRIMARY)
        self.screen.blit(surf, (self.WINDOW_WIDTH // 2 - surf.get_width() // 2, 30))

    def _back_button(self):
        pygame.draw.rect(self.screen, COLOR_PANEL, BACK_RECT)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, BACK_RECT, 2)
        if self.back_button_img:
            self.screen.blit(self.back_button_img, (30, 30))
        else:
            text = self.font_small.render("<-", True, COLOR_TEXT_PRIMARY)
            self.screen.blit(text, (42, 38))

    def _button(self, rect, label, *, active=False, hovered=False, enabled=True,
                sub=None, font=None):
        if not enabled:
            color = COLOR_BUTTON_DISABLED
        elif active:
            color = COLOR_BUTTON_ACTIVE
        elif hovered:
            color = COLOR_BUTTON_HOVER
        else:
            color = COLOR_BUTTON
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, rect, 2)

        fg = COLOR_TEXT_PRIMARY if enabled else COLOR_TEXT_DISABLED
        font = font or self.font_small
        text = font.render(label, True, fg)
        y = rect[1] + (18 if sub else (rect[3] - text.get_height()) // 2)
        self.screen.blit(text, (rect[0] + 30, y))
        if sub:
            sub_fg = COLOR_TEXT_SECONDARY if enabled else COLOR_TEXT_DISABLED
            sub_surf = self.font_small.render(sub, True, sub_fg)
            self.screen.blit(sub_surf, (rect[0] + 30, rect[1] + 40))

    def _render_mode_selection(self):
        """Render mode selection screen."""
        self._title("Chess")
        self._button((100, 140, 600, 80), "Human vs Human", font=self.font_medium)
        self._button((100, 270, 600, 80), "Human vs Engine", font=self.font_medium)
        self._button((100, 400, 600, 80), "Engine vs Engine", font=self.font_medium)

    def _render_engine_selection(self):
        """Render engine selection screen."""
        if self.engine_selection == "opponent":
            self._title("Select Opponent")
        elif self.engine_selection == "white":
            self._title("Select White Engine")
        else:
            self._title("Select Black Engine")

        self._back_button()

        for i, name in enumerate(self.engine_names):
            enabled = net_catalog.is_available(name)
            self._button(
                _row_rect(i),
                net_catalog.menu_label(name),
                active=self.selected_engines[self.engine_selection] == name,
                hovered=self.hovered_engine == i and enabled,
                enabled=enabled,
                sub=None if enabled else "weights not found in Weights/",
            )

        if not net_catalog.available():
            hint = "No weights found. Download them from Kaggle into Weights/ - see Weights/README.md"
            surf = self.font_small.render(hint, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(surf, (self.WINDOW_WIDTH // 2 - surf.get_width() // 2, 530))

    def _render_search_selection(self):
        """Policy vs MCTS, plus the simulation count."""
        name = self.selected_engines[self.engine_selection]
        self._title(self._get_engine_display_name(name))
        self._back_button()

        settings = self.search[self.engine_selection]

        self._button(
            (100, 140, 600, 70), "Policy only",
            active=settings.mode == "policy",
            sub="one look at the board, answers instantly, clearly weaker",
        )
        self._button(
            (100, 225, 600, 70), "Search (MCTS)",
            active=settings.mode == "mcts",
            sub="searches before moving, how the report measured Elo",
        )

        enabled = settings.mode == "mcts"
        label_fg = COLOR_TEXT_PRIMARY if enabled else COLOR_TEXT_DISABLED
        label = self.font_small.render("Simulations per move:", True, label_fg)
        self.screen.blit(label, (130, 333))

        box = (390, 320, 150, 46)
        pygame.draw.rect(self.screen, COLOR_INPUT_BG, box)
        border = COLOR_INPUT_FOCUS if (self.sims_focused and enabled) else COLOR_PANEL_BORDER
        pygame.draw.rect(self.screen, border, box, 2)
        value = self.font_medium.render(self.sims_text or "", True, label_fg)
        self.screen.blit(value, (box[0] + 12, box[1] + 8))

        if enabled:
            note = "1000 is the reported setting, about 3.5 s per move on CPU. 200 answers in under a second."
        else:
            note = "Simulation count does not apply to the policy-only mode."
        surf = self.font_small.render(note, True, COLOR_TEXT_SECONDARY)
        self.screen.blit(surf, (self.WINDOW_WIDTH // 2 - surf.get_width() // 2, 390))

        self._button((100, 460, 600, 70), "Continue", font=self.font_medium)

    def _render_color_selection(self):
        """Render color selection screen."""
        opponent_name = self._get_engine_display_name(self.selected_opponent_engine)
        self._title(f"Play as... ({opponent_name})")
        self._back_button()

        for i, label in enumerate(("White", "Black", "Random")):
            self._button(_row_rect(i), label, hovered=self.hovered_engine == i)
