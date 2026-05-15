"""Chess application entry point with menu system."""

from gui.menu import MenuScreen, GameMode, GameConfig
from gui.gui import ChessGUI
from game.game import Game
from engines.chess_engine import ChessEngine
from engines.human_engine import HumanInputEngine
from engines.random_engine import RandomEngine
from engines.stockfish_engine import StockfishEngine


def create_engine(engine_name: str) -> ChessEngine:
    """Create engine from name."""
    if engine_name == "human":
        return HumanInputEngine()
    elif engine_name == "random":
        return RandomEngine()
    elif "Stockfish" in engine_name:
        elo = None
        if "(Easy)" in engine_name:
            elo = 800
        elif "(Medium)" in engine_name:
            elo = 1600
        elif "(Hard)" in engine_name:
            elo = 2400
        try:
            return StockfishEngine(depth=10, elo=elo)
        except RuntimeError as e:
            print(f"Warning: {e}")
            return RandomEngine()
    else:
        return RandomEngine()


def main():
    """Run chess application with menu system."""
    last_config = None

    while True:
        # Show menu only if not replaying
        if last_config is None:
            menu = MenuScreen()
            config = menu.run()
            if not config:
                break
        else:
            config = last_config
            last_config = None

        # Create engines based on config
        if config.mode == GameMode.HUMAN_VS_HUMAN:
            white_engine = HumanInputEngine()
            black_engine = HumanInputEngine()
        elif config.mode == GameMode.HUMAN_VS_ENGINE:
            white_engine = create_engine(config.white_engine_name)
            black_engine = create_engine(config.black_engine_name)
        else:  # ENGINE_VS_ENGINE
            white_engine = create_engine(config.white_engine_name)
            black_engine = create_engine(config.black_engine_name)

        # Run game
        game = Game()
        gui = ChessGUI(game, white_engine, black_engine)

        # Initialize board orientation before the first frame so that in
        # Human-vs-Engine mode the human always sees their own side at the
        # bottom, even if the engine plays White and moves first.
        if config.mode == GameMode.HUMAN_VS_ENGINE and hasattr(gui, "board_flipped"):
            gui.board_flipped = (config.black_engine_name == "human")

        # If user returns False from gui.run(), they want to replay
        show_menu = gui.run()

        if not show_menu:
            # Replay with same configuration
            last_config = config
        # If show_menu is True, loop continues and menu is shown


if __name__ == '__main__':
    main()
