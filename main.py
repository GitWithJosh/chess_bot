"""Chess application entry point with menu system.

Local, CPU-only counterpart to the public Kaggle notebook. Play against any of
the five trained networks, or watch two of them play each other.

The weights are not in the repository, they are about 100 MB each. Download
them from the Kaggle dataset and put the .weights.h5 files in Weights/ at the
repo root. See Weights/README.md.
"""

from engines import net_catalog
from gui.menu import MenuScreen, GameMode, GameConfig, SearchSettings
from gui.gui import ChessGUI
from game.game import Game
from engines.engine import ChessEngine
from engines.human_engine import HumanInputEngine


def create_engine(engine_name: str | None, search: SearchSettings) -> ChessEngine:
    """Build the engine for one side. `None` or "human" means a human player."""
    if engine_name is None or engine_name == "human":
        return HumanInputEngine()

    if engine_name not in net_catalog.NETS:
        raise ValueError(f"Unknown engine {engine_name!r}")

    # Imported here rather than at module level so the menu opens immediately.
    # Pulling in tensorflow costs about twenty seconds and is only needed once a
    # network has actually been chosen.
    from engines.big_network_engine import BigNetworkEngine

    path = net_catalog.weights_path(engine_name)
    label = str(net_catalog.NETS[engine_name]["label"])
    print(f"Loading {label} from {path} ...")
    return BigNetworkEngine(
        path,
        mode=search.mode,
        sims=search.sims,
        display_name=label,
    )


def _report_weights():
    """Say up front which networks are playable, so an empty Weights/ is obvious."""
    found = net_catalog.available()
    if found:
        print(f"Weights found in {net_catalog.WEIGHTS_DIR}")
        for name in found:
            print(f"  {net_catalog.NETS[name]['label']:<18} {net_catalog.NETS[name]['about']}")
        gone = net_catalog.missing()
        if gone:
            print(f"  not present: {', '.join(gone)}")
    else:
        print(f"No network weights found in {net_catalog.WEIGHTS_DIR}")
        print("Download them from the Kaggle dataset and drop the .weights.h5 files")
        print("in there. See Weights/README.md. Human vs Human still works.")
    print()


def main():
    """Run chess application with menu system."""
    _report_weights()
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

        white_engine = create_engine(config.white_engine_name, config.white_search)
        black_engine = create_engine(config.black_engine_name, config.black_search)

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
