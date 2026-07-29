"""The trained networks you can play against, and where their weights live.

Deliberately free of tensorflow so the menu can list and check nets without
paying the twenty seconds it costs to import the deep learning stack.

Weights are not in the repository, they are about 100 MB each. Download them
from the Kaggle dataset and drop the .weights.h5 files into Weights/ at the
repo root. See Weights/README.md.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(REPO_ROOT, "Weights")

# Ordered weakest to strongest, which is also the order they were trained in.
# Elo figures are the ones measured in the report, against Stockfish at a fixed
# 0.75 seconds per move.
NETS = {
    "Supervised_v1": {
        "label": "Supervised v1",
        "elo": None,
        "about": "first supervised net, trained on game results only",
    },
    "Supervised_v2": {
        "label": "Supervised v2",
        "elo": None,
        "about": "supervised, value head still had a systematic flaw",
    },
    "Supervised_v3": {
        "label": "Supervised v3",
        "elo": None,
        "about": "supervised, flaw fixed, roughly 100 Elo below v4",
    },
    "Supervised_v4": {
        "label": "Supervised v4",
        "elo": 1920,
        "about": "the strongest net here, about 1920 Elo",
    },
    "RL": {
        "label": "RL (self-play)",
        "elo": 1330,
        "about": "learned purely from self-play, about 1330 Elo",
    },
}


def weights_path(name: str) -> str:
    """Absolute path the weights file for `name` is expected at."""
    return os.path.join(WEIGHTS_DIR, f"{name}.weights.h5")


def is_available(name: str) -> bool:
    return os.path.isfile(weights_path(name))


def available() -> list[str]:
    """Names whose weights are actually present, in catalogue order."""
    return [n for n in NETS if is_available(n)]


def missing() -> list[str]:
    return [n for n in NETS if not is_available(n)]


def menu_label(name: str) -> str:
    """Display string for the menu, e.g. 'Supervised v4  ~1920 Elo'."""
    entry = NETS[name]
    if entry["elo"]:
        return f"{entry['label']}   ~{entry['elo']} Elo"
    return str(entry["label"])
