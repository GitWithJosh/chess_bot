import sys
from pathlib import Path

# Ensure the repository root is on sys.path so tests can import the package
# when running pytest from this subdirectory.
ROOT = Path(__file__).resolve().parents[2]
RL_DIR = ROOT / "reinforcement_learning"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RL_DIR))