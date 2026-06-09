import os
import re


def extract_iteration(file_name: str) -> int:
    """Extract iteration number from a weights filename.

    Handles inputs like:
    - small_network_v0.weights.h5
    - /full/path/to/small_network_v10.weights.h5

    Returns -1 when no iteration is found.
    """
    base = os.path.basename(file_name)
    m = re.search(r"v(\d+)", base)
    if m:
        return int(m.group(1))
    return -1


def get_latest_weights(network: str, weights_dir: str) -> str:
    """Return full path to the latest weights file for `network` in `weights_dir`.

    If no file matches the prefix "{network}_network", falls back to any
    '*.weights.h5' file in the directory.
    """
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    all_files = [f for f in os.listdir(weights_dir) if f.endswith(".weights.h5")]
    if not all_files:
        raise FileNotFoundError(f"No weights found in {weights_dir}.")

    prefix = f"{network}_network"
    candidates = [f for f in all_files if f.startswith(prefix)] or all_files

    latest_file = max(candidates, key=extract_iteration)
    return os.path.join(weights_dir, latest_file)
