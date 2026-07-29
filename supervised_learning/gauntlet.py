"""
Gauntlet: the current best supervised net vs *every* training checkpoint of its
run, raw policy, one match each — plotted as an Elo-advantage curve.

This is the "did later training help or hurt?" picture. `16_sl_best` is the net
we kept; here it plays a full match against each `sl_XXX` snapshot of the run
that produced it. A checkpoint that scores well against sl_best (pulling the
line *down*) is one sl_best isn't clearly better than.

Move selection is **raw** (argmax of the legal-masked policy, no search) — the
most sensitive probe of what each net actually learned. The match machinery
(random openings, colour-reversed pairs, PGN output, scoring) is imported
straight from compare_nets.py so this stays a thin driver over the validated
path; see that file and [[net-comparison-eval-workflow]].

Each match is 50 games: OPENINGS random 3-ply openings, each played twice with
colours swapped (25 * 2 = 50). The y-axis is sl_best's Elo advantage from the
match score, elo = -400*log10(1/score - 1), capped +/-ELO_CAP.

Runs are resumable: results are appended to a JSON as each checkpoint finishes,
and re-running skips checkpoints already in it. Usage:

    python supervised_learning/gauntlet.py                 # run all, skip done
    python supervised_learning/gauntlet.py --force         # replay everything
    python supervised_learning/gauntlet.py --only 040 120  # just these labels
    python supervised_learning/gauntlet.py --plot-only     # rebuild plot from JSON

Edit the CONFIG block below to change counts / paths.
"""

import argparse
import json
import math
import os
import re
import sys

# --- path setup (mirror compare_nets) so `import compare_nets` resolves -------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ============================ CONFIG — edit, then run ============================
BEST = os.path.join(REPO_ROOT, "gen_4.weights.h5")   # the net we kept
CKPT_DIR = os.path.join(REPO_ROOT, "supervised_learning/16_checkpoint_export/")                                      # where sl_XXX live
CKPT_GLOB = "sl_*.weights.h5"                              # training snapshots
MODE = "raw"          # raw policy argmax (matches the reference graph)
OPENINGS = 25         # 25 openings x 2 colours = 50 games per checkpoint
OPENING_PLIES = 3     # random plies used to seed each opening
SIMS = 150            # only used if MODE == "mcts"
BATCH_SIZE = 16
SEED = 7              # shared across checkpoints -> everyone faces the same openings
ELO_CAP = 800.0       # clamp the score->Elo curve so a clean sweep isn't +inf
SAVE_PGN = True       # keep each match's PGN (input to the per-phase analysis)
OUT_DIR = os.path.join(REPO_ROOT, "supervised_learning", "results")
RESULTS_JSON = os.path.join(OUT_DIR, "gauntlet_results.json")
PLOT_PNG = os.path.join(OUT_DIR, "gauntlet_plot.png")
PGN_DIR = os.path.join(OUT_DIR, "gauntlet_pgns")
# ================================================================================

# compare_nets does the path/chdir setup at import and pulls in TF; reuse its
# validated, already-tested match code rather than duplicating it here.
from compare_nets import load_net, run_match, save_pgn  # noqa: E402
from helpers.converter import Converter  # noqa: E402


def checkpoint_label(path: str) -> str:
    """`.../sl_040.weights.h5` -> `040`, `.../sl_final.weights.h5` -> `final`."""
    base = os.path.basename(path)
    base = re.sub(r"\.weights\.h5$", "", base)
    return re.sub(r"^sl_", "", base)


def label_sort_key(label: str):
    """Numeric checkpoints in order; non-numeric (e.g. `final`) sort to the end."""
    return (0, int(label)) if label.isdigit() else (1, label)


def find_checkpoints() -> list[str]:
    import glob
    paths = glob.glob(os.path.join(CKPT_DIR, CKPT_GLOB))
    # exclude the reference net itself if the glob ever reaches it
    best_base = os.path.basename(BEST)
    paths = [p for p in paths if os.path.basename(p) != best_base]
    return sorted(paths, key=lambda p: label_sort_key(checkpoint_label(p)))


def elo_from_score(score: float, cap: float) -> float:
    """sl_best's Elo advantage from its match score, clamped to +/-cap."""
    if score <= 0.0:
        return -cap
    if score >= 1.0:
        return cap
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    return max(-cap, min(cap, elo))


def load_results() -> dict:
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def match_record(scores: list[float], label: str, path: str) -> dict:
    n = len(scores)
    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = sum(1 for s in scores if s == 0.0)
    score = sum(scores) / n if n else 0.0
    return {
        "label": label,
        "path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo": elo_from_score(score, ELO_CAP),
    }


def make_plot(results: dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(results, key=label_sort_key)
    xs = list(range(len(labels)))
    elos = [results[l]["elo"] for l in labels]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhline(0.0, ls="--", lw=1, color="#9aa0a6")
    ax.plot(xs, elos, "-o", color="#1a73e8", lw=2.2, markersize=5,
            markerfacecolor="#1a73e8", markeredgecolor="#1a73e8")

    for x, l in zip(xs, labels):
        r = results[l]
        ax.annotate(f"{r['wins']}-{r['draws']}-{r['losses']}", (x, r["elo"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", va="bottom", fontsize=7, color="#5f6368")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel(f"sl_best Elo advantage  (capped +/-{int(ELO_CAP)})")
    ax.set_title("Gauntlet vs sl_best  —  W-D-L is sl_best's; "
                 "lower line = stronger checkpoint")
    ax.grid(True, axis="both", alpha=0.25, lw=0.6)
    ax.margins(x=0.02, y=0.12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"  wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Best-net gauntlet vs training checkpoints.")
    ap.add_argument("--force", action="store_true",
                    help="replay checkpoints already present in the results JSON")
    ap.add_argument("--plot-only", action="store_true",
                    help="rebuild the plot from the results JSON without playing")
    ap.add_argument("--only", nargs="+", metavar="LABEL",
                    help="only these checkpoint labels, e.g. --only 040 120 final")
    args = ap.parse_args()

    results = load_results()

    if args.plot_only:
        if not results:
            sys.exit(f"no results in {RESULTS_JSON} to plot")
        make_plot(results, PLOT_PNG)
        return

    checkpoints = find_checkpoints()
    if args.only:
        wanted = set(args.only)
        checkpoints = [p for p in checkpoints if checkpoint_label(p) in wanted]
    if not checkpoints:
        sys.exit(f"no checkpoints matched {CKPT_GLOB} in {CKPT_DIR}")

    print(f"Loading best net: {BEST}")
    best = load_net(BEST)
    converter = Converter()

    for i, ckpt in enumerate(checkpoints, 1):
        label = checkpoint_label(ckpt)
        if label in results and not args.force:
            r = results[label]
            print(f"[{i}/{len(checkpoints)}] {label}: cached "
                  f"({r['wins']}-{r['draws']}-{r['losses']}, Elo {r['elo']:+.0f}) — skip")
            continue

        print(f"\n[{i}/{len(checkpoints)}] === sl_best vs {label} "
              f"({MODE}, {OPENINGS} openings -> {2 * OPENINGS} games) ===")
        net = load_net(ckpt)
        # A = best, B = checkpoint; run_match returns A's (=best's) scores.
        best_scores, games = run_match(
            best, net, "sl_best", label, converter, MODE, SIMS,
            BATCH_SIZE, OPENINGS, OPENING_PLIES, SEED,
        )
        rec = match_record(best_scores, label, ckpt)
        results[label] = rec
        save_results(results)  # checkpoint-by-checkpoint so a crash loses at most one
        print(f"  -> sl_best {rec['wins']}-{rec['draws']}-{rec['losses']}  "
              f"score {rec['score']:.3f}  Elo {rec['elo']:+.0f}")

        if SAVE_PGN:
            save_pgn(games, os.path.join(PGN_DIR, f"sl_best_vs_{label}.pgn"))

    make_plot(results, PLOT_PNG)
    print(f"\nDone. Results: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
