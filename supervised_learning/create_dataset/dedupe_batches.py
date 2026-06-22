"""
Remove whole-game duplicates from the Batch*.pgn dataset.

The batch generator's resume logic reprocesses an interrupted PGN file from the
START (the in-progress file isn't in processed_files yet), re-emitting games that
were already flushed to earlier batches. So the same game can appear many times —
observed up to 250x for a game sitting near a batch boundary in a file that was
resumed repeatedly. At the position level this exploded into ~35% exact-row
duplication in pool_games, even though only ~7% of *games* are redundant.

This removes those exact duplicate game records and nothing else. Two games are
"the same" iff their whitespace-normalised full record (headers + movetext) is
identical — bug duplicates are byte-identical, so games that merely share an
opening and then DIVERGE have different movetext and are all kept.

Run from the chess_bot root:
    python supervised_learning/create_dataset/dedupe_batches.py --dry-run   # report only
    python supervised_learning/create_dataset/dedupe_batches.py             # dedupe in place

In-place mode moves the originals to supervised_data/_batches_with_dupes_backup/
first (so it is reversible — delete that folder once you're happy), then writes
the deduped Batch*.pgn back into supervised_data/.

AFTER running: delete pools/pool_games_w*.csv, then rerun process_game_data.py and
build_dataset.py. The puzzle and tablebase pools are unaffected.
"""

import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_common as dc  # noqa: E402

DATA_DIR   = os.path.join(dc.repo_root(), "supervised_learning", "supervised_data")
BACKUP_DIR = os.path.join(DATA_DIR, "_batches_with_dupes_backup")
TMP_DIR    = os.path.join(DATA_DIR, "_dedup_tmp")

_norm = lambda block: re.sub(r"\s+", " ", block.strip())


def _batch_num(path: str) -> int:
    return int(re.search(r"Batch(\d+)", os.path.basename(path)).group(1))


def batch_files() -> list[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, "Batch*.pgn")), key=_batch_num)


def iter_games(files: list[str]):
    """Yield each game's raw text. Games start at a line beginning '[Date \"'."""
    cur: list[str] = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith('[Date "') and cur:
                    yield "".join(cur)
                    cur = [line]
                else:
                    cur.append(line)
    if cur:
        yield "".join(cur)


class _BatchWriter:
    """Repacks kept games into uniform Batch{N}.pgn files of `size` games each."""
    def __init__(self, out_dir: str, size: int):
        self.out_dir, self.size = out_dir, size
        self.buf: list[str] = []
        self.num, self.written = 1, 0

    def add(self, game_text: str) -> None:
        self.buf.append(game_text.rstrip("\n") + "\n\n")  # one blank line between games
        if len(self.buf) >= self.size:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        path = os.path.join(self.out_dir, f"Batch{self.num}.pgn")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(self.buf)
        print(f"  wrote {os.path.basename(path)}  ({len(self.buf):,} games)", flush=True)
        self.written += len(self.buf)
        self.num += 1
        self.buf = []

    def close(self) -> None:
        self._flush()


def main() -> None:
    dry = "--dry-run" in sys.argv
    files = batch_files()
    if not files:
        print(f"No Batch*.pgn in {DATA_DIR}")
        sys.exit(1)

    sizes = [sum(1 for l in open(p, encoding="utf-8", errors="replace") if l.startswith('[Date "'))
             for p in files]
    batch_size = max(sizes)
    print(f"{len(files)} batches | {sum(sizes):,} games | batch_size {batch_size:,}\n")

    writer = None
    if not dry:
        if os.path.isdir(TMP_DIR):
            shutil.rmtree(TMP_DIR)
        os.makedirs(TMP_DIR)
        writer = _BatchWriter(TMP_DIR, batch_size)

    seen: set[str] = set()
    total = kept = 0

    for g in iter_games(files):
        if not g.strip():
            continue
        total += 1
        key = _norm(g)
        if key in seen:
            continue
        seen.add(key)
        kept += 1
        if writer is not None:
            writer.add(g)

    removed = total - kept
    print(f"\n{total:,} games -> {kept:,} unique  "
          f"({removed:,} duplicate copies removed, {removed / total:.1%})")

    if dry:
        print("\n(dry run — nothing written)")
        return

    writer.close()
    assert writer.written == kept, (writer.written, kept)

    # Back up originals (reversible), then swap deduped batches into place.
    if os.path.isdir(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    os.makedirs(BACKUP_DIR)
    for p in files:
        shutil.move(p, os.path.join(BACKUP_DIR, os.path.basename(p)))
    for p in sorted(glob.glob(os.path.join(TMP_DIR, "Batch*.pgn")), key=_batch_num):
        shutil.move(p, os.path.join(DATA_DIR, os.path.basename(p)))
    shutil.rmtree(TMP_DIR)

    print(f"\noriginals backed up -> {BACKUP_DIR}")
    print(f"deduped Batch*.pgn   -> {DATA_DIR}  ({writer.num - 1} batches)")
    print("\nNext: delete pools/pool_games_w*.csv, then rerun "
          "process_game_data.py and build_dataset.py.")


if __name__ == "__main__":
    main()
