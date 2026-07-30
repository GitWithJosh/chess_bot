# Angelernt oder Autodidakt?

Supervised Learning gegen Self-Play Reinforcement Learning mit MCTS in
Schachagenten.

Student project for the module *Projekt Data Science und Künstliche
Intelligenz*, DHBW Mannheim, course WDSKI23B. Danny Hoffmann, Tim Lehmann,
Joshua Meyer, Philipp Meyer.

Two chess agents share one network architecture and one search. The only thing
that differs is where their knowledge came from. One learned from 23.5 million
human positions, the other from games against itself, starting on a checkpoint from supervised learning. The report in
`LaTeX/` compares them.

## Results

| Agent | Trained on | Strength | Compute |
| --- | --- | --- | --- |
| Supervised v4 | 23.5M Lichess positions, Stockfish soft targets | **~1920 Elo** (1855 to 1980) | 8 GPU-h training, 31 GPU-free h data prep |
| RL gen_4 | self-play only, seeded from a supervised checkpoint | **~1330 Elo** (extrapolated) | ~120 GPU-h |

Head to head over 50 games the RL net scored 0.07 against the supervised net.
It did however beat its own seed checkpoint by about 205 Elo, so self-play did
teach it something, just far less per GPU hour. Elo is measured against
Stockfish 17.1 at a fixed 0.75 seconds per move.

Both networks are the same 20 by 256 SE-residual tower, 25,042,444 trainable
parameters, reading a 20-plane position-only encoding and emitting 1858 policy
logits plus a win/draw/loss head. Search is PUCT with c_puct 1.5 at 1000
simulations.

## Play against them

The weights are not in the repository, they are about 100 MB each. Download
them from the Kaggle dataset, drop the `.weights.h5` files into `Weights/`,
then run the app. See [Weights/README.md](Weights/README.md) for the file names.

```bash
pip install -r requirements.txt
python main.py
```

Pick an opponent, choose whether it should search or answer from a single look
at the board, and set how many simulations it gets. Networks whose weights are
missing are greyed out, so an empty `Weights/` folder is obvious rather than
confusing. Human vs Human works without any weights at all.

Everything runs on CPU. A policy-only move takes well under a tenth of a
second, 200 simulations takes about 0.7 s and 1000 simulations, the setting
every reported Elo figure was measured at, takes roughly 3.5 s.

There is also a public Kaggle notebook that needs no installation and runs the
same networks on a GPU.

## Layout

```
main.py                     the playable app, local CPU counterpart to the notebook
Weights/                    put the downloaded .weights.h5 files here
LaTeX/                      the report
RL_kaggle_training_loop_clean.ipynb   self-play training loop

reinforcement_learning/     the shared core, plus the RL side
  networks/big_network.py     the 20x256 SE-residual network
  monte_carlo_tree_search/    PUCT search
  helpers/converter.py        board and move encoding
  move_lookup.json            the 1858 move index
  tests/                      pytest suite

supervised_learning/        the SL side
  train_supervised.py         training
  create_dataset/             Lichess filtering, Stockfish annotation
  inspect/                    probes and parameter sweeps
  compare_nets.py, gauntlet.py   match tooling

board/ move_generation/ game/ gui/ engines/ utils/    the app itself
```

`reinforcement_learning/` holds the network, the encoding and the search that
both paradigms use. The name is historical, it is where that code was first
written, and ten scripts under `supervised_learning/` import from it.

## Reproducing the numbers

Run these from the repository root. Most need Stockfish on `PATH` and the
weights in place, and several need the dataset, which is regenerable from the
public Lichess dumps but large.

| Script | Produces |
| --- | --- |
| `supervised_learning/train_supervised.py` | the training run and its convergence curves |
| `supervised_learning/compare_nets.py` | net against net matches |
| `supervised_learning/compare_nets_kaggle.py` | the Stockfish Elo anchors |
| `supervised_learning/gauntlet.py` | strength across training checkpoints |
| `supervised_learning/inspect/probe_value_head.py` | the value-head probes |
| `supervised_learning/inspect/tune_batch_size.py` | the MCTS batch size sweep |
| `supervised_learning/inspect/tune_search_params.py` | the c_puct and FPU sweep |
| `supervised_learning/inspect/inspect_network.py` | the parameter distribution |

## Tests

```bash
pytest reinforcement_learning/tests/
```

## Requirements

Python 3.11 or newer, TensorFlow, python-chess, pygame and numpy. Stockfish is
only needed for the evaluation scripts, not to play.
