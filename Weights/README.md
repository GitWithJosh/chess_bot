# Weights

Put the trained network weights in this folder. They are not in the repository
because each file is about 100 MB.

## Download

The five networks live in the Kaggle dataset that goes with the public play
notebook. Download it, then copy the `.weights.h5` files in here so the folder
looks like this.

```
Weights/
  Supervised_v1.weights.h5
  Supervised_v2.weights.h5
  Supervised_v3.weights.h5
  Supervised_v4.weights.h5
  RL.weights.h5
```

The file names matter, `main.py` looks them up by name. You do not need all
five. Anything missing is shown greyed out in the menu and everything else
still works.

## The networks

| File | What it is |
| --- | --- |
| `Supervised_v1.weights.h5` | first supervised net, trained on game results only |
| `Supervised_v2.weights.h5` | supervised, value head still had a systematic flaw |
| `Supervised_v3.weights.h5` | supervised, flaw fixed, roughly 100 Elo below v4 |
| `Supervised_v4.weights.h5` | the strongest net here, about 1920 Elo |
| `RL.weights.h5` | learned purely from self-play, about 1330 Elo |

Elo figures are the ones measured in the report, against Stockfish at a fixed
0.75 seconds per move.

All five share the same architecture, a 20 by 256 SE-residual tower with
25,042,444 trainable parameters, and the same 20-plane position-only input
encoding, so any of them loads into the same network.

## Then

```
python main.py
```
