# Monte Carlo Tree Search (MCTS) am Beispiel der Schach-Grundstellung

## Überblick

Die MCTS ist zentral für AlphaZero. Ihr Ziel: Für eine gegebene Stellung den besten Zug finden – und dabei Trainingsdaten generieren. Für **jeden einzelnen Zug** im Spiel wird eine komplett neue MCTS durchgeführt (bei AlphaZero mit 800 Simulationen pro Zug).

Jede Simulation besteht aus vier Phasen: **Selektion → Expansion → Simulation (Evaluation) → Backpropagation.**

Im Folgenden wird das konkret an der Schach-Grundstellung durchgespielt. Zur Vereinfachung betrachten wir nur drei der 20 möglichen Züge: **e4**, **d4** und **Nf3**.

---

## Schritt 0 – Initialisierung

Die Wurzel des Suchbaums ist die Grundstellung (Weiß am Zug). Bevor die eigentliche Suchschleife startet, wird die Wurzel **expandiert**: Die Grundstellung wird ins neuronale Netz gefüttert. Das Netz gibt zwei Dinge aus:

- **Zugwahrscheinlichkeiten (P)** für jeden legalen Zug – diese geben an, wie vielversprechend das Netz den jeweiligen Zug einschätzt.
- **Einen Bewertungswert (v)** für die Stellung – dieser gibt an, wie gut die Position für den aktuellen Spieler ist (Bereich -1 bis +1).

Nehmen wir an, das Netz gibt aus:

| Zug | Prior-Wahrscheinlichkeit P |
|-----|---------------------------|
| e4  | 0.40                      |
| d4  | 0.35                      |
| Nf3 | 0.25                      |

Bewertungswert: **v = 0.1** (leichter Vorteil für Weiß).

Für jede Kante (= Zug) werden die Werte initialisiert:
- **N = 0** (Besuchszähler)
- **W = 0** (kumulierter Wert)
- **Q = 0** (durchschnittlicher Wert = W/N)
- **P** = vom Netz gegebene Wahrscheinlichkeit

---

## Simulation 1 – e4 wird erkundet

### Selektion
Wir starten an der Wurzel und müssen ein Kind auswählen. Dafür berechnen wir für jede Kante den Wert:

$$U = Q + u, \quad \text{wobei } u = c \cdot P \cdot \frac{\sqrt{\sum_{m'} N_{m'}}}{1 + N_m}$$

Da alle Kinder N=0 haben, ist der Q-Term überall 0. Es dominiert also der **Prior P** des Netzes. e4 hat das höchste P (0.4) → **e4 wird gewählt.**

### Expansion
Der Knoten nach 1.e4 ist ein Blattknoten (noch nie besucht). Wir füttern die Stellung nach 1.e4 ins Netz und bekommen:
- Zugwahrscheinlichkeiten für Schwarz: e5=0.3, c5=0.3, e6=0.2, ...
- Bewertungswert: **v = -0.05** (ungefähr ausgeglichen)

Die Kindkanten von "nach 1.e4" werden mit den neuen P-Werten angelegt.

### Simulation (Evaluation)
Bei AlphaZero gibt es **keine zufälligen Rollouts**. Der Wert v = -0.05, den das Netz ausgegeben hat, *ist* die Evaluation. Das ersetzt komplett die klassischen Random-Playouts.

### Backpropagation
Der Wert v = -0.05 wird den Pfad zurück zur Wurzel propagiert. Die Kante e4 wird aktualisiert:
- N: 0 → **1**
- W: 0 → **-0.05**
- Q: W/N = **-0.05**

---

## Simulation 2 – d4 wird erkundet

### Selektion
Zurück an der Wurzel. Jetzt hat e4 schon N=1, was seinen Explorations-Bonus u senkt. d4 und Nf3 haben noch N=0, also ist ihr Bonus maximal. Von den beiden hat d4 das höhere P (0.35) → **d4 wird gewählt.**

### Expansion
Netz bewertet die Stellung nach 1.d4: **v = 0.0**

### Backpropagation
Kante d4: N=1, W=0.0, Q=0.0

---

## Simulation 3 – Nf3 wird erkundet

Gleiche Logik: Nf3 ist der einzige noch unbesuchte Zug → wird gewählt, expandiert, Wert zurückpropagiert.

---

## Simulation 4 – Der Baum wird tiefer

### Selektion
Jetzt wird es interessant: Alle drei Wurzelkinder wurden mindestens einmal besucht. Selektion wählt per Q+u denjenigen mit dem besten Gesamtwert. Nehmen wir an, **e4 gewinnt** diese Auswahl.

Wir gehen in den Knoten nach 1.e4. Dort gibt es Kindknoten (bei Simulation 1 erzeugt): e5, c5, e6, ... Alle haben N=0, also dominiert wieder P. Sagen wir **e5 wird gewählt** (P=0.3).

### Expansion
Das ist ein Blattknoten. Netz bewertet die Stellung nach 1.e4 e5: **v = 0.12**

### Backpropagation
Jetzt werden **alle Kanten auf dem Pfad** aktualisiert – von unten nach oben:
1. Kante e5 (Ebene 2): N=1, W=0.12, Q=0.12
2. Kante e4 (Ebene 1): N=2, W=-0.05+0.12=0.07, Q=0.07/2=0.035

Der positive Wert der tieferen Stellung verbessert also auch die Bewertung von e4 an der Wurzel!

---

## Zustand nach einigen Simulationen

Nach z.B. 8 Simulationen könnte der Baum so aussehen:

```
                    ┌─────────────────┐
                    │  Grundstellung  │
                    │  Weiß am Zug    │
                    └───┬────┬────┬───┘
                        │    │    │
                  e4    │  d4│    │ Nf3
                P=0.4   │P=0.35  │ P=0.25
                        │    │    │
               ┌────────┘    │    └────────┐
               ▼             ▼             ▼
        ┌─────────────┐ ┌──────────┐ ┌──────────┐
        │  nach 1.e4  │ │ nach 1.d4│ │nach 1.Nf3│
        │  N=5, Q=0.08│ │ N=2,Q=.02│ │N=1,Q=-.03│
        └──┬──────┬───┘ └──────────┘ └──────────┘
           │      │
     e5    │      │ c5
           ▼      ▼
    ┌───────────┐ ┌───────────┐
    │1.e4 e5    │ │1.e4 c5    │
    │N=2, Q=0.12│ │N=1, Q=0.05│
    └─────┬─────┘ └───────────┘
          │
    Nf3   │
          ▼
    ┌───────────┐
    │1.e4 e5    │
    │2.Nf3      │
    │N=1, Q=0.15│
    └───────────┘
```

Man sieht: Der Baum wächst **selektiv** – dort, wo das Netz vielversprechende Züge sieht (hohes P) und wo die bisherige Suche gute Werte gefunden hat (hohes Q), wird tiefer gesucht.

---

## Nach 800 Simulationen – Zugauswahl

Nach allen 800 Simulationen ist der Baum an den vielversprechenden Stellen vielleicht 10–15 Züge tief, während schwache Züge nur 1–2 Mal besucht wurden.

Jetzt kommt die Zugauswahl: Wir schauen uns die **Besuchszähler N** der direkten Kinder der Wurzel an:

| Zug | Besuche N | Zugwahrscheinlichkeit π |
|-----|-----------|------------------------|
| e4  | 350       | 350/800 = **0.4375**   |
| d4  | 300       | 300/800 = **0.3750**   |
| Nf3 | 150       | 150/800 = **0.1875**   |

Die Formel dafür ist: $\pi_m = \frac{N_m^{1/\tau}}{\sum_n N_n^{1/\tau}}$

Der Parameter τ steuert dabei die Exploration: Bei τ=1 entsprechen die Wahrscheinlichkeiten direkt den Besuchsverhältnissen. Bei τ→0 wird quasi immer der meistbesuchte Zug gewählt.

---

## Trainingsdaten

Diese Zugwahrscheinlichkeiten π sind wertvoller als das, was das Netz alleine ausgegeben hätte (die Prior-Wahrscheinlichkeiten P), weil die MCTS durch die Suche zusätzliches Wissen generiert hat.

Aus diesem einen Zug erhalten wir ein Trainings-Tripel:

- **Position**: Grundstellung (als Bit-Vektor kodiert)
- **Zugwahrscheinlichkeiten**: [e4: 0.4375, d4: 0.375, Nf3: 0.1875, ...]
- **Spielergebnis**: Wird am Ende des Spiels eingetragen (z.B. 1 für Weiß gewinnt)

Dann wird ein Zug gewählt (z.B. e4), der **gesamte Suchbaum wird verworfen**, und eine komplett neue MCTS startet für den nächsten Zug (Schwarz am Zug nach 1.e4).

---

## Unterschied zur klassischen MCTS

| Aspekt | Klassische MCTS | AlphaZero MCTS |
|--------|----------------|----------------|
| **Selektion** | UCT basierend auf Besuchszähler und Gewinnrate | UCT mit Prior-Wahrscheinlichkeit P vom Netz |
| **Expansion** | Ein neuer Kindknoten | Alle Kindknoten + Netz liefert P-Werte |
| **Simulation** | Zufällige Rollouts bis Spielende | **Keine Rollouts** – Netz gibt Bewertung v aus |
| **Backpropagation** | Spielergebnis wird zurückpropagiert | Netz-Bewertung v wird zurückpropagiert |

Der entscheidende Vorteil: Das Netz ersetzt die zufälligen Rollouts durch eine informierte Bewertung. Dadurch ist die Suche viel effizienter – wenige hundert Simulationen reichen aus, statt tausender zufälliger Spiele.

---

## Zusammenfassung des Kreislaufs

1. **MCTS für einen Zug**: 800× (Selektion → Expansion → Evaluation durch Netz → Backpropagation)
2. **Zugauswahl**: Besuchszähler → Zugwahrscheinlichkeiten π
3. **Zug spielen**, neuen Suchbaum starten
4. **Wiederholen** bis Spielende
5. **Trainingsdaten**: Für jeden Zug das Tripel (Position, π, Spielergebnis)
6. **Netz trainieren** mit den gesammelten Daten → verbessertes Netz
7. **Neuer Self-Play-Zyklus** mit dem verbesserten Netz
