# Projektbericht — Gliederung

Zielumfang: ~18 Textseiten (ohne Titel/Inhaltsverzeichnis/Literatur).
Hauptautoren: **Danny** (Supervised Learning) und **Tim** (Reinforcement Learning).
Grundlagen und Umsetzung je etwa 50:50 gewichtet.

| § | Abschnitt | Seiten | Owner |
|---|---|---:|---|
| 1 | **Einleitung** — Use-Case, Forschungsfrage, Beitrag, Aufbau | 1.5 | Tim |
| 2 | **Theoretische Grundlagen** | **8.5** | |
| 2.1 | Schach als formales Problem | 1.5 | Tim |
| 2.2 | Die Elo-Zahl: Spielstärke quantifizieren | 0.75 | Tim |
| 2.3 | Klassische Engines: Minimax, Alpha-Beta, Stockfish/NNUE | 1.0 | Danny |
| 2.4 | Brettrepräsentation & Zugkodierung | 1.0 | Danny |
| 2.5 | Neuronale Netze für Schach: CNN/ResNet, Policy- + Value-Head | 1.25 | Danny |
| 2.6 | Supervised Learning im Schachkontext | 1.0 | Danny |
| 2.7 | Monte Carlo Tree Search (UCT, PUCT) | 1.25 | Tim |
| 2.8 | AlphaZero-Paradigma: Self-Play Reinforcement Learning | 0.75 | Tim |
| 3 | **Implementierung** | **5.0** | |
| 3.1 | Systemarchitektur — geteilte Komponenten + spielbare Engine (GUI) | 1.0 | gemeinsam |
| 3.2 | Supervised Learning: Datenpipeline, Modell, Training | 2.0 | Danny |
| 3.3 | Reinforcement Learning: MCTS-Kern, Self-Play, Trainings-Loop | 2.0 | Tim |
| 4 | **Evaluation & Ergebnisse** | **4.0** | |
| 4.1 | Versuchsaufbau (ein gemeinsames Protokoll) | 0.75 | gemeinsam |
| 4.2 | Ergebnisse Supervised Learning | 1.0 | Danny |
| 4.3 | Ergebnisse Reinforcement Learning | 1.0 | Tim |
| 4.4 | Direktvergleich: SL vs. RL (Elo, Head-to-Head) | 0.75 | gemeinsam |
| 4.5 | Rechenaufwand & Inferenzzeit | 0.5 | gemeinsam |
| 5 | **Diskussion** — Interpretation, Limitierungen, Ausblick | 1.5 | gemeinsam |
| 6 | **Fazit** | 0.5 | gemeinsam |

## Leitgedanke

SL- und RL-Agent teilen sich dieselbe Netzarchitektur (`networks/big_network.py`),
dieselbe Zugkodierung (`helpers/converter.py`) und dieselbe Suche. Nur das
Trainingssignal unterscheidet sich. Dadurch misst Kapitel 4 ausschließlich das
Trainingsparadigma — ein sauberes kontrolliertes Experiment. §3.1 macht diese
Gemeinsamkeit explizit.

## Was ist das "weitere ML-Modell"?

**Nicht Stockfish** — Stockfish ist nur Referenzgegner und Elo-Anker. Verglichen
werden die **eigenen** Modelle gegeneinander: das RL-Modell gegen das SL-Modell
(§4.4), sowie die verschiedenen SL-Ansätze untereinander (§4.2, z. B.
Outcome-only vs. Stockfish-annotiert).

## Prototyp / Proof-of-Concept

Die spielbare Engine (GUI) selbst — daher **kein eigener Abschnitt**, sondern in
die Systemarchitektur (§3.1) integriert und in der Einleitung benannt.

## Evaluation: ein Protokoll, getrennte Ergebnisse

Tim schlug vor, die Evaluation nach SL/RL zu trennen. Umgesetzt als: **ein**
gemeinsames Protokoll (§4.1), danach nach Ansatz getrennte Ergebnisse
(§4.2 SL / §4.3 RL) und ein gemeinsamer **Direktvergleich** (§4.4), wo die
Elo-Differenz zwischen den beiden Agenten tatsächlich zusammenläuft.

## Offene Punkte

1. `bibliography.bib` ist noch leer — Literatur eintragen (AlphaZero 2018,
   AlphaGo Zero 2017, Shannon 1950, Kocsis & Szepesvári 2006 (UCT),
   Browne et al. 2012 (MCTS-Survey), Campbell et al. 2002 (Deep Blue),
   NNUE-Doku, lc0-Doku via web.archive.org).
2. Ein einziges Evaluationsprotokoll festlegen, bevor weitere Ergebnisse
   erzeugt werden (Gegnersatz, Stockfish-Level, Zeit-/Sim-Budget, Eröffnungen,
   Partienzahl, Farbausgleich, Elo-Anker).
3. Trainingszeiten als Einschränkung berichten, nicht als Kernaussage
   (unterschiedliche Hardware).
