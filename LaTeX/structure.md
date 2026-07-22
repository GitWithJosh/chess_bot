# Projektbericht — Gliederung

Zielumfang: ~18 Textseiten (4 Personen × 4–5 Seiten, ohne Titel/Inhaltsverzeichnis/Literatur).
Grundlagen und Umsetzung je etwa 50:50 gewichtet.

| § | Abschnitt | Seiten |
|---|---|---:|
| 1 | **Einleitung** — Use-Case, Forschungsfrage, Beitrag, Aufbau | 1.5 |
| 2 | **Theoretische Grundlagen** | **8.5** |
| 2.1 | Schach als formales Problem | 1.5 |
| 2.2 | Klassische Engines: Minimax, Alpha-Beta, Stockfish/NNUE | 1.0 |
| 2.3 | Brettrepräsentation & Zugkodierung | 1.0 |
| 2.4 | Neuronale Netze für Schach: CNN/ResNet, Policy- + Value-Head | 1.5 |
| 2.5 | Supervised Learning im Schachkontext | 1.0 |
| 2.6 | Monte Carlo Tree Search (UCT, PUCT) | 1.5 |
| 2.7 | AlphaZero-Paradigma: Self-Play Reinforcement Learning | 1.0 |
| 3 | **Implementierung** | **5.0** |
| 3.1 | Systemarchitektur — geteilte Komponenten (Netz, Encoding, Suche) | 1.0 |
| 3.2 | Supervised Learning: Datenpipeline, Modell, Training | 2.0 |
| 3.3 | Reinforcement Learning: MCTS-Kern, Self-Play, Trainings-Loop | 2.0 |
| 4 | **Evaluation & Ergebnisse** | **3.5** |
| 4.1 | Versuchsaufbau (ein gemeinsames Protokoll) | 0.75 |
| 4.2 | Trainingsverhalten & Konvergenz | 0.75 |
| 4.3 | Spielstärke: Stockfish-Level, Elo, Direktvergleich | 1.5 |
| 4.4 | Rechenaufwand & Inferenzzeit | 0.5 |
| 5 | **Prototyp / Demo (GUI)** | 0.75 |
| 6 | **Diskussion** — Interpretation, Limitierungen, Ausblick | 1.5 |
| 7 | **Fazit** | 0.5 |

## Leitgedanke

SL- und RL-Agent teilen sich dieselbe Netzarchitektur (`networks/big_network.py`),
dieselbe Zugkodierung (`helpers/converter.py`) und dieselbe Suche. Nur das
Trainingssignal unterscheidet sich. Dadurch misst Kapitel 4 ausschließlich das
Trainingsparadigma — ein sauberes kontrolliertes Experiment. §3.1 macht diese
Gemeinsamkeit explizit; §4 ist ein gemeinsames Kapitel mit einem einzigen Protokoll.