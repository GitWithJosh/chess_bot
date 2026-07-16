import chess, numpy as np
from helpers.converter import Converter
from monte_carlo_tree_search.mcts_v2 import MCTS
from monte_carlo_tree_search.nodes_and_edges_v2 import Node

def _ctx():
    net = globals().get("_NET")
    if net is None:
        raise RuntimeError("Run the play() cell first so _NET is set.")
    return net, (getattr(net, "_search_converter", None) or Converter())

def _val(net, conv, board):
    """Value head: (scalar win-loss, wdl) from side-to-move perspective."""
    _, wdl = net.predict(conv.board_to_input_tensor(board))
    wdl = np.asarray(wdl, float)
    return float(wdl[0] - wdl[2]), wdl

def _search(net, conv, board, sims, c_puct, bs=16):
    mcts = MCTS(network=net, converter=conv, num_simulations=sims, c_puct=c_puct)
    r = Node(board.copy())
    mcts.search_batched(r, add_noise=False, batch_size=bs)
    return r

def _parse(board, mv):
    try: return board.parse_san(mv)
    except ValueError: return chess.Move.from_uci(mv)

def _pv(net, conv, board, node, depth):
    """Most-visited path: per-ply move, visits, backed-up Q, raw value head."""
    line, b = [], board.copy()
    for _ in range(depth):
        if not node.edges: break
        e = max(node.edges, key=lambda x: x.N)
        raw, _ = _val(net, conv, b)
        line.append(dict(san=b.san(e.move), N=e.N, Q=e.Q, raw=raw))
        b.push(e.move); node = e.child_node
        if node is None: break
    return line

def _print_pv(title, line):
    print(title)
    print(f"    {'mv':<7}{'N':>6}{'Q(search)':>11}{'raw(value)':>12}   note")
    for p in line:
        disagree = abs(p['Q'])>0.15 and abs(p['raw'])>0.15 and (p['Q']>0)!=(p['raw']>0)
        note = "<-- value head & search DISAGREE" if disagree else ""
        print(f"    {p['san']:<7}{p['N']:>6}{p['Q']:>+11.3f}{p['raw']:>+12.3f}   {note}")

def analyze(fen, blunder=None, c_pucts=(0.5,1.5,3.0,6.0), sims=(100,400,1000),
            deep_c_puct=1.5, deep_sims=1000, top=10, pv_depth=10, bs=16):
    net, conv = _ctx()
    base = chess.Board(fen)
    bmove = _parse(base, blunder) if blunder else None
    turn = "White" if base.turn else "Black"

    # ---- static root read ----
    v0, wdl0 = _val(net, conv, base)
    print("="*74)
    print(f"{turn} to move   FEN {fen}")
    print(f"value head  W/D/L {wdl0[0]*100:4.0f}/{wdl0[1]*100:4.0f}/{wdl0[2]*100:4.0f}"
          f"   scalar {v0:+.3f}")
    r0 = Node(base.copy()); r0.expand(net, conv)
    pol = sorted(r0.edges, key=lambda e: e.P, reverse=True)
    print("\npolicy head — top priors:")
    for e in pol[:top]:
        tag = "  <-- BLUNDER" if bmove and e.move==bmove else ""
        print(f"    {base.san(e.move):<7}{e.P*100:5.1f}%{tag}")
    if bmove and bmove not in [e.move for e in pol[:top]]:
        be = next(e for e in pol if e.move==bmove)
        print(f"    ... blunder {base.san(bmove)} ranks {pol.index(be)+1}/{len(pol)}"
              f" at {be.P*100:.2f}%")

    # ---- sweep grid ----
    if bmove:
        print("\n" + "="*74)
        print("SWEEP  c_puct (rows) x sims (cols)")
        print("cell = chosenMove | blunderN%/blunderQ   ('*' = blunder was chosen)\n")
        corner = "cpuct|sims"
        head = f"{corner:>10}" + "".join(f"{s:>23d}" for s in sims)
        print(head); print("-"*len(head))
        for c in c_pucts:
            row = f"{c:>10.2f}"
            for s in sims:
                r = _search(net, conv, base, s, c, bs)
                tot = sum(e.N for e in r.edges) or 1
                best = max(r.edges, key=lambda e: e.N)
                be = next((e for e in r.edges if e.move==bmove), None)
                npct = 100*be.N/tot if be else 0.0
                q = be.Q if be else float("nan")
                star = "*" if (be and best.move==bmove) else ""
                row += f"{base.san(best.move)+'|'+f'{npct:3.0f}%/{q:+.2f}'+star:>23}"
            print(row)

    # ---- deep dive ----
    print("\n" + "="*74)
    print(f"DEEP DIVE  c_puct={deep_c_puct}  sims={deep_sims}")
    root = _search(net, conv, base, deep_sims, deep_c_puct, bs)
    edges = sorted(root.edges, key=lambda e: e.N, reverse=True)
    tot = sum(e.N for e in edges) or 1
    print(f"\nroot edges (top {top}):  move / N / N% / Q / P")
    for e in edges[:top]:
        tag = "  <-- BLUNDER" if bmove and e.move==bmove else ""
        print(f"    {base.san(e.move):<7}{e.N:6d}{100*e.N/tot:6.1f}%"
              f"{e.Q:+8.3f}{e.P*100:6.1f}%{tag}")

    print()
    _print_pv("PRINCIPAL VARIATION (search's chosen line):",
              _pv(net, conv, base, root, pv_depth))

    if bmove:
        be = next((e for e in root.edges if e.move==bmove), None)
        if be and be.child_node is not None:
            b2 = base.copy(); b2.push(bmove)
            head = [dict(san=base.san(bmove), N=be.N, Q=be.Q, raw=v0)]
            print()
            _print_pv(f"LINE AFTER THE BLUNDER {base.san(bmove)} — what search expects:",
                      head + _pv(net, conv, b2, be.child_node, pv_depth-1))

# usage, e.g. position right before ...Kf8:
analyze("2kr4/1pp2p2/p6Q/2p1p2b/2P1P2q/8/PPP2PPN/4R1K1 b - - 0 22", "f6")
