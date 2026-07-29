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

def _depth(root):
    """(mean, sel, pv) plies. mean = avg depth a sim reached — every sim walks
    root->leaf bumping one edge per ply, so sum(edge.N)/sims IS the mean path
    length (exact, not sampled). sel = deepest node built. pv = most-visited
    chain, the closest thing to an alpha-beta 'depth N'."""
    tot = sel = 0
    st = [(root, 0)]
    while st:
        n, d = st.pop()
        if d > sel: sel = d
        for e in n.edges:
            tot += e.N
            if e.child_node is not None: st.append((e.child_node, d+1))
    mean = tot / root.total_visits if root.total_visits else 0.0
    pv, n = 0, root
    while n.edges:
        e = max(n.edges, key=lambda x: x.N)
        if e.N == 0 or e.child_node is None: break
        n = e.child_node; pv += 1
    return mean, sel, pv

def _pv(net, conv, board, node, depth):
    """Most-visited path: per-ply move, visits, backed-up Q, raw value head."""
    line, b = [], board.copy()
    for _ in range(depth):
        if not node.edges: break
        e = max(node.edges, key=lambda x: x.N)
        if e.N == 0: break   # unsearched top prior at a leaf — not part of the PV
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

def analyze(fen, c_pucts=(0.5,1.5,3.0,6.0), sims=(100,400,800,1000,2000),
            deep_c_puct=1.5, deep_sims=2000, top=10, pv_depth=10, bs=16):
    net, conv = _ctx()
    base = chess.Board(fen)
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
        print(f"    {base.san(e.move):<7}{e.P*100:5.1f}%")

    # ---- sweep grid ----
    print("\n" + "="*74)
    print("SWEEP  c_puct (rows) x sims (cols)")
    print("cell = chosenMove | itsN%/itsQ   (N% = visit share = how convinced)")
    corner, w = "cpuct|sims", 22
    head = f"{corner:>10}" + "".join(f"{s:>{w}d}" for s in sims)
    print(head); print("-"*len(head))
    dep = {}
    for c in c_pucts:
        row = f"{c:>10.2f}"
        for s in sims:
            r = _search(net, conv, base, s, c, bs)
            dep[(c,s)] = _depth(r)
            tot = sum(e.N for e in r.edges) or 1
            best = max(r.edges, key=lambda e: e.N)
            cell = f"{base.san(best.move)}|{100*best.N/tot:3.0f}%/{best.Q:+.2f}"
            row += f"{cell:>{w}}"
        print(row)

    # ---- depth grid (same searches, no extra cost) ----
    print("\nDEPTH  mean/sel/pv plies   mean = avg depth a sim reached,")
    print("       sel = deepest node built, pv = most-visited chain length")
    head = f"{corner:>10}" + "".join(f"{s:>14d}" for s in sims)
    print(head); print("-"*len(head))
    for c in c_pucts:
        row = f"{c:>10.2f}"
        for s in sims:
            m, sd, pv = dep[(c,s)]
            row += f"{f'{m:.1f}/{sd}/{pv}':>14}"
        print(row)

    # ---- deep dive ----
    print("\n" + "="*74)
    print(f"DEEP DIVE  c_puct={deep_c_puct}  sims={deep_sims}")
    root = _search(net, conv, base, deep_sims, deep_c_puct, bs)
    edges = sorted(root.edges, key=lambda e: e.N, reverse=True)
    tot = sum(e.N for e in edges) or 1
    dmean, dsel, dpv = _depth(root)
    print(f"\ndepth: mean {dmean:.2f} plies   deepest node {dsel}   PV chain {dpv}")
    print(f"\nroot edges (top {top}):  move / N / N% / Q / P")
    for e in edges[:top]:
        print(f"    {base.san(e.move):<7}{e.N:6d}{100*e.N/tot:6.1f}%"
              f"{e.Q:+8.3f}{e.P*100:6.1f}%")

    print()
    # print the whole PV even when it runs past pv_depth — truncating it is what
    # made the reachable depth invisible in the first place
    _print_pv("PRINCIPAL VARIATION (search's chosen line):",
              _pv(net, conv, base, root, max(pv_depth, dpv)))

# usage, e.g. position right before ...Kf8:
analyze("2kr4/1pp2p2/p6Q/2p1p2b/2P1P2q/8/PPP2PPN/4R1K1 b - - 0 22")
