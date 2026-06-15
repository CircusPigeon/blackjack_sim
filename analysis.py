"""Analysis layer. Consumes the engine's record log (a columnar dict, one entry
per column per row) and produces tables, CSV/JSON exports, and graphs.

The aggregations use only the standard library so they always work; pandas and
matplotlib are imported lazily inside the functions that need them."""

import csv
import json
import math

# One row per (round, guest). This schema is the contract between the engine and
# every downstream table/graph, so it is kept deliberately small and stable.
RECORD_COLUMNS = ["round", "strategy", "true_count", "bet", "wagered", "result", "bankroll"]


def num_rows(records):
    return len(records["round"])


def export_csv(records, path):
    n = num_rows(records)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RECORD_COLUMNS)
        for i in range(n):
            w.writerow([records[c][i] for c in RECORD_COLUMNS])
    return path


def export_meta(config_dict, summary_rows, path):
    blob = {"config": config_dict,
            "summary": [{"strategy": s, "wagered": w, "profit": r, "edge_pct": e}
                        for (s, w, r, e) in summary_rows]}
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    return path


def summary(records):
    """Per-strategy totals: (strategy, wagered, profit, edge%)."""
    acc = {}
    for i in range(num_rows(records)):
        s = records["strategy"][i]
        d = acc.setdefault(s, [0, 0])
        d[0] += records["wagered"][i]
        d[1] += records["result"][i]
    rows = []
    for s in acc:
        wagered, profit = acc[s]
        edge = (100.0 * profit / wagered) if wagered else 0.0
        rows.append((s, wagered, profit, edge))
    return rows


def summary_stats(records):
    """Per-strategy stats with a 95% confidence interval on the edge and the win
    rate in units. A single run's edge is a noisy estimate, and the CI says how
    noisy: edge = sum(result)/sum(bet), and Var(sum result) ~ hands * var(result),
    so SE(edge%) = 100 * sd(result) / (mean_bet * sqrt(hands)).

    Returns a list of dicts: strategy, wagered, profit, edge (%), ci (95% half-
    width, %), win100 (units per 100 hands), hands. Bets/results are in dollars
    (unit x units); the unit is the smallest positive bet, so win100 is in units."""
    n = num_rows(records)
    pos = [records["bet"][i] for i in range(n) if records["bet"][i] > 0]
    unit = min(pos) if pos else 1.0
    acc = {}
    for i in range(n):
        s = records["strategy"][i]
        d = acc.setdefault(s, [0.0, 0.0, 0.0, 0])     # bet, result, result^2, hands
        r = records["result"][i]
        d[0] += records["bet"][i]
        d[1] += r
        d[2] += r * r
        d[3] += 1
    out = []
    for s, (wagered, profit, res2, hands) in acc.items():
        edge = (100.0 * profit / wagered) if wagered else 0.0
        mean_bet = (wagered / hands) if hands else 0.0
        var_res = (res2 / hands - (profit / hands) ** 2) if hands else 0.0
        se = (100.0 * math.sqrt(max(var_res, 0.0)) / (mean_bet * math.sqrt(hands))) \
            if (mean_bet > 0 and hands > 0) else 0.0
        win100 = (100.0 * (profit / hands) / unit) if (hands and unit) else 0.0
        out.append({"strategy": s, "wagered": wagered, "profit": profit, "edge": edge,
                    "ci": 1.96 * se, "win100": win100, "hands": hands})
    return out


def edge_by_true_count(records, strategy, lo=-5, hi=10):
    """Bucket rounds by floor(true count) and return (bucket, hands, edge%).

    This is the key validation curve: edge should climb with the count and cross
    zero near +1."""
    buckets = {}
    for i in range(num_rows(records)):
        if records["strategy"][i] != strategy:
            continue
        if records["wagered"][i] == 0:          # skip rounds the hero sat out (barred/ruined)
            continue
        b = int(math.floor(records["true_count"][i]))
        b = max(lo, min(hi, b))
        d = buckets.setdefault(b, [0, 0, 0])   # hands, wagered, result
        d[0] += 1
        d[1] += records["wagered"][i]
        d[2] += records["result"][i]
    rows = []
    for b in sorted(buckets):
        hands, wagered, result = buckets[b]
        edge = (100.0 * result / wagered) if wagered else 0.0
        rows.append((b, hands, edge))
    return rows


def accumulate_edge(records, acc, lo=-5, hi=10):
    """Fold one run's records into a running edge-by-true-count accumulator so the
    curve can be pooled across many trials without storing every hand.
    acc maps strategy -> {bucket: [hands, wagered, result]}."""
    for i in range(num_rows(records)):
        if (records["wagered"][i] == 0):
            continue
        s = records["strategy"][i]
        b = max(lo, min(hi, int(math.floor(records["true_count"][i]))))
        d = acc.setdefault(s, {}).setdefault(b, [0, 0, 0])
        d[0] += 1
        d[1] += records["wagered"][i]
        d[2] += records["result"][i]


def edge_rows_from_acc(acc, strategy):
    buckets = acc.get(strategy, {})
    rows = []
    for b in sorted(buckets):
        hands, wagered, result = buckets[b]
        edge = (100.0 * result / wagered) if wagered else 0.0
        rows.append((b, hands, edge))
    return rows


def print_table(rows, headers):
    body = [[str(x) for x in r] for r in rows]
    grid = [list(headers)] + body
    widths = [max(len(grid[r][c]) for r in range(len(grid))) for c in range(len(headers))]
    fmt = lambda r: "  ".join(r[c].rjust(widths[c]) for c in range(len(headers)))
    print(fmt([str(h) for h in headers]))
    print("  ".join("-" * w for w in widths))
    for r in body:
        print(fmt(r))


def to_dataframe(records):
    import pandas as pd
    return pd.DataFrame({c: records[c] for c in RECORD_COLUMNS})


def rounds_per_hour(num_players):
    """Standard casino pacing: each round takes longer with more players seated.
    Anchored to ~209 rounds/hr heads-up and ~52 rounds/hr at a full 7-player
    table (each player's hand adds time on top of dealer/shuffle overhead)."""
    seconds_per_round = 8.5 + 8.7 * num_players
    return 3600.0 / seconds_per_round


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# --- Figure builders (object API, return a Figure for embedding in a GUI) ---

def _new_figure(figsize=(9.0, 5.3)):
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return None
    # constrained layout reflows labels when the canvas is resized in a GUI.
    return Figure(figsize=figsize, layout="constrained")


def _style(ax, title=None, xlabel=None, ylabel=None):
    if (title is not None):
        ax.set_title(title, fontsize=14)
    if (xlabel is not None):
        ax.set_xlabel(xlabel, fontsize=12)
    if (ylabel is not None):
        ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=11)


def _drop_sparse(rows):
    """Hide true-count buckets with too few hands to mean anything. A bucket with
    a handful of hands can show a +/-100%+ edge (one lost double, say) that renders
    as a single giant bar and crushes the real +/-5% signal. Threshold scales with
    the run size; falls back to the raw rows if everything would be filtered."""
    if (not rows):
        return rows
    total = sum(r[1] for r in rows)
    min_hands = max(25, total // 2000)
    kept = [r for r in rows if r[1] >= min_hands]
    return kept or rows


def fig_edge_by_true_count(records, strategy):
    data = _drop_sparse(edge_by_true_count(records, strategy))
    fig = _new_figure()
    if (fig is None or not data):
        return None
    ax = fig.add_subplot(111)
    xs = [d[0] for d in data]
    ys = [d[2] for d in data]
    ax.axhline(0, color="0.5", lw=0.8)
    ax.bar(xs, ys, color=["#c0392b" if y < 0 else "#27ae60" for y in ys])
    _style(ax, "Player edge by true count  [%s]" % strategy,
           "true count (bucket)", "edge %  (green = player favored)")
    return fig


def fig_edge_rows(edge_rows, hero):
    data = edge_rows.get(hero) if edge_rows else None
    data = _drop_sparse(data) if data else data
    fig = _new_figure()
    if (fig is None or not data):
        return None
    ax = fig.add_subplot(111)
    xs = [d[0] for d in data]
    ys = [d[2] for d in data]
    ax.axhline(0, color="0.5", lw=0.8)
    ax.bar(xs, ys, color=["#c0392b" if y < 0 else "#27ae60" for y in ys])
    _style(ax, "Player edge by true count  [%s]" % hero,
           "true count (bucket)", "edge %  (green = player favored)")
    return fig


def fig_edge_rows_multi(rows_by_strategy, title="Player edge by true count"):
    """Overlay edge-by-true-count for several strategies on one axes (as lines, so
    they don't fight for the same bars). rows_by_strategy: {strategy: [(bucket,
    hands, edge%), ...]}."""
    fig = _new_figure()
    if (fig is None or not rows_by_strategy):
        return None
    ax = fig.add_subplot(111)
    ax.axhline(0, color="0.5", lw=0.8)
    plotted = 0
    for strat, rows in rows_by_strategy.items():
        data = _drop_sparse(rows) if rows else None
        if (not data):
            continue
        ax.plot([d[0] for d in data], [d[2] for d in data], marker="o", lw=1.6, label=strat)
        plotted += 1
    if (plotted == 0):
        return None
    _style(ax, title, "true count (bucket)", "edge %  (above 0 = player favored)")
    ax.legend(fontsize=11)
    return fig


def fig_param_sweep(param_label, xs, series, ylabel="player edge %"):
    """Plot a metric against a swept parameter, one line per strategy.
    series: {strategy: [y aligned with xs]}."""
    fig = _new_figure()
    if (fig is None or not xs):
        return None
    ax = fig.add_subplot(111)
    ax.axhline(0, color="0.5", lw=0.8)
    for strat, ys in series.items():
        ax.plot(xs, ys, marker="o", lw=1.8, label=strat)
    _style(ax, "Edge vs " + param_label, param_label, ylabel)
    ax.legend(fontsize=11)
    return fig


def fig_bankroll(records):
    fig = _new_figure()
    if (fig is None):
        return None
    ax = fig.add_subplot(111)
    series = {}
    for i in range(num_rows(records)):
        s = records["strategy"][i]
        series.setdefault(s, ([], []))
        series[s][0].append(records["round"][i])
        series[s][1].append(records["bankroll"][i])
    for s in series:
        ax.plot(series[s][0], series[s][1], lw=0.9, label=s)
    ax.axhline(0, color="0.5", lw=0.8)
    _style(ax, "Bankroll over time", "hand number", "bankroll")
    ax.legend(fontsize=11)
    return fig


def fig_survival(hands_by_strategy, cause="leaving the table"):
    """`cause` names what ends a session (barring from heat, going broke from the
    bankroll layer, or both) so the title matches what is actually being simulated
    -- barring only happens with live heat, ruin only with the live bankroll."""
    fig = _new_figure()
    if (fig is None):
        return None
    ax = fig.add_subplot(111)
    for s in hands_by_strategy:
        ax.hist(hands_by_strategy[s], bins=40, alpha=0.5,
                color=STRATEGY_COLORS.get(s), label=s)
    _style(ax, "How long the player lasts before " + cause,
           "hands played in a session", "number of sessions")
    ax.legend(fontsize=11)
    return fig


STRATEGY_COLORS = {"BASIC": "#2980b9", "COUNT": "#c0392b", "COUNTX": "#e67e22",
                   "COUNT0": "#d35400", "WONG": "#16a085", "TRACK": "#27ae60",
                   "ORACLE": "#8e44ad", "DEALER": "#7f8c8d"}


def fig_trajectory_fan(trajectories_by_strategy, hero=None):
    """Bankroll paths across sessions, one colour per strategy. Accepts a dict
    {strategy: [path, ...]} (the multi-strategy form) or, for backward
    compatibility, a bare list of paths (treated as the hero's)."""
    import matplotlib.lines as mlines
    if (isinstance(trajectories_by_strategy, list)):
        trajectories_by_strategy = {hero or "hero": trajectories_by_strategy}
    fig = _new_figure()
    if (fig is None or not trajectories_by_strategy):
        return None
    ax = fig.add_subplot(111)
    base, handles, n = None, [], 0
    for s, paths in trajectories_by_strategy.items():
        if (not paths):
            continue
        c = STRATEGY_COLORS.get(s, "#2980b9")
        for traj in paths:
            if (len(traj) > 2000):                   # downsample long paths to stay snappy
                step = len(traj) // 2000
                xs, ys = list(range(0, len(traj), step)), traj[::step]
            else:
                xs, ys = list(range(len(traj))), traj
            ax.plot(xs, ys, color=c, lw=0.6, alpha=0.28)
            if (base is None and traj):
                base = traj[0]
            n += 1
        handles.append(mlines.Line2D([], [], color=c, lw=2, label=s))
    if (not handles):
        return None
    if (base is not None):
        ax.axhline(base, color="0.4", lw=0.9)
    _style(ax, "Bankroll over time across %d sessions" % n, "hand number", "bankroll")
    ax.legend(handles=handles, fontsize=11)
    return fig


def fig_result_hist(results_by_strategy):
    fig = _new_figure()
    if (fig is None):
        return None
    ax = fig.add_subplot(111)
    colors = {"BASIC": "#2980b9", "COUNT": "#c0392b", "TRACK": "#27ae60",
              "ORACLE": "#8e44ad", "DEALER": "#7f8c8d"}
    for s in results_by_strategy:
        ax.hist(results_by_strategy[s], bins=40, alpha=0.5, color=colors.get(s), label=s)
    ax.axvline(0, color="0.3", lw=1.0)
    _style(ax, "Distribution of session results over trials",
           "session profit / loss (units)", "number of sessions")
    ax.legend(fontsize=11)
    return fig


def fig_heat_curve(rows):
    fig = _new_figure()
    if (fig is None or not rows):
        return None
    ax = fig.add_subplot(111)
    slopes = [r[0] for r in rows]
    total = [r[3] for r in rows]
    length = [r[2] for r in rows]
    best = max(range(len(rows)), key=lambda i: total[i])
    ax.plot(slopes, total, "o-", color="#8e44ad", lw=2, label="total profit / session")
    ax.scatter([slopes[best]], [total[best]], color="#8e44ad", s=140, zorder=5,
               edgecolor="black", label="best ramp")
    _style(ax, "Bet too steep and the pit bars you before you cash in",
           "bet aggressiveness (extra units per true count)", "total profit before back-off (units)")
    ax2 = ax.twinx()
    ax2.plot(slopes, length, "s--", color="#7f8c8d", lw=1)
    ax2.set_ylabel("hands before back-off", fontsize=12)
    ax2.tick_params(labelsize=11)
    ax.legend(fontsize=11, loc="upper right")
    return fig


def fig_risk_curve(rows):
    fig = _new_figure()
    if (fig is None or not rows):
        return None
    ax1 = fig.add_subplot(111)
    fr = [r[0] for r in rows]
    ror = [r[1] * 100 for r in rows]
    growth = [r[2] for r in rows]
    ax1.plot(fr, ror, "o-", color="#c0392b", label="risk of ruin")
    _style(ax1, "Risk vs reward by bet size (Kelly fraction)",
           "Kelly fraction  (1.0 = 'full Kelly')", "risk of going broke %")
    ax1.yaxis.label.set_color("#c0392b")
    ax1.axvline(1.0, color="0.7", ls=":", lw=1)
    ax2 = ax1.twinx()
    ax2.plot(fr, growth, "s--", color="#27ae60")
    ax2.set_ylabel("median bankroll growth (x)", color="#27ae60", fontsize=12)
    ax2.tick_params(labelsize=11)
    return fig


def plot_edge_by_true_count(records, strategy, path):
    plt = _pyplot()
    if (plt is None):
        return None
    data = edge_by_true_count(records, strategy)
    xs = [d[0] for d in data]
    ys = [d[2] for d in data]
    plt.figure(figsize=(7, 4))
    plt.axhline(0, color="0.5", lw=0.8)
    plt.bar(xs, ys, color=["#c0392b" if y < 0 else "#27ae60" for y in ys])
    plt.xlabel("true count (bucket)")
    plt.ylabel("edge %")
    plt.title("Edge vs true count  [" + strategy + "]")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def plot_edge_rows_multi(records, strategies, path):
    """Save the multi-strategy edge-by-true-count overlay (matches the GUI plot)."""
    rows_by = {s: edge_by_true_count(records, s) for s in strategies}
    fig = fig_edge_rows_multi(rows_by)
    if (fig is None):
        return None
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
        fig.savefig(path, dpi=110)
    except Exception:
        return None
    return path


def shuffle_quality(shuffler, n=312, trials=1000, slug=52, window=None):
    """Measure leftover structure after one application of a shuffle, using
    rotation-invariant statistics (so the final cut can't hide structure):

      neighbor_gap   = mean circular distance between cards that were adjacent
                       before the shuffle. Random shoe ~ n/4; small = slugs
                       stay together.
      slug_in_half   = fraction of a marked high-card slug that lands in its
                       best contiguous half-shoe window. Random ~ 0.5; higher =
                       the slug stays clumped and is trackable.
    """
    import numpy as np
    if (window is None):
        window = n // 2
    arangen = np.arange(n)
    gap_sum = 0.0
    cohesion_sum = 0.0
    for _ in range(trials):
        cards = list(range(n))
        shuffler.shuffle(cards)
        c = np.asarray(cards)
        pos = np.empty(n, dtype=int)
        pos[c] = arangen                      # pos[v] = new index of original card v
        d = np.abs(pos[1:] - pos[:-1])
        d = np.minimum(d, n - d)              # circular distance (rotation-invariant)
        gap_sum += float(d.mean())
        sp = np.sort(pos[:slug])
        ext = np.concatenate([sp, sp + n])
        best = 0
        for i in range(slug):
            cnt = int(np.searchsorted(ext, sp[i] + window) - i)
            if (cnt > best):
                best = cnt
        cohesion_sum += best / slug
    return gap_sum / trials, cohesion_sum / trials


def plot_shuffle_quality(results, path):
    plt = _pyplot()
    if (plt is None):
        return None
    labels = [r[0] for r in results]
    vals = [r[2] for r in results]
    baseline = results[0][2]               # the random.shuffle row is the floor
    x = list(range(len(labels)))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, vals, color="#8e44ad")
    plt.axhline(baseline, color="0.4", lw=1.0, ls="--",
                label="random baseline (%.2f)" % baseline)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("high-card slug in best half-shoe")
    plt.ylim(min(0.55, baseline - 0.05), 1.02)
    plt.title("Slug cohesion after shuffle  (dashed = random, higher = more trackable)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def plot_tracker_decay(riffles, track_edges, count_ref, path):
    plt = _pyplot()
    if (plt is None):
        return None
    plt.figure(figsize=(7, 4.5))
    plt.plot(riffles, track_edges, "o-", color="#c0392b", lw=2, label="shuffle tracker")
    plt.axhline(count_ref, color="#2980b9", ls="--", label="counter (Hi-Lo)")
    plt.axhline(0, color="0.7", lw=0.8)
    plt.xlabel("number of riffles (no cut)")
    plt.ylabel("player edge %")
    plt.title("Shuffle-tracker edge collapses as the shuffle improves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def plot_survival_hist(hands_by_strategy, path):
    plt = _pyplot()
    if (plt is None):
        return None
    plt.figure(figsize=(8, 4.5))
    colors = {"COUNT": "#c0392b", "TRACK": "#27ae60", "ORACLE": "#8e44ad"}
    for s in hands_by_strategy:
        plt.hist(hands_by_strategy[s], bins=40, alpha=0.5,
                 color=colors.get(s), label=s)
    plt.xlabel("hands survived before leaving the table")
    plt.ylabel("sessions")
    plt.title("How long the hero lasts (over trials)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def plot_heat_curve(rows, path):
    """rows = (slope, ev_per_hand, mean_length, total_profit, p_backed)."""
    plt = _pyplot()
    if (plt is None):
        return None
    slopes = [r[0] for r in rows]
    total = [r[3] for r in rows]
    length = [r[2] for r in rows]
    best = max(range(len(rows)), key=lambda i: total[i])
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(slopes, total, "o-", color="#8e44ad", lw=2, label="total profit / session")
    ax1.scatter([slopes[best]], [total[best]], color="#8e44ad", s=120, zorder=5,
                edgecolor="black", label="optimum")
    ax1.set_xlabel("ramp aggressiveness (units per true count)")
    ax1.set_ylabel("total profit until backoff (units)", color="#8e44ad")
    ax1.tick_params(axis="y", labelcolor="#8e44ad")
    ax2 = ax1.twinx()
    ax2.plot(slopes, length, "s--", color="#7f8c8d", lw=1, label="hands before backoff")
    ax2.set_ylabel("hands before backoff", color="#7f8c8d")
    ax2.tick_params(axis="y", labelcolor="#7f8c8d")
    ax1.set_title("Bet too steep and the pit bars you before you cash in")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_risk_curve(rows, path):
    plt = _pyplot()
    if (plt is None):
        return None
    fr = [r[0] for r in rows]
    ror = [r[1] * 100 for r in rows]
    growth = [r[2] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(fr, ror, "o-", color="#c0392b", label="risk of ruin")
    ax1.set_xlabel("Kelly fraction")
    ax1.set_ylabel("risk of ruin %", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.axvline(1.0, color="0.7", ls=":", lw=1)
    ax2 = ax1.twinx()
    ax2.plot(fr, growth, "s--", color="#27ae60", label="median growth (x)")
    ax2.set_ylabel("median bankroll growth (x)", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax1.set_title("Risk vs reward by Kelly fraction (dotted = full Kelly)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_bankroll_fan(traj, B0, path):
    plt = _pyplot()
    if (plt is None):
        return None
    plt.figure(figsize=(8, 4.5))
    for i in range(traj.shape[0]):
        plt.plot(traj[i], color="#2980b9", lw=0.4, alpha=0.25)
    plt.axhline(B0, color="0.4", lw=0.8)
    plt.xlabel("hands played")
    plt.ylabel("bankroll (units)")
    plt.title("Bankroll trajectories (half Kelly)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def plot_bankroll(records, path):
    plt = _pyplot()
    if (plt is None):
        return None
    series = {}
    for i in range(num_rows(records)):
        s = records["strategy"][i]
        series.setdefault(s, ([], []))
        series[s][0].append(records["round"][i])
        series[s][1].append(records["bankroll"][i])
    plt.figure(figsize=(8, 4))
    for s in series:
        plt.plot(series[s][0], series[s][1], lw=0.8, label=s)
    plt.axhline(0, color="0.5", lw=0.8)
    plt.xlabel("round")
    plt.ylabel("bankroll")
    plt.title("Bankroll trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path
