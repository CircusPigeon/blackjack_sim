"""Config-driven experiment runner.

run(config) dispatches on config.experiment and performs one of:
  game     -- play the card engine, report per-strategy edge + edge-by-true-count
  heat      -- the detection game: sweep ramp aggressiveness vs casino backoff
  bankroll  -- fractional-Kelly risk of ruin / growth sweep
  ceiling   -- composition-exact playing ceiling (combinatorial analysis)

The heat and bankroll experiments derive a counter calibration from the same
engine (cached on disk), and the ceiling experiment uses the same rules. Heavy
modules are imported lazily so a single experiment only pulls in what it needs."""

import os
import random
import numpy as np
import analysis as A
from blackjack import Blackjack, COUNTERS


def run_experiment(config, verbose=False, record=True, cancel=None, progress=None):
    """The card-game primitive: play config.rounds seeded hands, return the game.
    cancel (if given) is called periodically and may raise to abort the run;
    progress(done, total, label) is called periodically for UI feedback."""
    random.seed(config.seed)
    game = Blackjack(config=config, verbose=verbose, record=record)
    step = max(1, config.rounds // 100)
    for i in range(config.rounds):
        game.run()
        if (cancel is not None and (i & 1023) == 0):
            cancel()
        if (progress is not None and i % step == 0):
            progress(i, config.rounds, "hands")
    return game


def edges(game):
    """Final per-strategy edge (%) for the tracked guests of a finished game."""
    return {game.players[i].strategy: game.players[i].getEdge() * 100.0
            for i in range(game.numTracked)}


def sweep_edges(base_config, field, values, cancel=None, progress=None):
    """Run the game once per value of `field` (a Config field, e.g. 'numPacks' or
    'penetration'), holding everything else fixed. Returns (values, {strategy:
    [edge% aligned with values]}) -- the data for a parameter-sweep plot."""
    import dataclasses
    strategies = list(base_config.strategies)
    series = {s: [] for s in strategies}
    used = []
    for k, v in enumerate(values):
        if (cancel is not None):
            cancel()
        if (progress is not None):
            progress(k, len(values), "sweep")
        game = run_experiment(dataclasses.replace(base_config, **{field: v}), record=True)
        e = edges(game)
        used.append(v)
        for s in strategies:
            series[s].append(e.get(s, float("nan")))
    return used, series


def run(config, outdir="results", save_plots=True, cancel=None, progress=None):
    """Run the experiment selected by config.experiment. save_plots=False skips
    writing PNG/CSV/JSON (the returned dict carries the plot data so a GUI can
    render figures itself). cancel/progress, if given, are callables for abort
    and UI feedback."""
    os.makedirs(outdir, exist_ok=True)
    kind = config.experiment
    if (kind == "game"):
        return _run_game(config, outdir, save_plots, cancel, progress)
    if (kind == "heat"):
        return _run_heat(config, outdir, save_plots, cancel, progress)
    if (kind == "bankroll"):
        return _run_bankroll(config, outdir, save_plots, cancel, progress)
    if (kind == "stake"):
        return _run_stake(config, outdir, save_plots, cancel, progress)
    if (kind == "ceiling"):
        return _run_ceiling(config, outdir, save_plots, cancel)
    raise ValueError("unknown experiment '" + str(kind) + "' (game|heat|bankroll|stake|ceiling)")


def _fmt(e):
    return "%+.4f%%" % e


def _pick_hero(strategies):
    for s in ("COUNT", "ORACLE", "TRACK", "BASIC", "DEALER"):
        if (s in strategies):
            return s
    return strategies[0] if strategies else None


def _calibration(config=None, cancel=None, progress=None):
    import bankroll
    return bankroll.calibrate(bankroll.load_or_make_calibration(
        config=config, cancel=cancel, progress=progress))


def _run_game(config, outdir, save_plots=True, cancel=None, progress=None):
    if (config.trials > 1):
        return _run_game_trials(config, outdir, save_plots, cancel, progress)
    game = run_experiment(config, record=True, cancel=cancel, progress=progress)
    rec = game.records
    rows = A.summary(rec)
    stats = A.summary_stats(rec)
    print("[%s]  %d hands | shuffle=%s | dummies=%d"
          % (config.label, config.rounds, config.shuffle, config.dummyPlayers))
    A.print_table([(d["strategy"], "%.0f" % d["wagered"], "%+.0f" % d["profit"],
                    "%+.3f +/- %.3f" % (d["edge"], d["ci"]), "%+.2f" % d["win100"])
                   for d in stats],
                  ["strategy", "wagered", "profit", "edge % (95% CI)", "units/100"])
    if (any(d["strategy"] in COUNTERS and abs(d["edge"]) < d["ci"] for d in stats)):
        print("Note: a counting edge within its +/- CI of zero is not yet distinguishable "
              "from break-even -- raise Rounds (or use Trials) to resolve it.")

    present = [r[0] for r in rows]
    hero = next((s for s in ("COUNT", "ORACLE", "TRACK", "BASIC") if s in present), None)
    edge_rows = {s: A.edge_by_true_count(rec, s) for s in present}
    # One combined edge-by-true-count table: a true-count bucket per row, one edge
    # column per strategy (a strategy that sat the bucket out -- e.g. WONG -- shows "-").
    per = {s: {b: (n, e) for (b, n, e) in edge_rows[s]} for s in present}
    bucketsets = [set(per[s]) for s in present if per[s]]
    buckets = sorted(set().union(*bucketsets)) if bucketsets else []
    if (buckets):
        print("\nEdge by true count (edge %, one column per strategy):")
        tbl = []
        for b in buckets:
            hands = max((per[s][b][0] for s in present if b in per[s]), default=0)
            tbl.append(tuple([str(b), str(hands)]
                             + [(_fmt(per[s][b][1]) if b in per[s] else "-") for s in present]))
        A.print_table(tbl, ["true_count", "hands"] + present)
    if (save_plots and present):
        A.plot_edge_rows_multi(rec, present, os.path.join(outdir, config.label + "_edge.png"))

    if (config.heat_live or config.bankroll_live):
        print("\nLive counter outcomes (one composed session):")
        srows = []
        for g in game.guests:
            if (g.strategy in COUNTERS):
                status = "ruined" if g.ruined else ("barred" if g.barred else "still in")
                srows.append((g.strategy, "%.0f" % g.money, g.handsPlayed, status))
        A.print_table(srows, ["strategy", "final_roll", "hands_played", "status"])
        if (config.bankroll_live and save_plots):
            A.plot_bankroll(rec, os.path.join(outdir, config.label + "_bankroll.png"))

    if (save_plots):
        A.export_csv(rec, os.path.join(outdir, config.label + ".csv"))
        A.export_meta(config.to_dict(), rows, os.path.join(outdir, config.label + ".json"))
        print("\nWrote " + os.path.join(outdir, config.label + ".{csv,json}"))
    return {"summary": rows, "edges": edges(game), "records": rec, "hero": hero, "edge_rows": edge_rows}


def _run_game_trials(config, outdir, save_plots=True, cancel=None, progress=None):
    """Repeat the session config.trials times with different shuffles and report
    the distribution of per-session outcomes for every tracked strategy. Works for
    any strategy (BASIC included): the profit distribution shows the variance. With
    live heat/bankroll you also get bar/ruin rates and a survival distribution. A
    trial ends early only once every tracked player has left the table."""
    tracked = list(config.strategies)
    hero = _pick_hero(tracked)
    agg = {s: {"hands": [], "profit": [], "wagered": 0.0,
               "ruined": 0, "barred": 0, "unit": 10.0} for s in tracked}
    edge_acc = {}                                       # pooled edge-by-true-count
    trajectories = {s: [] for s in tracked}             # a few faint sample paths per strategy
    # The bold mean line is accumulated over ALL trials (not just the sampled paths),
    # so it is trustworthy even for low-edge/high-variance strategies (e.g. WONG),
    # whose 40-session mean would otherwise be pure noise. Ended sessions are carried
    # forward at their final balance (msum holds the running sum per hand index;
    # mfinal_sum lets a longer later trial back-fill the shorter earlier ones).
    msum = {s: np.zeros(0) for s in tracked}
    mfinal_sum = {s: 0.0 for s in tracked}
    n_traj = {s: 0 for s in tracked}
    for t in range(config.trials):
        if (cancel is not None):
            cancel()
        if (progress is not None):
            progress(t, config.trials, "trials")
        random.seed(config.seed + t)
        game = Blackjack(config=config, verbose=False, record=True)
        for _ in range(config.rounds):
            game.run()
            if (all(g.out for g in game.guests)):     # nobody left to simulate
                break
        A.accumulate_edge(game.records, edge_acc)      # fold this trial in, then drop it
        rec = game.records
        paths = {s: [] for s in tracked}
        for i in range(len(rec["round"])):
            s = rec["strategy"][i]
            if (s in paths):
                paths[s].append(rec["bankroll"][i])
        for s in tracked:
            p = paths[s]
            if (not p):
                continue
            if (len(trajectories[s]) < 40):
                trajectories[s].append(p)               # keep a few for the faint background
            arr = np.asarray(p, dtype=float)            # fold every trial into the mean
            Lp, f, Lg = len(arr), float(arr[-1]), len(msum[s])
            if (Lp > Lg):                               # this trial is longer: back-fill the rest
                msum[s] = np.concatenate([msum[s], np.full(Lp - Lg, mfinal_sum[s])])
            msum[s][:Lp] += arr
            if (Lp < len(msum[s])):                     # this trial ended early: hold its final
                msum[s][Lp:] += f
            mfinal_sum[s] += f
            n_traj[s] += 1
        for g in game.guests:
            a = agg[g.strategy]
            a["hands"].append(g.handsPlayed)
            a["profit"].append(g.money - g.startMoney)
            a["wagered"] += g.totalWagered
            a["ruined"] += int(g.ruined)
            a["barred"] += int(g.barred)
            a["unit"] = g.unit

    n = config.trials
    print("[%s]  %d trials x up to %d hands  (heat_live=%s, bankroll_live=%s, shuffle=%s, dummies=%d)"
          % (config.label, n, config.rounds, config.heat_live, config.bankroll_live,
             config.shuffle, config.dummyPlayers))
    rows = []
    for s in tracked:
        a = agg[s]
        hands = np.array(a["hands"])
        profit = np.array(a["profit"])
        edge = 100.0 * float(profit.sum()) / a["wagered"] if a["wagered"] > 0 else 0.0
        # CI on the pooled edge from the spread of per-trial profits (i.i.d. trials)
        sd = float(profit.std(ddof=1)) if len(profit) > 1 else 0.0
        se = 100.0 * float(np.sqrt(len(profit))) * sd / a["wagered"] if a["wagered"] > 0 else 0.0
        total_hands = float(hands.sum())
        win100 = 100.0 * (float(profit.sum()) / total_hands) / a["unit"] if (total_hands and a["unit"]) else 0.0
        rows.append((s,
                     "%.0f%%" % (100.0 * a["ruined"] / n),
                     "%.0f%%" % (100.0 * a["barred"] / n),
                     "%d" % int(np.median(hands)),
                     "%+.0f" % np.median(profit),
                     "%+.3f +/- %.3f" % (edge, 1.96 * se),
                     "%+.2f" % win100))
    A.print_table(rows, ["strategy", "P(ruin)", "P(bar)", "med.hands",
                         "med.profit", "edge % (95% CI)", "units/100"])
    survival = {s: agg[s]["hands"] for s in tracked}
    results = {s: agg[s]["profit"] for s in tracked}
    edge_rows = {s: A.edge_rows_from_acc(edge_acc, s) for s in tracked}
    # Name what actually ends a session: barring needs live heat, ruin needs the
    # live bankroll. Without either, sessions just run to the hand limit.
    if (config.heat_live and config.bankroll_live):
        cause = "being barred or going broke"
    elif (config.heat_live):
        cause = "being barred"
    elif (config.bankroll_live):
        cause = "going broke"
    else:
        cause = "the session ends"
    if (save_plots):
        A.plot_survival_hist(survival, os.path.join(outdir, config.label + "_survival.png"))
        print("Wrote results/" + config.label + "_survival.png")
    traj_means = {s: msum[s] / n_traj[s] for s in tracked if (n_traj[s] and len(msum[s]))}
    return {"trials": agg, "survival": survival, "survival_cause": cause, "results": results,
            "edge_rows": edge_rows, "hero": hero, "trajectories": trajectories,
            "trajectory_means": traj_means}


def _run_heat(config, outdir, save_plots=True, cancel=None, progress=None):
    import heat
    calib = _calibration(config, cancel, progress)
    slopes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    rows = heat.aggressiveness_sweep(calib, slopes, threshold=config.heat_threshold,
                                     warmup=config.heat_warmup, base_rate=config.heat_rate,
                                     pivot=config.ramp_start, min_bet=config.spread_min,
                                     max_bet=config.spread_max, maxHands=config.heat_maxHands,
                                     seed=config.seed)
    print("[heat] casino backoff threshold=%.1f, spread %g-%g units from TC %g, session <= %d hands"
          % (config.heat_threshold, config.spread_min, config.spread_max,
             config.ramp_start, config.heat_maxHands))
    A.print_table([("%.1f" % s, "%.4f" % ev, "%.0f" % ln, "%.1f" % tot, "%.1f%%" % (pb * 100))
                   for (s, ev, ln, tot, pb) in rows],
                  ["ramp", "ev/hand", "hands/sess", "total/sess", "P(barred)"])
    if (save_plots):
        A.plot_heat_curve(rows, os.path.join(outdir, "heat_curve.png"))
    best = max(rows, key=lambda r: r[3])
    print("optimal ramp ~%.1f units/TC (total %.1f)." % (best[0], best[3]))
    return {"heat": rows}


def _run_bankroll(config, outdir, save_plots=True, cancel=None, progress=None):
    import bankroll
    calib = _calibration(config, cancel, progress)
    print("[bankroll] edge %+.4f%%/unit, N0 %.0f hands. B0=%.0f units, ruin at %.0f%%, horizon %d"
          % (calib["edge_bw"] * 100, calib["n0"], config.bankroll_units,
             config.ruin_frac * 100, config.bankroll_horizon))
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    rows = bankroll.risk_curve(calib, fractions, B0=config.bankroll_units, table_max=500.0,
                               ruin_frac=config.ruin_frac, goal_mult=0,
                               maxHands=config.bankroll_horizon, M=8000, seed=config.seed)
    A.print_table([("%.2f" % f, "%.1f%%" % (ror * 100), "%.2fx" % g, "%.1f%%" % (dd * 100))
                   for (f, ror, g, pg, dd) in rows],
                  ["kelly", "RoR", "med.growth", "med.DD"])
    if (save_plots):
        A.plot_risk_curve(rows, os.path.join(outdir, "risk_vs_kelly.png"))
    return {"risk": rows}


def _run_stake(config, outdir, save_plots=True, cancel=None, progress=None):
    """Operational bankroll / risk of ruin for the *actual* discrete spread the
    counter plays (the same ramp Game and Heat use), bet at a fixed unit. Reports
    win rate, N0, DI / SCORE, the lifetime risk of ruin for the chosen bankroll, and
    -- the number a counter really wants -- the bankroll required to hold RoR to each
    target. Uses the cached per-unit calibration for the table (no re-dealing)."""
    import bankroll
    calib = _calibration(config, cancel, progress)
    wb = config.wong_below if ("WONG" in config.strategies) else None
    st = bankroll.spread_stats(calib, config.spread_min, config.spread_max,
                               config.ramp_start, config.spread_slope, wb)
    mu, sigma = st["mu"], st["sigma"]
    upd, hph, B0 = config.unit_dollars, config.hands_per_hour, config.bankroll_units
    print("[my spread] %g-%g units, ramp from TC %g at %g u/TC%s | %d-deck, pen %g, %s"
          % (config.spread_min, config.spread_max, config.ramp_start, config.spread_slope,
             (" | Wong < %g" % wb) if wb is not None else "", config.numPacks,
             config.penetration, "H17" if config.hitSoft17 else "S17"))
    if (mu <= 0):
        print("\nThis spread has no edge (EV per hand %+.4f units <= 0): no bankroll makes it "
              "safe -- widen the spread, deepen penetration, or find a better game." % mu)
        return {"stake": st}

    print("\nWin rate and variance (1 unit = the table minimum):")
    A.print_table([
        ("EV per hand", "%+.4f u" % mu, "$%+.2f" % (mu * upd)),
        ("win rate / 100 hands", "%+.2f u" % (mu * 100), "$%+.0f" % (mu * 100 * upd)),
        ("win rate / hour (%d hands)" % hph, "%+.2f u" % (mu * hph), "$%+.0f" % (mu * hph * upd)),
        ("average bet", "%.2f u" % st["avg_bet"], "$%.0f" % (st["avg_bet"] * upd)),
        ("edge per unit wagered", "%.3f%%" % (st["edge_bw"] * 100), ""),
        ("SD per hand", "%.2f u" % sigma, "$%.0f" % (sigma * upd)),
    ], ["metric", "value", "at $%g/unit" % upd])

    print("\nEfficiency (the figures that rank one game/spread against another):")
    A.print_table([
        ("N0  (hands until the win = 1 SD)", format(round(st["n0"]), ",")),
        ("DI  (desirability index, 1000*EV/SD)", "%.1f" % st["di"]),
        ("SCORE  ($/100 hands per $10k roll @ 13.5% RoR)", format(round(st["score"]), ",")),
    ], ["metric", "value"])

    print("\nBankroll your spread needs (fixed unit, lifetime risk of ruin):")
    rows = []
    for q in (0.005, 0.01, 0.05, 0.135, 0.25):
        b = bankroll.bankroll_for_ror(q, mu, sigma)
        rows.append(("%.1f%%" % (q * 100), format(round(b), ","), "$" + format(round(b * upd), ",")))
    A.print_table(rows, ["risk of ruin", "bankroll (units)", "bankroll ($)"])
    cur = bankroll.lifetime_ror(B0, mu, sigma)
    print("\nYour bankroll of %s units ($%s) -> lifetime risk of ruin %.1f%%."
          % (format(round(B0), ","), format(round(B0 * upd), ","), cur * 100))

    # finite-horizon Monte-Carlo: a sanity check on the formula + paths to plot
    sim = bankroll.simulate_fixed_unit(calib, config.spread_min, config.spread_max,
                                       config.ramp_start, config.spread_slope, wb, B0=B0,
                                       maxHands=config.bankroll_horizon, M=8000,
                                       seed=config.seed, record_traj=120)
    print("(%s-hand Monte-Carlo check: %.1f%% of trips busted within the horizon.)"
          % (format(config.bankroll_horizon, ","), sim["ror"] * 100))

    bundle = {"stake": st, "stake_curve": (mu, sigma, B0, (0.005, 0.01, 0.05, 0.135, 0.25), upd)}
    if (sim["traj"] is not None):
        bundle["trajectories"] = {"COUNT": [list(p) for p in sim["traj"]]}
    if (save_plots):
        fig = A.fig_required_bankroll(mu, sigma, B0, unit_dollars=upd)
        if (fig is not None):
            fig.savefig(os.path.join(outdir, config.label + "_required_bankroll.png"),
                        dpi=120, bbox_inches="tight")
    return bundle


def _run_ceiling(config, outdir, save_plots=True, cancel=None):
    import ca
    from deck import eor_tags, _EOR_BASE
    print("[ceiling] composition-exact playing ceiling, %d samples (%d decks, h17=%s)"
          % (config.ceiling_samples, config.numPacks, config.hitSoft17))
    print("\nEoR betting weights, scaled to Hi-Lo's RMS for comparison (Hi-Lo is a coarse"
          " rounding; Griffin's generic single-deck values shown for reference):")
    import math
    w = eor_tags(config.hitSoft17, config.surrender)
    grms = math.sqrt((4 * sum(_EOR_BASE[c] ** 2 for c in range(1, 10))
                      + 16 * _EOR_BASE[10] ** 2) / 52.0)
    gs = math.sqrt(40.0 / 52.0) / grms          # put Griffin on the same Hi-Lo RMS scale
    hilo = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}
    label = {1: "A", 10: "T"}
    A.print_table([(label.get(c, str(c)), "%+.2f" % w[c], "%+.2f" % (_EOR_BASE[c] * gs),
                    "%+d" % hilo[c]) for c in range(1, 11)],
                  ["card", "this game", "Griffin", "Hi-Lo"])

    r = ca.measure_playing_ceiling(n_samples=config.ceiling_samples, numPacks=config.numPacks,
                                   h17=config.hitSoft17, surrender=config.surrender,
                                   seed=config.seed, cancel=cancel)
    print("\nPlay (flat bet, all penetrations):")
    A.print_table([("perfect over basic", _fmt(r["opt_over_basic_pct"])),
                   ("Hi-Lo dev over basic", _fmt(r["hilo_over_basic_pct"])),
                   ("perfect over Hi-Lo", _fmt(r["opt_over_hilo_pct"]))],
                  ["comparison", "edge/hand"])

    print("\nPlay ceiling by penetration:")
    band_n = max(10000, config.ceiling_samples // 4)
    n_cards = config.numPacks * 52
    cut = int(n_cards * 0.75)
    edges = [0.0, 0.17, 0.43, 0.73, 1.0]
    prows = []
    for i in range(4):
        lo, hi = int(cut * edges[i]), int(cut * edges[i + 1])
        b = ca.measure_playing_ceiling(n_samples=band_n, numPacks=config.numPacks,
                                       h17=config.hitSoft17, surrender=config.surrender,
                                       seed=config.seed + 1, rem_lo=lo, rem_hi=hi, cancel=cancel)
        prows.append(("%d-%d" % (lo, hi), _fmt(b["opt_over_basic_pct"]),
                      "%.1f" % ((n_cards - (lo + hi) / 2) / 52.0)))
    A.print_table(prows, ["cards dealt", "perfect-basic", "decks left"])
    return {"ceiling": r}
