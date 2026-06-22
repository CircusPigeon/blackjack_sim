"""Bankroll / risk engine.

Card counting tells you your edge; the bankroll engine tells you whether you
survive the variance. We calibrate the counter's per-hand outcome distribution
once (result per unit bet, as a function of true count), then Monte-Carlo many
independent "trips" under fractional-Kelly bet sizing to measure risk of ruin,
probability of reaching a goal, growth, and drawdown.

Key fact that makes this exact and fast: the per-unit outcome (result / bet) does
not depend on how much was wagered -- doubling the bet doubles the result -- so
we can resample calibrated per-unit outcomes and scale each by whatever Kelly bet
the bankroll calls for, without re-dealing cards.

Bets are sized as a fraction of the *current* bankroll (resizing Kelly):
    bet = kellyFrac * edge(trueCount) / variance * bankroll,
clamped to the table limits."""

import os
import numpy as np

EDGE_LO = -6
EDGE_HI = 12


def calib_path(config=None):
    """Where a COUNT calibration is cached. With a config, the path is keyed by the
    game parameters that change the per-count outcome (decks, penetration, rules,
    shuffle), so heat/bankroll reflect the chosen table instead of a fixed default."""
    if (config is None):
        return "results/calib.npz"
    es = "e" if getattr(config, "earlySurrender", False) else ""
    das = int(getattr(config, "doubleAfterSplit", True))
    return os.path.join("results", "calib_%dd_p%s_h%d_s%d%s_das%d_bj%s_%s.npz" % (
        config.numPacks, ("%g" % config.penetration), int(config.hitSoft17),
        int(config.surrender), es, das, ("%g" % config.blackjackPays), config.shuffle))


def load_or_make_calibration(path=None, rounds=1000000, config=None, cancel=None, progress=None):
    """Load the cached COUNT calibration, or run it once and cache it. With a
    config, the calibration matches that game (decks, penetration, rules, shuffle);
    without one, it uses the default game (what make_figures relies on). cancel /
    progress let a long first-time calibration be aborted and reported."""
    from config import Config
    from experiment import run_experiment
    if (path is None):
        path = calib_path(config)
    if (os.path.exists(path)):
        d = np.load(path)
        return {"true_count": d["tc"], "bet": d["bet"], "result": d["result"]}
    if (config is not None):
        cal_cfg = Config(rounds=rounds, strategies=("COUNT",), shuffle=config.shuffle,
                         numPacks=config.numPacks, penetration=config.penetration,
                         hitSoft17=config.hitSoft17, surrender=config.surrender,
                         blackjackPays=config.blackjackPays)
    else:
        cal_cfg = Config(rounds=rounds, strategies=("COUNT",), shuffle="random")
    g = run_experiment(cal_cfg, record=True, cancel=cancel, progress=progress)
    rec = g.records
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, tc=rec["true_count"], bet=rec["bet"], result=rec["result"])
    return rec


def calibrate(records):
    """Build the per-unit outcome model from a recorded COUNT run."""
    tc = np.asarray(records["true_count"], dtype=float)
    bet = np.asarray(records["bet"], dtype=float)
    result = np.asarray(records["result"], dtype=float)
    rpu = result / bet                                  # per-unit outcome
    buckets = np.clip(np.floor(tc).astype(int), EDGE_LO, EDGE_HI)
    edge_arr = np.zeros(EDGE_HI - EDGE_LO + 1)
    for b in range(EDGE_LO, EDGE_HI + 1):
        m = buckets == b
        if (m.any()):
            edge_arr[b - EDGE_LO] = rpu[m].mean()
    ev_hand = float(result.mean())                      # avg units won per hand
    sd_hand = float(result.std())                       # SD of units per hand
    edge_bw = float(result.sum() / bet.sum())           # edge per unit wagered (the real edge)
    return {
        "tc": tc,
        "rpu": rpu,
        "edge_arr": edge_arr,
        "var_unit": float(rpu.var()),
        "edge_bw": edge_bw,
        "sd_unit": float(rpu.std()),                    # SD per unit bet
        "n0": (sd_hand / ev_hand) ** 2 if ev_hand > 0 else float("inf"),
    }


def _edge_of(tc, edge_arr):
    b = np.clip(np.floor(tc).astype(int), EDGE_LO, EDGE_HI) - EDGE_LO
    return edge_arr[b]


def simulate_trips(calib, kellyFrac=0.5, B0=2000.0, table_min=1.0, table_max=200.0,
                   ruin_frac=0.5, goal_mult=2.0, maxHands=30000, M=5000, seed=0,
                   wong=False, record_traj=0):
    """Run M independent trips. A trip ends on ruin (bankroll <= ruin_frac*B0),
    on reaching the goal (goal_mult*B0), or after maxHands."""
    rng = np.random.default_rng(seed)
    tc_s = calib["tc"]
    rpu_s = calib["rpu"]
    edge_arr = calib["edge_arr"]
    var_unit = calib["var_unit"]
    n = len(rpu_s)

    bankroll = np.full(M, float(B0))
    alive = np.ones(M, dtype=bool)
    ruined = np.zeros(M, dtype=bool)
    reached = np.zeros(M, dtype=bool)
    peak = np.full(M, float(B0))
    maxdd = np.zeros(M)
    ruin_level = max(ruin_frac * B0, table_min)
    goal = goal_mult * B0 if goal_mult else None
    floor_bet = 0.0 if wong else table_min

    traj = None
    if (record_traj):
        traj = np.empty((record_traj, maxHands + 1))
        traj[:, 0] = B0

    steps = 0
    for t in range(maxHands):
        if (not alive.any()):
            break
        steps = t + 1
        idx = rng.integers(0, n, M)
        tc = tc_s[idx]
        rpu = rpu_s[idx]
        e = _edge_of(tc, edge_arr)
        bet = kellyFrac * (e / var_unit) * bankroll
        bet = np.where(e > 0, bet, floor_bet)
        bet = np.clip(bet, floor_bet, table_max)
        bet = np.minimum(bet, bankroll)
        bankroll = bankroll + np.where(alive, bet * rpu, 0.0)
        peak = np.maximum(peak, bankroll)
        dd = np.where(peak > 0, (peak - bankroll) / peak, 0.0)
        maxdd = np.maximum(maxdd, np.where(alive, dd, 0.0))
        nr = alive & (bankroll <= ruin_level)
        ruined |= nr
        alive = alive & ~nr
        if (goal is not None):
            ng = alive & (bankroll >= goal)
            reached |= ng
            alive = alive & ~ng
        if (traj is not None):
            traj[:, t + 1] = bankroll[:record_traj]

    if (traj is not None):
        traj = traj[:, :steps + 1]
    return {
        "ror": float(ruined.mean()),
        "p_goal": float(reached.mean()),
        "median_growth": float(np.median(bankroll) / B0),
        "mean_growth": float(bankroll.mean() / B0),
        "median_maxdd": float(np.median(maxdd)),
        "final": bankroll,
        "traj": traj,
    }


def risk_curve(calib, fractions, **kw):
    """Sweep Kelly fraction; return (frac, ror, median_growth, p_goal, median_maxdd)."""
    rows = []
    for f in fractions:
        r = simulate_trips(calib, kellyFrac=f, **kw)
        rows.append((f, r["ror"], r["median_growth"], r["p_goal"], r["median_maxdd"]))
    return rows


# --- fixed-unit spread analysis -------------------------------------------------
# The Kelly engine above answers "given continuous resizing, how does ruin trade
# off against growth?". A real counter instead bets a *fixed unit* on a discrete
# spread ramp and wants the operational numbers: win rate, N0/SCORE, and -- above
# all -- how big a bankroll the spread needs. Those follow in closed form from the
# mean and variance of the per-hand result under that ramp, which we read straight
# from the same per-unit calibration (the per-unit outcome is bet-independent).

def bet_schedule(tc, spread_min, spread_max, ramp_start, slope, wong_below=None):
    """Units wagered at each true count under a discrete spread ramp -- the SAME
    ramp Game and Heat use (guest.calculateBet), so the three experiments reconcile.
    Below ramp_start you bet the spread floor; above it the bet grows `slope` units
    per +1 true count, rounded and capped at spread_max. wong_below (if set) sits the
    hand out (bets 0) while the count is below it -- back-counting."""
    tc = np.asarray(tc, dtype=float)
    ramped = np.clip(np.round(spread_min + slope * (tc - ramp_start)), spread_min, spread_max)
    units = np.where(tc < ramp_start, float(spread_min), ramped)
    if (wong_below is not None):
        units = np.where(tc < wong_below, 0.0, units)
    return units


def spread_stats(calib, spread_min, spread_max, ramp_start, slope, wong_below=None):
    """Per-hand outcome of a fixed-unit discrete spread, from the calibrated per-unit
    outcomes. Each recorded hand carries its true count and its per-unit result; we
    re-bet it with our own (bankroll-independent) spread and measure the distribution
    of units won per hand. Returns the mean/SD plus the standard advantage-play
    figures derived from them (N0, DI, SCORE)."""
    tc = np.asarray(calib["tc"], dtype=float)
    rpu = np.asarray(calib["rpu"], dtype=float)
    bet = bet_schedule(tc, spread_min, spread_max, ramp_start, slope, wong_below)
    won = bet * rpu                                     # units won this hand
    mu = float(won.mean())                              # EV per hand, units
    sigma = float(won.std())                            # SD per hand, units
    avg_bet = float(bet.mean())
    n0 = (sigma / mu) ** 2 if mu > 0 else float("inf")  # hands until the win equals 1 SD
    di = 1000.0 * mu / sigma if sigma > 0 else 0.0      # desirability index
    return {"mu": mu, "sigma": sigma, "avg_bet": avg_bet,
            "edge_bw": (mu / avg_bet if avg_bet > 0 else 0.0),
            "played_frac": float((bet > 0).mean()),
            "n0": n0, "di": di, "score": di * di}       # SCORE = DI^2 = 1e6 / N0


def lifetime_ror(bankroll_units, mu, sigma):
    """Classic fixed-bet lifetime risk of ruin -- play forever at a fixed unit,
    never resizing: RoR = exp(-2 * mu * B / sigma^2) (the diffusion approximation,
    mu and sigma in units per hand, B in units)."""
    if (mu <= 0 or sigma <= 0):
        return 1.0
    return float(np.exp(-2.0 * mu * bankroll_units / (sigma * sigma)))


def bankroll_for_ror(q, mu, sigma):
    """Invert lifetime_ror: the bankroll (units) needed to hold lifetime RoR to q."""
    import math
    if (mu <= 0 or sigma <= 0 or q <= 0 or q >= 1):
        return float("inf")
    return math.log(1.0 / q) * (sigma * sigma) / (2.0 * mu)


def simulate_fixed_unit(calib, spread_min, spread_max, ramp_start, slope, wong_below=None,
                        B0=2000.0, maxHands=20000, M=8000, seed=0, record_traj=0):
    """Monte-Carlo fixed-unit trips: resample real hands (true count + per-unit
    outcome), bet the spread at a fixed unit, ruin at <= 0. Validates the analytic
    lifetime RoR over a finite horizon and yields a few trajectories to plot."""
    rng = np.random.default_rng(seed)
    tc = np.asarray(calib["tc"], dtype=float)
    rpu = np.asarray(calib["rpu"], dtype=float)
    won = bet_schedule(tc, spread_min, spread_max, ramp_start, slope, wong_below) * rpu
    n = len(won)
    bankroll = np.full(M, float(B0))
    alive = np.ones(M, dtype=bool)
    ruined = np.zeros(M, dtype=bool)
    traj = None
    if (record_traj):
        traj = np.empty((record_traj, maxHands + 1))
        traj[:, 0] = B0
    steps = 0
    for t in range(maxHands):
        if (not alive.any()):
            break
        steps = t + 1
        idx = rng.integers(0, n, M)
        bankroll = bankroll + np.where(alive, won[idx], 0.0)
        nr = alive & (bankroll <= 0.0)
        ruined |= nr
        alive = alive & ~nr
        if (traj is not None):
            traj[:, t + 1] = np.maximum(bankroll[:record_traj], 0.0)
    if (traj is not None):
        traj = traj[:, :steps + 1]
    return {"ror": float(ruined.mean()), "median_final": float(np.median(bankroll)),
            "traj": traj, "hands": steps}
