# Blackjack Lab

A from-scratch blackjack engine and advantage-play research lab. One config-driven
runner measures basic strategy, card counting, shuffle tracking, the casino
detection ("heat") game, bankroll risk-of-ruin, and the composition-exact
theoretical ceiling.

## Key results (TL;DR)

- **Basic strategy loses (≈ −0.5%); Hi-Lo counting flips it positive.** A continuous
  shuffler zeroes the count, and 6:5 payouts (≈ −1.3%) sink the game past what
  counting can recover.
- **Hi-Lo already sits at the betting ceiling.** Its betting correlation is 0.96 vs
  1.00 for the effect-of-removal–optimal weights, and the edge-by-count curves
  overlap — because the betting edge really *is* linear in shoe composition: a
  flexible nonlinear predictor adds no out-of-sample power over the best linear count.
- **…but Hi-Lo is nowhere near the playing ceiling.** Composition-perfect play beats
  basic by ≈ +0.11%/hand off the top, rising to ≈ +0.52% deep in the shoe; the full
  Illustrious 18 captures only ≈ 1/7 of that. *Betting is linear and a simple count
  nails it; playing is nonlinear and no linear count comes close.*
- **Profit hides in rare hands.** For a spreading counter, hands at true count ≥ +3
  are < 8% of all hands but supply ~170% of net profit — and those few big bets are
  exactly what the pit watches.
- **The whole stack vs a real casino:** Hi-Lo + the full I18 + wonging + a modest 1–8
  spread off a 300-unit roll, at a 6-deck table with a real casino shuffle and a live
  pit → ≈ 61% grind out a profit, ≈ 21% get barred, ≈ 17% go broke; median and mean
  career both finish ≈ 324 units (+8%). The edge is real on average, but a sixth still
  bust and a fifth get barred.

Full numbers, methods, and 95% confidence intervals in [Selected findings](#selected-findings) below.

## Run it yourself

Python 3.9+ on Windows, macOS, or Linux.

```
git clone <this-repo>
cd Blackjack_Sim
pip install -r requirements.txt      # numpy + matplotlib
python gui.py
```

`tkinter` (the GUI toolkit) ships with CPython; on Debian/Ubuntu install it with
`sudo apt install python3-tk`. `pandas` is optional (CSV/DataFrame export only).

A point-and-click front-end — dropdowns for the config, checkboxes for the plots,
live progress, and a "Save plots" toggle:

```
python gui.py
```

Or the CLI:

```
python main.py game
python main.py heat
python main.py bankroll
python main.py ceiling
```

Every flag is a `Config` field (see `config.py`). Examples:

```
python main.py game strategies=BASIC,COUNT,TRACK shuffle=casino dummyPlayers=4
python main.py game strategies=BASIC,COUNT,ORACLE     # Hi-Lo vs EoR betting
python main.py game shuffle=csm                       # continuous shuffler kills the count
python main.py game blackjackPays=1.2                 # 6:5 instead of 3:2
python main.py ceiling ceiling_samples=40000
```

Tables print to stdout; CSV/JSON records and PNG plots land in `results/`.

## Experiments (`experiment=`)

| kind | measures |
|---|---|
| `game` | per-strategy edge + edge-by-true-count over `rounds` dealt hands |
| `heat` | total profit vs ramp aggressiveness under casino back-off |
| `bankroll` | risk of ruin / growth / drawdown vs Kelly fraction |
| `ceiling` | composition-exact perfect play vs basic / Hi-Lo, by penetration (respects H17/S17, surrender, deck count; no splits) |

## Strategies (sit any mix at one table, same shoes)

- `BASIC`, `DEALER` (mimic), `RANDOM`, `STAND` — baselines
- `COUNT` — Hi-Lo: bet spread + the full Illustrious-18 deviations
- `COUNT0` — Hi-Lo bet spread, **no** deviations (COUNT − COUNT0 = what index plays are worth)
- `COUNTX` — Hi-Lo bet spread + **engine-derived** index thresholds (`precompute_indices.py`)
- `WONG` — Hi-Lo, but back-counts: sits out hands below `wong_below` true count (bets 0,
  takes no cards), plays the rest like COUNT
- `ORACLE` — effect-of-removal (best *linear*) betting weights
- `HIOPT2`, `ZEN`, `OMEGA2` — level-2 counts: finer tags trade a little betting
  correlation for better playing decisions (bet *and* deviate on their own count)
- `TRACK` — shuffle tracker: predicts high-card slugs from the pre-shuffle pile

## What the algorithms do

**COUNT — Hi-Lo card counter (the practical human strategy).** Running count:
**+1** per 2–6, **0** per 7–9, **−1** per 10/J/Q/K/A; divide by the decks remaining
to get the **true count (TC)**, an estimate of how rich the unseen cards are. One
number drives both (a) **bet sizing** — a linear ramp on TC (flat minimum below
`ramp_start`, then `spread_slope` extra units per +1 TC, capped at `spread_max`)
— and (b) **play** — basic strategy plus the hard-total Hi-Lo "Illustrious-18"
index deviations (override basic when the TC crosses a cell's threshold; e.g.
stand 16 vs 10 at TC ≥ 0). Insurance is taken at TC ≥ +3 and the two pair-split
plays (10,10 v 5 at +5, v 6 at +4) are included, so all 18 plays are modeled.
Two variants bracket it: `COUNT0` (same bet spread, no deviations — the paired
difference is what the index plays are worth) and `COUNTX` (same cells, but with
index thresholds re-derived from this engine for the exact ruleset by
`precompute_indices.py`, keyed by `hitSoft17` — the textbook numbers date to older
S17 games; the engine finds e.g. 10 v A at +3 under H17, not the book's +4).

**WONG — Hi-Lo with back-counting.** Identical to COUNT on the hands it plays, but
it sits out entirely (bets 0, takes no cards) while the true count is below
`wong_below`, only stepping in on the favorable shoes. Skipping the bad counts
raises the edge on the hands actually played; the cost is throughput. Most of the
gain comes from wonging out just the clearly negative counts.

**ORACLE — the best *linear* betting count for our exact game.** Plays
*identically* to COUNT (same deviations) — deliberately, to isolate one question:
*can smarter bet-sizing weights beat Hi-Lo's?* The only difference is the count it
bets on: an **effect-of-removal (EoR)** count whose per-rank tags are proportional
to how much removing one card of each rank shifts the player's edge in *this* rule
set (H17, no surrender — the default game; the weights are keyed by ruleset). Those
weights are derived from the engine by regressing
realized flat-bet edge on the remaining-shoe composition (`precompute_eor.py`),
then balanced and scaled to the Hi-Lo magnitude. ORACLE marks the betting ceiling
among linear counts — and the finding is that Hi-Lo already reaches it.

**HIOPT2 / ZEN / OMEGA2 — level-2 counting systems.** Finer per-card tags (to ±2)
that read the *playing* value of the deck better than Hi-Lo at the cost of a
little betting correlation (BC ≈ 0.91–0.96 vs Hi-Lo's 0.96; playing efficiency
≈ 0.74–0.78 vs 0.63). Each bets **and** makes its index deviations on its *own*
true count. The engine tracks every system's running count with tags rescaled to
Hi-Lo's per-card RMS, so one bet ramp and the same deviation thresholds apply to
all counts on a shared true-count scale (system-specific index tables are not
modeled). The trade pays where composition swings hardest: in our paired runs
they beat Hi-Lo by ≈ +0.3–0.4%/hand in single deck and tie it in a 6-deck shoe.

**TRACK — shuffle tracker.** Exploits the *sequence* correlation a weak casino
shuffle leaves behind (which counting ignores). It learns the shuffle by
simulating the dealer's exact procedure ~4000× into a **position-transition
matrix** `T[i,d]` = P(a card at pre-shuffle pile position *i* is dealt at step
*d*); one matrix–vector product then forecasts the expected Hi-Lo richness at each
upcoming deal position. It bets the ramp on that *forecast* (not the running
count), betting up just before a high-card slug arrives, and plays basic strategy.
Idealized full-knowledge tracker (sees the whole pre-shuffle pile) → an *upper
bound* on tracking; the cut card is the casino's main defense.

**CEILING — composition-exact perfect play (an analysis, not a player; `ca.py`).**
For thousands of sampled (remaining composition, your two cards, dealer upcard)
states it computes, by dynamic programming, the dealer's exact final-total
distribution and the exact EV of stand / hit / double / **late surrender** under
the *actual* composition, then takes the max-EV action. It reports perfect-play
edge − basic-strategy edge, averaged by penetration — the **upper bound on playing
edge**. Respects **H17/S17, late surrender, and deck count**; pairs are played by
total (**splits not modeled**), so it is the *no-split* playing ceiling.

## Rules & terms

| term | meaning |
|---|---|
| **H17 / S17** | dealer hits / stands on soft 17 (an ace as 11, e.g. A-6). H17 is ~0.2% worse for the player. `hitSoft17` |
| **3:2 vs 6:5** | blackjack payout: 3:2 pays 1.5×; 6:5 pays 1.2× and costs the player ≈ 1.3%. `blackjackPays` |
| **late surrender** | forfeit half the bet and fold after seeing your two cards and the dealer upcard (first decision only, after the dealer checks for blackjack). Off by default — most casinos don't offer it. `surrender` |
| **penetration** | fraction of the shoe dealt before the cut card forces a reshuffle (0.75 = 75%); deeper helps a counter. `penetration` |
| **decks** | number of 52-card packs in the shoe (Vegas: 6 or 8). `numPacks` |
| **true count (TC)** | running count ÷ decks remaining — normalizes the count for shoe depth |
| **effect of removal (EoR)** | how much removing one card of a given rank shifts the player's edge; the "true" per-card weights an optimal betting count targets (`precompute_eor.py`) |
| **betting correlation (BC)** | how well a count's per-card tags track the EoR weights (1.0 = perfect). Hi-Lo ≈ 0.96; ORACLE = 1.0 by construction |
| **playing efficiency (PE)** | how much of the composition-perfect playing gain a count captures via its index plays. Hi-Lo ≈ 0.63; level-2 counts (Hi-Opt II / Zen / Omega II) ≈ 0.74–0.78 |
| **bet spread / ramp** | how a counter scales bets with TC: `spread_min`–`spread_max` units, climbing `spread_slope` units per +1 TC from `ramp_start` |
| **wonging / back-counting** | sitting out hands while the count is unfavorable, playing only the good shoes (the `WONG` strategy). `wong_below` |
| **index play / deviation** | a count-triggered exception to basic strategy; the *index* is the true count at which the alternative action becomes correct (the Illustrious 18 are the 18 most valuable) |
| **casino shuffle** | a realistic hand shuffle — N GSR riffles + strips + a final cut (`shuffleRiffles/Strips/Cut`); not fully randomizing, hence trackable |
| **CSM** | continuous shuffle machine — dealt cards return to the shoe each hand, so no count ever builds (`shuffle=csm`) |
| **dummy players** | untracked bystanders who consume cards; they don't change your per-hand edge, only hands/hour |
| **Kelly fraction** | bet size as a fraction of the growth-optimal (full-Kelly) bet; < 1 trades growth for lower risk of ruin. `kelly_fraction` |
| **heat / back-off** | the pit estimating your bet-vs-count slope and barring you once it looks like counting. `heat_*` |

## Module map

```
play.py        Play enum
deck.py        the shoe: dealing, Hi-Lo + EoR counts, shuffle hook, carryover
player/dealer/guest.py   actors (hands, money, per-strategy decisions)
strategy.py    basic strategy tables + Hi-Lo deviations + surrender
shuffle.py     shuffle procedures: GSR riffle / strip / cut, CSM
tracker.py     shuffle-tracking position-transition matrix
blackjack.py   the round engine (deal, play, settle, record)
config.py      the Config dataclass (one object = one reproducible run)
experiment.py  run(config) dispatcher + run_experiment primitive
analysis.py    record aggregation, tables, plots, CSV/JSON export
bankroll.py    counter calibration + fractional-Kelly risk-of-ruin Monte Carlo
heat.py        the detection / back-off game
ca.py          combinatorial-analysis playing ceiling
main.py        CLI
precompute_eor.py  derive the EoR betting weights per ruleset + the nonlinear betting-ceiling check
precompute_indices.py  derive the index-play thresholds (Illustrious 18) for this exact game
make_figures.py  regenerate the curated write-up figures (SVG/PNG) + manifest
                 (incl. counting_systems BC/PE scatter and the edge_crossover figure)
```

## Figures for the write-up

`make_figures.py` regenerates the curated figure set from pinned, reproducible
Configs and writes them (SVG + PNG) plus a `figures/manifest.json` of the headline
numbers — so a write-up can quote one source of truth and stay in sync with the
images. Every edge estimate is Monte-Carlo: each condition runs over many
independent trials and is shown as a mean with a **95% confidence interval**
(paired comparisons like TRACK vs BASIC are differenced per trial to cancel
shoe-to-shoe noise).

```
python make_figures.py --list                 # list the figure specs
python make_figures.py                         # regenerate everything (minutes)
python make_figures.py --only heat,penetration
python make_figures.py --trials 40 --rounds 60000   # higher fidelity, slower
python make_figures.py --smoke                 # tiny sizes, just to test plumbing
```

## What composes

The `game` engine is the ground truth: rules, shuffle, dummy players, any mix of
strategies, and the live **heat** (`heat_live=true`, casino back-off) and
**bankroll** (`bankroll_live=true`, fractional Kelly + ruin) layers all stack in
one run on shared shoes:

```
python main.py game strategies=BASIC,COUNT,TRACK shuffle=casino \
    dummyPlayers=4 heat_live=true bankroll_live=true
```

That single run plays a realistic session — a counter and a shuffle tracker bet
fractional Kelly off finite bankrolls at a full table against a real casino
shuffle, while the pit watches their bet/count slope and backs them off. (An
emergent result: the tracker survives longer than the counter, because its bets
track slugs rather than the visible count, so it's harder to spot.)

Add `trials=N` to Monte-Carlo that composed session over N seeds and get the
outcome *distribution* (ruin/bar rates, survival-hand percentiles, a histogram)
instead of one realization — each session ends early when the heroes leave, so a
few hundred to a few thousand trials runs in minutes:

```
python main.py game strategies=BASIC,COUNT,TRACK shuffle=casino \
    dummyPlayers=4 heat_live=true bankroll_live=true rounds=5000 trials=500
```

(`rounds` is the per-session horizon; cap it at a realistic session length to get
"P(barred within one session)". A literal million trials is the calibration-based
`heat`/`bankroll` experiments' job — those resample fast but aren't composed.)

The `heat` / `bankroll` / `ceiling` **experiments** remain as fast batch tools
(Monte-Carlo sweeps over a cached calibration, or pure combinatorics) for
*distributions* rather than single sessions. Only composition-exact **perfect
play** stays outside the live engine (~ms/decision); it lives in `ceiling`.

## Selected findings

- Basic strategy ≈ −0.5%; Hi-Lo counting flips it positive.
- 6:5 payouts cost ≈ 1.3% and counting can't overcome them; a CSM zeroes the count.
- A 2–3 riffle casino shuffle stays trackable; the tracker beats weak shuffles
  (≈ +1.0–1.8%/hand, more the sloppier the shuffle), and the final cut sharply
  reduces it (the casino's main defense).
- More players don't change per-hand EV (the "card-eater" myth) — they cut hands/hour.
- Full Kelly ≈ 46% chance of losing half your roll; half-Kelly (~14%) is the sweet spot.
- **Hi-Lo sits at the betting ceiling** (betting correlation 0.96 vs 1.00 for the
  EoR-optimal weights, and the two edge-by-count curves overlap). The ceiling is not
  merely the best *linear* one: fitting a flexible nonlinear predictor of edge from
  the exact remaining composition adds **no** out-of-sample power over the best
  linear count, because the betting edge really is linear in composition.
- **But Hi-Lo is far below the playing ceiling.** Composition-perfect play beats
  basic by ≈ +0.11%/hand off the top, rising to ≈ +0.52% deep in the shoe (H17, no
  surrender), while the full Illustrious-18 index plays capture almost none of that
  (≈ 0.08%/hand at best, ~1/7 of the perfect-play gain). Completing the I18 (13v3,
  exact 12v4/5/6) barely moves it: at 6 decks the count rarely reaches the thresholds,
  and a Hi-Lo deviation can only ever capture the count-*correlated* slice of the
  composition-perfect playing edge. Betting is linear and a simple count nails it;
  playing is nonlinear and no linear count comes close.
- **Level-2/3 counts trade betting correlation for playing efficiency.** Hi-Opt II,
  Zen, and Omega II have slightly lower betting correlation than Hi-Lo (≈ 0.91–0.96)
  but markedly higher playing efficiency (≈ 0.74–0.78 vs Hi-Lo's 0.63). Zen is the
  balanced all-rounder; the extra playing efficiency pays off most in single- and
  double-deck games, where the composition swings are largest. Engine-validated:
  as playable strategies (`HIOPT2`/`ZEN`/`OMEGA2`), all three beat Hi-Lo by
  ≈ +0.3–0.4%/hand in paired single-deck runs and tie it at 6 decks
  (`make_figures.py --only counting_systems` for the BC/PE scatter).
- **Best of both worlds: optimal betting + perfect play.** Combining the ORACLE bet
  spread with composition-perfect play yields ≈ +3.2%/hand in single deck, falling to
  ≈ +1.2% by eight decks (1–12 spread, 75% pen). The playing gain such a player captures
  is ≈ 1.3× the *flat-bet* playing ceiling, because it bets biggest in exactly the skewed
  shoes where perfect play deviates most. Betting is still ≈ 70–80% of the total edge in
  every game (`make_figures.py --only edge_crossover`).
- **Index plays are worth having, not worth perfecting.** The full Illustrious 18 is
  worth ≈ +1%/hand in single deck and ≈ +0.4% in a 6-deck shoe over the same counter
  with no deviations; re-deriving the thresholds for the exact game (`COUNTX`) edges out
  the textbook numbers by an amount that stays inside the error bars — the actions are
  near-tied in EV at their index, so a count off costs almost nothing.
- **Profit is concentrated in rare hands.** For a spreading Hi-Lo counter, hands at TC ≥ +3
  are < 8% of all hands but supply ~170% of the net profit; the majority, played near the
  minimum bet, lose slowly. Those few big bets are exactly what the pit watches.
- **Back-counting (wonging) helps, but mildly at a shoe game.** Sitting out the clearly
  negative counts roughly doubles the edge per hand played; sitting out too aggressively
  costs throughput for little extra edge.
- **The whole stack vs a real casino.** A player using everything (Hi-Lo + full I18 +
  wonging + a modest 1–8 spread off a 300-unit roll) at a 6-deck table with a casino
  shuffle and a live pit: ≈ 61% still grinding a profit, ≈ 21% barred (and leaving ~29
  units ahead), ≈ 17% broke. Median and mean career both finish ≈ 324 units, a +24 (8%)
  profit (`make_figures.py --only practical_player`). The edge is real and shows through
  on average — but a sixth still bust and a fifth get barred, so it stays a hard, swingy
  way to earn.

## Caveats

The CA ceiling plays pairs by total (**no splits**) and approximates based on the
deck composition going into a hand. It respects H17/S17, late surrender, and deck
count. COUNT plays the full Illustrious 18 (textbook thresholds; the engine-derived
alternative is COUNTX). The level-2 counters (HIOPT2/ZEN/OMEGA2) reuse the Hi-Lo deviation
cells and thresholds on their own RMS-rescaled count rather than system-specific
index tables. ORACLE deliberately plays like COUNT (it isolates the *betting*
count, not play). Risk sims resample i.i.d. hands (no within-shoe serial
correlation).
