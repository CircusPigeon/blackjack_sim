"""Small correctness suite for the blackjack engine.

Reproducibility (fixed seeds in make_figures.py) guarantees the figures don't
*drift*; these tests guarantee the primitives they rest on are *right*. They lock
down the pieces that can't be "approximately correct" -- hand evaluation, dealer
play, basic strategy, the count deviations, and the balance of the counts -- plus
one end-to-end check that the assembled engine reproduces known edges.

Runs with no third-party dependencies:

    python tests/test_engine.py        # plain runner, prints PASS/FAIL, exits 1 on failure

It is also plain-pytest compatible (every check is a `test_*` function of asserts):

    pytest tests/test_engine.py -q
"""

import os
import sys
import io
import contextlib
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from play import Play
from player import Player
from dealer import Dealer
import strategy as S
from deck import _HILO_TAGS, EOR_WEIGHTS, COUNT_SYSTEMS, _FREQ

STAND, HIT, DOUBLE, SPLIT, SURRENDER = (Play.STAND.value, Play.HIT.value,
                                        Play.DOUBLE.value, Play.SPLIT.value,
                                        Play.SURRENDER.value)


def _player(cards, h17=True):
    """A Player holding one hand of `cards` (1 = ace, 10 = T/J/Q/K)."""
    p = Player()
    p.name = "P"
    p.hand = [list(cards)]
    p.handDone = [False]
    p.surrendered = [False]
    p.rules = {"hitSoft17": h17}
    return p


# --- hand evaluation -------------------------------------------------------

def test_hand_value_and_soft_aces():
    cases = [
        ([1, 6], 17, True),          # soft 17
        ([1, 6, 10], 17, False),     # ace forced to 1 -> hard 17
        ([1, 1], 12, True),          # one ace 11, one ace 1
        ([1, 1, 9], 21, True),       # 1+1+9 = 11, +10 = soft 21
        ([10, 10], 20, False),
        ([1, 10], 21, True),         # natural
        ([1, 5, 5], 21, True),       # 11, +10 -> soft 21
        ([10, 6], 16, False),
    ]
    for cards, total, soft in cases:
        p = _player(cards)
        assert p.getTotal(0) == total, (cards, p.getTotal(0), total)
        assert p.soft == soft, (cards, p.soft, soft)


def _is_natural(p):
    # how settle() identifies a natural: 21 on exactly two cards, single hand
    return len(p.hand) == 1 and len(p.hand[0]) == 2 and p.blackjack(0)

def test_blackjack_and_bust():
    assert _player([1, 10]).blackjack(0)                # blackjack() means "totals 21"
    assert _player([10, 6, 5]).blackjack(0)             # ...true on a three-card 21 too
    assert _is_natural(_player([1, 10]))                # a natural is 21 on two cards
    assert not _is_natural(_player([10, 6, 5]))         # a three-card 21 is NOT a natural
    p = _player([10, 6, 10])
    assert p.bust(0) and p.getTotal(0) == 26
    assert _player([10, 6]).isPair(0) is False
    assert _player([10, 10]).isPair(0)                  # any two tens are a pair (K,Q)


# --- dealer play (H17 vs S17) ----------------------------------------------

def _dealer_play(cards, h17):
    d = Dealer(hitSoft17=h17)
    d.hand = [list(cards)]
    return d.getPlay(0, 10, 0.0, False, False, False)   # 1 = hit, 0 = stand

def test_dealer_h17_s17():
    assert _dealer_play([1, 6], h17=True) == HIT         # soft 17: H17 hits
    assert _dealer_play([1, 6], h17=False) == STAND      # soft 17: S17 stands
    assert _dealer_play([10, 7], h17=True) == STAND      # hard 17 always stands
    assert _dealer_play([10, 7], h17=False) == STAND
    assert _dealer_play([10, 6], h17=True) == HIT        # 16 hits
    assert _dealer_play([10, 10], h17=True) == STAND     # 20 stands
    assert _dealer_play([1, 1], h17=True) == HIT         # soft 12 hits (below 17)


# --- basic strategy --------------------------------------------------------

def _basic(cards, up, d=True, sp=True, su=False):
    return S.basicPlay(_player(cards), 0, up, d, sp, su)

def test_basic_strategy_cells():
    assert _basic([10, 6], 10) == HIT                    # hard 16 v 10, no surrender
    assert _basic([10, 6], 10, su=True) == SURRENDER     # ...surrender when allowed
    assert _basic([10, 2], 4) == STAND                   # 12 v 4 stands
    assert _basic([10, 2], 3) == HIT                     # 12 v 3 hits
    assert _basic([10, 3], 2) == STAND                   # 13 v 2 stands
    assert _basic([5, 6], 10) == DOUBLE                  # 11 v 10 doubles (H17)
    assert _basic([5, 6], 10, d=False) == HIT            # ...but not when can't double
    assert _basic([1, 7], 9) == HIT                      # soft 18 v 9 hits
    assert _basic([1, 7], 6) == DOUBLE                   # soft 18 v 6 doubles
    assert _basic([1, 7], 7) == STAND                    # soft 18 v 7 stands
    assert _basic([10, 7], 10) == STAND                  # hard 17 stands

def test_pairs_and_surrender():
    assert _basic([8, 8], 10) == SPLIT                   # 8,8 always splits
    assert _basic([1, 1], 8) == SPLIT                    # A,A always splits
    assert _basic([10, 10], 6) == STAND                  # T,T never splits (stands on 20)
    assert _basic([5, 5], 6) == DOUBLE                   # 5,5 plays as hard 10, not a split
    assert _basic([9, 9], 7) == STAND                    # 9,9 v 7 stands (not a split cell)
    assert _basic([8, 8], 1, su=True) == SURRENDER       # 8,8 v A (H17): surrender, not split


# --- count deviations (Illustrious 18) -------------------------------------

def _count(cards, up, tc, d=False, sp=False, su=False):
    return S.countPlay(_player(cards), 0, up, tc, d, sp, su)

def test_count_deviations():
    assert _count([10, 6], 10, +1) == STAND              # 16 v 10 stands at TC >= 0
    assert _count([10, 6], 10, -1) == HIT                # ...reverts to basic below
    assert _count([10, 2], 3, +2) == STAND               # 12 v 3 stands at TC >= +2
    assert _count([10, 2], 3, +1) == HIT                 # ...basic hits below the index
    assert _count([10, 2], 2, +3) == STAND               # 12 v 2 stands at TC >= +3
    assert _count([10, 3], 2, -2) == HIT                 # 13 v 2 hits at TC < -1
    # COUNTX (engine-derived thresholds) is a separate, self-consistent table.
    assert S.countPlay(_player([10, 6]), 0, 10, +1, False, False, False, engine=True) == STAND


# --- the counts are balanced (depth-neutral) -------------------------------

def _freq_weighted_sum(tags):
    return sum(_FREQ[r] * tags[r] for r in range(1, 11))

def test_counts_are_balanced():
    assert abs(_freq_weighted_sum(_HILO_TAGS)) < 1e-9    # Hi-Lo sums to zero
    for name, tags in COUNT_SYSTEMS.items():
        assert abs(_freq_weighted_sum(tags)) < 1e-9, name      # level-2 systems
    for rules, w in EOR_WEIGHTS.items():
        assert abs(_freq_weighted_sum(w)) < 0.05, rules        # EoR weights (balanced floats)


# --- end-to-end: the assembled engine reproduces known edges ---------------

def test_engine_reproduces_known_edges():
    """A real seeded game must put basic strategy slightly negative, dealer-mimic
    deeply negative, and Hi-Lo counting ahead of basic. This exercises dealing,
    play, dealer logic, and settlement (3:2 blackjacks, busts, pushes) together.
    Catches gross payout/settle regressions a reproducible-but-wrong run would hide."""
    from config import Config
    import experiment
    log = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(log):
        b = experiment.run(Config(experiment="game", strategies=("BASIC", "DEALER", "COUNT"),
                                  rounds=60000, seed=7), outdir=tmp, save_plots=False) or {}
    edge = {s: e for (s, _w, _p, e) in b.get("summary", [])}   # e is edge in percent
    assert set(edge) >= {"BASIC", "DEALER", "COUNT"}, edge
    assert -1.5 < edge["BASIC"] < 0.0, edge["BASIC"]           # basic strategy: small house edge
    assert edge["DEALER"] < -3.0, edge["DEALER"]               # mimic-the-dealer bleeds ~5%
    assert edge["DEALER"] < edge["BASIC"] < edge["COUNT"]      # counting beats basic beats mimic


# --- runner (no pytest required) -------------------------------------------

def _main():
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("  PASS  " + name)
        except Exception as e:
            failed.append(name)
            print("  FAIL  " + name + "  ->  " + repr(e))
    print("\n%d passed, %d failed" % (passed, len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
