"""Basic strategy (6-deck, double-after-split, late surrender) plus the hard-total
and doubling members of the Hi-Lo "Illustrious 18" count-based deviations.

Upcards are 1-10, where 1 is an Ace and 10 covers T/J/Q/K. Functions take the
acting player and a hand index and return a Play value (int). Whether the dealer
hits soft 17 is read from player.rules (it shifts a few surrender cells)."""

from play import Play

STAND = Play.STAND.value
HIT = Play.HIT.value
DOUBLE = Play.DOUBLE.value
SPLIT = Play.SPLIT.value
SURRENDER = Play.SURRENDER.value


def _hitSoft17(player):
    rules = getattr(player, "rules", None)
    if (rules):
        return rules.get("hitSoft17", True)
    return True


def _das(player):
    rules = getattr(player, "rules", None)
    return rules.get("doubleAfterSplit", True) if rules else True


def _early(player):
    rules = getattr(player, "rules", None)
    return bool(rules.get("earlySurrender")) if rules else False


def basicPlay(player, i, upcard, canDouble, canSplit, canSurrender):
    total = player.getTotal(i)
    soft = player.soft
    if (canSurrender and not soft and _surrender(player, i, upcard, _hitSoft17(player))):
        return SURRENDER
    if (canSplit and player.isPair(i) and _pairSplit(player.hand[i][0], upcard, _das(player))):
        return SPLIT
    if (soft):
        return _softPlay(total, upcard, canDouble)
    return _hardPlay(total, upcard, canDouble)


def countPlay(player, i, upcard, trueCount, canDouble, canSplit, canSurrender,
              engine=False):
    total = player.getTotal(i)
    soft = player.soft
    # Index plays apply to hard totals; pairs get the two 10,10 split plays. A
    # None result means the count does not trigger a deviation, so defer to
    # basic strategy (which may itself surrender). engine=True plays the
    # engine-derived thresholds (COUNTX) instead of the textbook ones.
    dev = None
    if (not soft):
        h17 = _hitSoft17(player)
        if (player.isPair(i)):
            if (canSplit):
                dev = _pairDeviation(player.hand[i][0], upcard, trueCount, engine, h17)
        else:
            if (engine):
                dev = _deviationEngine(total, upcard, trueCount, canDouble, h17)
            else:
                dev = _deviation(total, upcard, trueCount, canDouble)
    if (dev is not None):
        return dev
    return basicPlay(player, i, upcard, canDouble, canSplit, canSurrender)


def _surrender(player, i, up, hitSoft17):
    # Evaluated only on a fresh two-card hard hand.
    if (_early(player)):
        return _early_surrender(player, i, up)
    # Late surrender, H17 chart.
    if (player.isPair(i) and player.hand[i][0] == 8):
        return up == 1 and hitSoft17          # 8,8 vs A (H17): surrender, don't split
    if (player.isPair(i)):
        return False
    total = player.getTotal(i)
    if (total == 16):
        return up in (9, 10, 1)
    if (total == 15):
        return up == 10 or (hitSoft17 and up == 1)
    if (total == 17):
        return hitSoft17 and up == 1          # 17 vs A (H17)
    return False


def _early_surrender(player, i, up):
    """Early surrender (decided before the dealer peeks for blackjack), H17 chart.
    The wide vs-ace / vs-ten plays -- giving up even when the dealer may have a
    natural -- are the gain over late surrender."""
    total = player.getTotal(i)
    pair = player.hand[i][0] if player.isPair(i) else 0
    if (up == 1):                              # vs Ace
        if (pair in (3, 6, 7, 8)):
            return True                        # 3,3 / 6,6 / 7,7 / 8,8
        return total in (5, 6, 7, 12, 13, 14, 15, 16, 17)
    if (up == 10):                             # vs Ten
        if (pair in (7, 8)):
            return True                        # 7,7 / 8,8
        return total in (14, 15, 16)
    if (up == 9):
        return total == 16                     # same as late
    return False


def _pairSplit(card, up, das=True):
    # Without double-after-split (das=False), a few pairs split against fewer
    # upcards (you can't recoup by doubling the split hand): 2,2/3,3 drop 2-3,
    # 4,4 never splits, 6,6 drops 2.
    if (card == 1):          # A,A
        return True
    if (card == 10):         # T,T
        return False
    if (card == 9):
        return up in (2, 3, 4, 5, 6, 8, 9)
    if (card == 8):
        return True
    if (card == 7):
        return up in (2, 3, 4, 5, 6, 7)
    if (card == 6):
        return up in (2, 3, 4, 5, 6) if das else up in (3, 4, 5, 6)
    if (card == 5):          # play as hard 10
        return False
    if (card == 4):
        return up in (5, 6) if das else False
    if (card in (2, 3)):
        return up in (2, 3, 4, 5, 6, 7) if das else up in (4, 5, 6, 7)
    return False


def _softPlay(total, up, canDouble):
    if (total >= 20):                    # soft 20, 21
        return STAND
    if (total == 19):                    # A,8  (H17 doubles vs 6)
        if (up == 6 and canDouble):
            return DOUBLE
        return STAND
    if (total == 18):                    # A,7
        if (up in (2, 3, 4, 5, 6)):
            return DOUBLE if canDouble else STAND
        if (up in (7, 8)):
            return STAND
        return HIT
    if (total == 17):                    # A,6
        if (up in (3, 4, 5, 6)):
            return DOUBLE if canDouble else HIT
        return HIT
    if (total in (15, 16)):              # A,4 / A,5
        if (up in (4, 5, 6)):
            return DOUBLE if canDouble else HIT
        return HIT
    if (total in (13, 14)):              # A,2 / A,3
        if (up in (5, 6)):
            return DOUBLE if canDouble else HIT
        return HIT
    return HIT                           # soft 12 (A,A held)


def _hardPlay(total, up, canDouble):
    if (total >= 17):
        return STAND
    if (total in (13, 14, 15, 16)):
        return STAND if up in (2, 3, 4, 5, 6) else HIT
    if (total == 12):
        return STAND if up in (4, 5, 6) else HIT
    if (total == 11):                    # H17 doubles vs everything
        return DOUBLE if canDouble else HIT
    if (total == 10):
        if (up in (2, 3, 4, 5, 6, 7, 8, 9)):
            return DOUBLE if canDouble else HIT
        return HIT
    if (total == 9):
        if (up in (3, 4, 5, 6)):
            return DOUBLE if canDouble else HIT
        return HIT
    return HIT                           # 5-8


def _deviation(total, up, tc, canDouble):
    """Return a count-based override, or None to defer to basic strategy.

    The hard-total and doubling members of the Hi-Lo "Illustrious 18" index plays
    (Schlesinger), each tagged with its Hi-Lo index. Insurance (+3) and the two
    pair-split plays (10,10 v 5 / v 6) are not hard-total index plays this engine
    applies here. 11 v A (+1) is included for completeness but is a no-op in this
    H17 game, where basic already doubles 11 v A."""
    # Stiff totals: stand as the deck turns rich, hit as it turns poor.
    if (total == 16 and up == 10):
        return STAND if tc >= 0 else None        # 16 v T   index  0
    if (total == 16 and up == 9):
        return STAND if tc >= 5 else None        # 16 v 9   index +5
    if (total == 15 and up == 10):
        return STAND if tc >= 4 else None        # 15 v T   index +4
    if (total == 13 and up == 2):
        return HIT if tc < -1 else None          # 13 v 2   index -1
    if (total == 13 and up == 3):
        return HIT if tc < -2 else None          # 13 v 3   index -2
    if (total == 12 and up == 2):
        return STAND if tc >= 3 else None        # 12 v 2   index +3
    if (total == 12 and up == 3):
        return STAND if tc >= 2 else None        # 12 v 3   index +2
    if (total == 12 and up == 4):
        return HIT if tc < 0 else None           # 12 v 4   index  0
    if (total == 12 and up == 5):
        return HIT if tc < -2 else None          # 12 v 5   index -2
    if (total == 12 and up == 6):
        return HIT if tc < -1 else None          # 12 v 6   index -1
    # Strong totals: double when the deck is rich enough.
    if (total == 11 and up == 1):
        return DOUBLE if (tc >= 1 and canDouble) else None    # 11 v A  index +1
    if (total == 10 and up in (10, 1)):
        return DOUBLE if (tc >= 4 and canDouble) else None    # 10 v T/A index +4
    if (total == 9 and up == 2):
        return DOUBLE if (tc >= 1 and canDouble) else None    # 9 v 2   index +1
    if (total == 9 and up == 7):
        return DOUBLE if (tc >= 3 and canDouble) else None    # 9 v 7   index +3
    return None


def _pairDeviation(card, up, tc, engine=False, h17=True):
    """The two Illustrious-18 split plays: break a 20 (10,10) against a weak
    upcard when the deck is ten-rich enough that two ten-anchored hands beat
    standing on it. Textbook Hi-Lo indices: v 5 at +5, v 6 at +4."""
    if (card == 10):
        idx = ENGINE_INDICES[bool(h17)]["split10"] if engine else {5: 5.0, 6: 4.0}
        t = idx.get(up)
        if (t is not None and tc >= t):
            return SPLIT
    return None


# Engine-derived index thresholds for our exact game (6 decks, no surrender),
# from precompute_indices.py: for each cell, the Hi-Lo true count where the
# deviation action's exact EV overtakes basic's (regression of the
# per-composition EV gap on the true count). Keyed by hitSoft17 -- the dealer
# rule shifts several indices (e.g. 10 v A: +2.9 under H17 vs +3.7 under S17).
# The surrender rule does not move these cells (none of the EV gaps involve
# the surrender action). COUNTX plays these; COUNT plays the textbook
# Illustrious-18 numbers above.
ENGINE_INDICES = {
    True: {   # H17
        "stand": {(16, 10): 0.14, (16, 9): 4.19, (15, 10): 3.94, (12, 2): 2.51, (12, 3): 0.87},
        "hit": {(13, 2): -1.51, (13, 3): -2.86, (12, 4): -0.58, (12, 5): -1.85, (12, 6): -3.72},
        # (11,1) omitted: a no-op under H17, where basic already doubles 11 v A
        "double": {(10, 10): 3.64, (10, 1): 2.91, (9, 2): 0.96, (9, 7): 3.47},
        "split10": {5: 4.90, 6: 3.83},
        "insurance": 3.31,
    },
    False: {  # S17
        "stand": {(16, 10): 0.14, (16, 9): 4.19, (15, 10): 3.94, (12, 2): 3.00, (12, 3): 1.28},
        "hit": {(13, 2): -1.02, (13, 3): -2.45, (12, 4): -0.21, (12, 5): -1.69, (12, 6): -1.32},
        "double": {(10, 10): 3.64, (10, 1): 3.71, (11, 1): 1.48, (9, 2): 0.95, (9, 7): 3.47},
        "split10": {5: 4.95, 6: 4.35},
        "insurance": 3.31,
    },
}


def engine_indices(h17=True):
    return ENGINE_INDICES[bool(h17)]


def _deviationEngine(total, up, tc, canDouble, h17=True):
    """Count deviations using the engine-derived thresholds (COUNTX)."""
    idx = ENGINE_INDICES[bool(h17)]
    t = idx["stand"].get((total, up))
    if (t is not None and tc >= t):
        return STAND
    t = idx["hit"].get((total, up))
    if (t is not None and tc < t):
        return HIT
    if (canDouble):
        t = idx["double"].get((total, up))
        if (t is not None and tc >= t):
            return DOUBLE
    return None
