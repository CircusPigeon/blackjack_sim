"""Unified CLI for the blackjack lab.

    python main.py <experiment> [field=value ...]

experiments: game | heat | bankroll | stake | ceiling

examples:
    python main.py game
    python main.py game label=casino_track strategies=BASIC,COUNT,TRACK shuffle=casino
    python main.py game shuffle=csm
    python main.py game dummyPlayers=4
    python main.py game strategies=BASIC,COUNT,ORACLE      # Hi-Lo vs EoR betting
    python main.py heat
    python main.py bankroll
    python main.py ceiling ceiling_samples=40000

Every flag is a Config field; see config.py for the full list."""

import sys
from config import Config
import experiment

EXPERIMENTS = ("game", "heat", "bankroll", "stake", "ceiling")


def _coerce(value):
    low = value.lower()
    if (low in ("true", "false")):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main():
    args = sys.argv[1:]
    if (not args or args[0] in ("-h", "--help")):
        print(__doc__)
        return
    kind = args[0]
    if (kind not in EXPERIMENTS):
        print("unknown experiment '%s' -- choose from: %s" % (kind, ", ".join(EXPERIMENTS)))
        return

    overrides = {"experiment": kind}
    for a in args[1:]:
        if ("=" not in a):
            print("ignoring '%s' (expected field=value)" % a)
            continue
        key, val = a.split("=", 1)
        if (key == "strategies"):
            overrides[key] = tuple(val.split(","))
        else:
            overrides[key] = _coerce(val)

    try:
        config = Config(**overrides)
    except TypeError as e:
        print("bad option: %s" % e)
        return
    experiment.run(config)


main()
