from dataclasses import dataclass, asdict


@dataclass
class Config:
    """A single, reproducible experiment definition. Every knob a run reads
    lives here, so a run is fully described by (and re-runnable from) one Config
    plus its seed. `experiment` selects which analysis experiment.run() performs;
    the card-game fields below feed all of them (the heat and ceiling experiments
    derive their inputs from the same engine)."""

    label: str = "experiment"
    seed: int = 42
    rounds: int = 100000          # hands per session (a session ends early if heroes leave)
    trials: int = 1               # >1 = Monte-Carlo the live composed session over N seeds

    # shoe
    numPacks: int = 6
    penetration: float = 0.75

    # table rules
    hitSoft17: bool = True
    blackjackPays: float = 1.5      # 3:2 = 1.5, 6:5 = 1.2
    surrender: bool = False         # surrender offered at all (off by default)
    earlySurrender: bool = False    # if surrender on: early (before the dealer peeks) vs late
    doubleAfterSplit: bool = True   # may double a hand formed by splitting a pair (DAS)
    maxHands: int = 4

    # who sits at the table (strategy tags the engine understands today)
    strategies: tuple = ("BASIC", "COUNT", "DEALER")

    # bet spread for advantage strategies (COUNT / TRACK / ORACLE): a 1-to-N
    # ramp that adds ~1 unit per true count above ramp_start, capped at spread_max
    spread_min: int = 1           # bet at/below the ramp start, in units (spread floor)
    spread_max: int = 20          # bet cap, in units (the top of the spread)
    ramp_start: float = 1.0       # true count at which the bet ramp begins
    spread_slope: float = 1.0     # extra units bet per +1 true count above ramp_start
    wong_below: float = 1.0       # WONG only: sit out hands while the true count is below this

    # shuffle
    shuffle: str = "random"         # random | casino | csm
    shuffleRiffles: int = 3         # casino: number of GSR riffles
    shuffleStrips: int = 1          # casino: number of strips
    shuffleCut: bool = True         # casino: final cut

    # table population: N untracked bystanders that consume cards
    dummyPlayers: int = 0
    dummyStrategy: str = "BASIC"

    # which experiment run() performs: game | heat | bankroll | stake | ceiling
    experiment: str = "game"

    # heat (detection): the pit watches your bet-vs-count ramp slope
    heat_threshold: float = 2.0       # tolerated ramp slope (units per true count)
    heat_warmup: int = 25             # hands the pit observes before it can act
    heat_rate: float = 0.12           # per-hand back-off probability once flagged
    heat_width: float = 0.5           # softness of the threshold (scale with the bet unit)
    heat_maxHands: int = 2000         # session length cap (heat experiment)

    # bankroll / risk of ruin
    bankroll_units: float = 2000.0    # starting bankroll, in betting units
    ruin_frac: float = 0.5            # ruin = bankroll falls to this fraction of start
    bankroll_horizon: int = 50000     # hands per simulated trip (bankroll experiment)
    unit_dollars: float = 25.0        # $ value of one betting unit (table minimum), for the 'stake' experiment
    hands_per_hour: int = 100         # rounds dealt per hour, for the win-rate-per-hour figure

    # live composition: apply heat / bankroll to counting heroes inside a game run
    heat_live: bool = False           # casino watches the hero's bet/count slope
    bankroll_live: bool = False       # hero bets fractional Kelly of a finite bankroll
    kelly_fraction: float = 0.5

    # combinatorial-analysis playing ceiling
    ceiling_samples: int = 60000

    def rules(self):
        return {
            "hitSoft17": self.hitSoft17,
            "blackjackPays": self.blackjackPays,
            "surrender": self.surrender,
            "earlySurrender": self.earlySurrender,
            "doubleAfterSplit": self.doubleAfterSplit,
            "maxHands": self.maxHands,
        }

    def make_shuffler(self):
        from shuffle import RandomShuffle, CasinoShuffle, CSM
        if (self.shuffle == "csm"):
            return CSM()
        if (self.shuffle == "casino"):
            proc = ["riffle"] * self.shuffleRiffles + ["strip"] * self.shuffleStrips
            if (self.shuffleCut):
                proc.append("cut")
            return CasinoShuffle(procedure=proc)
        return RandomShuffle()

    def to_dict(self):
        return asdict(self)
