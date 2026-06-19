"""A Tkinter front-end for the blackjack lab.

Choose an experiment (each is explained), set the config (hover any label for a
plain-English definition), pick plots, and Run. Run / Cancel / progress live at
the bottom of the window; results and plots stack in the scrollable right panel.
Run:  python gui.py
"""

import os
import sys
import io
import threading


def _enable_dpi():
    if (sys.platform == "win32"):
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


_enable_dpi()

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import matplotlib
matplotlib.use("Agg")

from config import Config
import experiment

OUTDIR = "results"
FONT_PT = 10
STRATEGIES = ["BASIC", "COUNT", "COUNT0", "COUNTX", "WONG", "DEALER", "TRACK",
              "ORACLE", "HIOPT2", "ZEN", "OMEGA2"]
COUNTERS = ("COUNT", "TRACK", "ORACLE", "HIOPT2", "ZEN", "OMEGA2", "COUNT0",
            "COUNTX", "WONG")


class Cancelled(Exception):
    pass


def _slug(s):
    """Filesystem-safe lowercase token from a label/title."""
    return "".join(c if c.isalnum() else "_" for c in str(s).strip().lower()).strip("_") or "plot"


EXP_TAGS = [
    ("game", "Game  —  simulate real hands (the flexible one)"),
    ("heat", "Heat  —  when does the pit catch a counter?"),
    ("bankroll", "Bankroll  —  risk of going broke vs bet size"),
    ("ceiling", "Ceiling  —  best play that is theoretically possible"),
]

DESCRIPTIONS = {
    "game": ("Simulate real hands with the rules, shuffle, strategies and table you choose. This is "
             "the flexible one: it uses every setting, can switch on live Heat and Bankroll, and can "
             "repeat the session many times (Trials). Plots: edge by true count, bankroll over time, "
             "and with Trials > 1 the result distribution and survival."),
    "heat": ("How steeply can a card counter raise their bets before the pit backs them off? Sweeps "
             "bet-aggressiveness for a Hi-Lo counter calibrated to YOUR table (decks, penetration, "
             "rules, shuffle), using the spread and Heat threshold / warm-up / rate. The first run for "
             "a new table spends ~30s calibrating, then caches it. Plot: heat curve."),
    "bankroll": ("With a finite bankroll, how does your chance of going broke trade off against growth "
                 "as you bet a bigger fraction (Kelly)? Uses a Hi-Lo counter calibrated to YOUR table "
                 "(decks, penetration, rules, shuffle) and your Bankroll units. The first run for a new "
                 "table spends ~30s calibrating, then caches it. Plot: risk-of-ruin curve."),
    "ceiling": ("The theoretical best: a solver computes perfect play for the exact remaining cards "
                "(no game is dealt) and shows how much it beats basic strategy. Uses Ceiling samples, "
                "Decks, Late surrender, and Dealer-hits-soft-17. Output: tables only (no plot)."),
}

DEFS = {
    "label": "A name for this run.",
    "strategies": ("Who sits at the table:\n"
                   "- BASIC: perfect basic strategy, same flat bet every hand\n"
                   "- COUNT: Hi-Lo card counter (bets more when the deck is rich, plus play changes)\n"
                   "- COUNT0: Hi-Lo bet spread but NO play deviations (isolates what deviations add)\n"
                   "- COUNTX: Hi-Lo spread + engine-derived index thresholds for this exact game\n"
                   "- WONG: like COUNT, but sits out hands while the true count is below the\n"
                   "  'Wong out below' threshold (back-counting; plays only the good counts)\n"
                   "- DEALER: just mimics the dealer (hits to 17)\n"
                   "- TRACK: shuffle tracker (follows clumps of high cards through a weak shuffle)\n"
                   "- ORACLE: bets using the mathematically optimal per-card weights (effect of removal)\n"
                   "- HIOPT2 / ZEN / OMEGA2: level-2 counters -- finer per-card tags that read the\n"
                   "  playing value of the deck better than Hi-Lo (slightly worse for betting).\n"
                   "  They bet AND deviate on their own count, strongest in 1-2 deck games."),
    "spread_min": ("Counter betting (COUNT / TRACK / ORACLE, and the heat experiment). The bet placed at or "
                   "below the ramp start, in units (1 unit = the table minimum). The bottom of the spread."),
    "spread_max": ("Counter betting. The largest bet, in units -- the top of the spread (a 1-to-N spread). "
                   "Your bet ramps up with the count and is capped here. Wider spreads earn more per hand "
                   "but draw heat faster -- raise this and re-run the heat experiment to see the curve shift."),
    "ramp_start": ("Counter betting. The true count at which you start raising your bet. Below it you flat-bet "
                   "the minimum; at and above it the bet ramps up."),
    "spread_slope": ("Counter betting. Extra units added to your bet per +1 true count above the ramp start -- "
                     "the steepness of the spread. 1 = gentle; 2-3 = aggressive (hits the max bet sooner, "
                     "earns more per hand but swingier and easier to detect)."),
    "wong_below": ("WONG only: sit out (no bet, no cards) while the true count is below this. The classic "
                   "back-counting play -- skip the bad counts entirely, play only the good ones."),
    "shuffle": ("How the shoe is shuffled:\n"
                "- random: perfectly mixed every shoe (the ideal)\n"
                "- casino: a realistic hand shuffle (riffles, a strip, a cut). Not fully mixed -> trackable.\n"
                "- csm: continuous shuffle machine -- dealt cards go back in every hand, so a count never builds"),
    "shuffleRiffles": ("Casino shuffle only: number of riffle passes. Fewer = a sloppier shuffle that "
                       "leaves more trackable structure (~7 fully randomize a deck; casinos do 2-3)."),
    "shuffleStrips": "Casino shuffle only: number of strip passes -- a bit of extra mixing on top of the riffles.",
    "shuffleCut": ("Casino shuffle only: a final cut. This is the casino's main defense against shuffle "
                   "tracking -- it scrambles where the tracked clumps land. Turn it OFF to see a tracker win."),
    "decks": "Number of 52-card decks in the shoe. Vegas shoes are usually 6 or 8.",
    "penetration": ("How far into the shoe the dealer deals before reshuffling (the 'cut card'). "
                    "0.75 = deal 75% of the cards, then shuffle. Deeper penetration helps a counter."),
    "blackjackPays": "Payout for a natural blackjack. 1.5 = the standard 3:2; 1.2 = the stingy 6:5 (much worse for you).",
    "hitSoft17": ("Whether the dealer draws on a 'soft 17' (an ace as 11, e.g. A-6). Hitting soft 17 "
                  "(H17) is slightly worse for the player than standing on it (S17)."),
    "surrender": ("Give up half your bet and fold after seeing your two cards and the dealer's upcard.\n"
                  "- No surrender: not offered (the default; most casinos)\n"
                  "- Late surrender: only after the dealer peeks for blackjack (a dealer natural ends the\n"
                  "  hand first, so you can't surrender to it). Worth ~0.07%.\n"
                  "- Early surrender: BEFORE the peek, so you may fold even when the dealer makes a natural\n"
                  "  (you lose only half). Rare but valuable -- worth ~0.5%, with a much wider chart."),
    "das": ("Double after split: may you double down on a hand created by splitting a pair? On is "
            "player-friendly (worth ~0.13%) and changes a few split decisions (you split 4,4 / 6,6 / "
            "2,2 / 3,3 against more upcards when you can double the result)."),
    "dummyPlayers": ("Extra bystanders who use up cards but aren't tracked. They do NOT change your "
                     "per-hand edge, but they slow the game down (fewer hands per hour)."),
    "rounds": "How many hands to deal in one session (or per trial in Trials mode).",
    "seed": "Random seed. The same seed reproduces the exact same run.",
    "trials": ("Repeat the whole session this many times with different shuffles, then report the spread "
               "of outcomes (e.g. how variable your profit is). 1 = a single session."),
    "heat_live": ("While playing, let the pit watch how your bets track the count and back you off "
                  "(bar you from the table) if it gets too obvious."),
    "bankroll_live": ("Bet a fraction of a finite, evolving bankroll (Kelly sizing) and stop if you go "
                      "broke. Lets you watch ruin happen during a session."),
    "heat_threshold": ("The bet-vs-count ramp steepness (extra units wagered per +1 true count) the pit "
                       "tolerates. Lower = stricter (catches gentler ramps sooner)."),
    "heat_warmup": ("How many hands the pit watches before it can act. Lower = it can catch a blatant "
                    "spread sooner; higher = a longer grace period."),
    "heat_rate": ("Once your betting looks like counting, the chance per hand of getting backed off. "
                  "Higher = the pit acts faster once it's onto you."),
    "bankroll_units": "Starting bankroll, in betting units (1 unit = the table minimum bet).",
    "ceiling_samples": "How many random card situations the solver evaluates exactly. More = smoother results, but slower.",
}

PLOT_BUILDERS = {
    "Edge vs true count": (None, None),            # handled specially (all selected strategies)
    "Bankroll trajectory": (None, None),          # handled specially (line vs per-strategy fan)
    "Result distribution": ("fig_result_hist", ("results",)),
    "Survival histogram": ("fig_survival", ("survival", "survival_cause")),
    "Heat curve": ("fig_heat_curve", ("heat",)),
    "Risk-of-ruin curve": ("fig_risk_curve", ("risk",)),
}

# Parameter sweep (game experiment only): label -> (Config field, type, default values).
# Run the game once per value, holding everything else fixed, and plot edge vs value.
SWEEP_FIELDS = {
    "Decks": ("numPacks", int, "1,2,4,6,8"),
    "Penetration": ("penetration", float, "0.50,0.65,0.75,0.85"),
    "Max bet (units)": ("spread_max", int, "4,8,12,16,20"),
    "Ramp slope (units/TC)": ("spread_slope", float, "1.0,1.5,2.0,3.0"),
}


def relevant_controls(exp, heat_live, bankroll_live, shuffle, strategies=()):
    base = {"label", "seed"}
    heat_knobs = {"heat_threshold", "heat_warmup", "heat_rate"}
    spread_knobs = {"spread_min", "spread_max", "ramp_start", "spread_slope"}
    # Heat sweeps the ramp slope itself, so it uses the spread bounds but not the slope.
    ramp_bounds = {"spread_min", "spread_max", "ramp_start"}
    # The game/table rules that change the per-count calibration heat & bankroll use.
    table_knobs = {"shuffle", "decks", "penetration", "blackjackPays", "hitSoft17", "surrender", "das"}
    has_counter = any(s in COUNTERS for s in strategies)
    if (exp == "game"):
        s = base | {"strategies", "dummyPlayers", "rounds", "trials",
                    "heat_live", "bankroll_live"} | table_knobs
        if (shuffle == "casino"):
            s |= {"shuffleRiffles", "shuffleStrips", "shuffleCut"}
        if (has_counter):
            s |= spread_knobs
        if ("WONG" in strategies):
            s.add("wong_below")
        if (heat_live):
            s |= heat_knobs
        if (bankroll_live):
            s.add("bankroll_units")
        return s
    if (exp == "heat"):
        s = base | heat_knobs | ramp_bounds | table_knobs
        if (shuffle == "casino"):
            s |= {"shuffleRiffles", "shuffleStrips", "shuffleCut"}
        return s
    if (exp == "bankroll"):
        s = base | {"bankroll_units"} | table_knobs
        if (shuffle == "casino"):
            s |= {"shuffleRiffles", "shuffleStrips", "shuffleCut"}
        return s
    if (exp == "ceiling"):
        return base | {"hitSoft17", "surrender", "decks", "ceiling_samples"}
    return base


def estimate_seconds(cfg):
    try:
        import bankroll
        calib_missing = not os.path.exists(bankroll.calib_path(cfg))
    except Exception:
        calib_missing = True
    e = cfg.experiment
    if (e == "game"):
        players = len(cfg.strategies) + cfg.dummyPlayers + 1
        if (cfg.trials > 1):
            all_counters = all(s in COUNTERS for s in cfg.strategies)
            if ((cfg.heat_live or cfg.bankroll_live) and all_counters):
                hands = cfg.trials * min(cfg.rounds, 5000)
            else:
                hands = cfg.trials * cfg.rounds
        else:
            hands = cfg.rounds
        t = hands * players * 3.0e-5
        if ("TRACK" in cfg.strategies):
            t += 2.0
        return t
    if (e == "heat"):
        return 15.0 + (130.0 if calib_missing else 0.0)
    if (e == "bankroll"):
        return 35.0 + (130.0 if calib_missing else 0.0)
    if (e == "ceiling"):
        total = cfg.ceiling_samples + 4 * max(10000, cfg.ceiling_samples // 4)
        return total / 750.0
    return 1.0


def human_time(sec):
    if (sec < 2):
        return "instant"
    if (sec < 90):
        return "~%d sec" % int(round(sec))
    return "~%d min" % int(round(sec / 60.0))


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _=None):
        if (self.tip is not None or not self.text):
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry("+%d+%d" % (x, y))
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, padx=6, pady=4, wraplength=480).pack()

    def _hide(self, _=None):
        if (self.tip is not None):
            self.tip.destroy()
            self.tip = None


class ScrollArea(ttk.Frame):
    """Vertically scrollable panel. stretch=True makes the content fill the width
    (for plots); stretch=False keeps the content's natural width (for controls)."""

    def __init__(self, master, stretch=True):
        super().__init__(master)
        self.stretch = stretch
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        if (stretch):
            self.canvas.bind("<Configure>",
                             lambda e: self.canvas.itemconfigure(self._win, width=e.width))

    def _on_inner(self, _=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if (not self.stretch):
            self.canvas.configure(width=self.inner.winfo_reqwidth())

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Blackjack Lab")
        self._canvases = []
        self._cancel = threading.Event()
        self.controls = {}
        self._running = False
        self._prog_val = 0.0
        self._prog_label = ""
        self._run_is_game = False
        self._run_label = "gui_run"
        self._saved = 0

        self.exp_var = tk.StringVar(value="game")

        # --- bottom action bar (always visible) ---
        action = ttk.Frame(root, padding=(10, 6))
        action.pack(side="bottom", fill="x")
        self.run_btn = ttk.Button(action, text="Run", command=self.on_run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(action, text="Cancel", command=self.on_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.saveplots_var = tk.BooleanVar(value=False)
        save_cb = ttk.Checkbutton(action, text="Save plots", variable=self.saveplots_var)
        save_cb.pack(side="left", padx=(12, 0))
        Tooltip(save_cb, "When checked, the shown plots are written as PNGs to the %s/ folder "
                         "(named after the run label). Off by default." % OUTDIR)
        cli_btn = ttk.Button(action, text="Copy CLI", command=self.on_copy_cli)
        cli_btn.pack(side="left", padx=(8, 0))
        Tooltip(cli_btn, "Copy the current settings as a reproducible 'python main.py ...' command "
                         "(to the clipboard and the results panel), so an exploration you found by "
                         "clicking can be rerun, scripted, or saved.")
        self.progbar = ttk.Progressbar(action, mode="determinate", maximum=100, length=220)
        self.progbar.pack(side="right")
        self.estimate_var = tk.StringVar(value="")
        ttk.Label(action, textvariable=self.estimate_var).pack(side="left", padx=12)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(action, textvariable=self.status, foreground="#225").pack(side="left", padx=12)

        # --- 1. experiment chooser (top) ---
        expbox = ttk.LabelFrame(root, text="1.  Choose an experiment", padding=8)
        expbox.pack(side="top", fill="x", padx=10, pady=(10, 4))
        radios = ttk.Frame(expbox)
        radios.pack(side="top", fill="x")
        for value, label in EXP_TAGS:
            ttk.Radiobutton(radios, text=label, value=value, variable=self.exp_var,
                            command=self.on_change).pack(side="left", padx=(0, 18))
        self.desc_var = tk.StringVar()
        self.desc_lbl = ttk.Label(expbox, textvariable=self.desc_var, justify="left", foreground="#333")
        self.desc_lbl.pack(side="top", fill="x", pady=(6, 0))
        self.desc_lbl.bind("<Configure>",
                           lambda e: self.desc_lbl.config(wraplength=max(200, e.width - 8)))

        # --- body: scrollable controls (left) + scrollable results (right) ---
        body = ttk.Frame(root)
        body.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 6))
        self.leftscroll = ScrollArea(body, stretch=False)
        self.leftscroll.pack(side="left", fill="y")
        left = self.leftscroll.inner
        self.results = ScrollArea(body, stretch=True)
        self.results.pack(side="left", fill="both", expand=True, padx=(12, 0))

        cfgbox = ttk.LabelFrame(left, text="2.  Settings  (hover a label for what it means)", padding=8)
        cfgbox.pack(side="top", fill="x")

        # --- config vars ---
        self.label_var = tk.StringVar(value="gui_run")
        self.shuffle_var = tk.StringVar(value="random")
        self.riffles_var = tk.StringVar(value="3")
        self.strips_var = tk.StringVar(value="1")
        self.cut_var = tk.BooleanVar(value=True)
        self.packs_var = tk.StringVar(value="6")
        self.pen_var = tk.StringVar(value="0.75")
        self.bjp_var = tk.StringVar(value="1.5")
        self.h17_var = tk.BooleanVar(value=True)
        self.surr_var = tk.StringVar(value="No surrender")
        self.das_var = tk.BooleanVar(value=True)
        self.dummies_var = tk.StringVar(value="0")
        self.rounds_var = tk.StringVar(value="500000")
        self.seed_var = tk.StringVar(value="42")
        self.trials_var = tk.StringVar(value="1")
        self.heatlive_var = tk.BooleanVar(value=False)
        self.bankrolllive_var = tk.BooleanVar(value=False)
        self.heatthr_var = tk.StringVar(value="2.0")
        self.heatwarm_var = tk.StringVar(value="25")
        self.heatrate_var = tk.StringVar(value="0.12")
        self.bunits_var = tk.StringVar(value="2000")
        self.ceil_var = tk.StringVar(value="60000")
        self.spreadmin_var = tk.StringVar(value="1")
        self.spreadmax_var = tk.StringVar(value="20")
        self.rampstart_var = tk.StringVar(value="1.0")
        self.spreadslope_var = tk.StringVar(value="2.0")
        self.wong_var = tk.StringVar(value="1.0")
        self.sweep_var = tk.StringVar(value="(none)")
        self.sweepvals_var = tk.StringVar(value="")
        self._sweep_field = None
        self.strat_vars = {s: tk.BooleanVar(value=(s in ("BASIC", "COUNT", "DEALER")))
                           for s in STRATEGIES}

        self._r = 0

        def row(text, widget, name=None, enabled="normal"):
            lbl = ttk.Label(cfgbox, text=text)
            lbl.grid(row=self._r, column=0, sticky="w", pady=2)
            widget.grid(row=self._r, column=1, sticky="we", pady=2, padx=(8, 0))
            if (name):
                self.controls[name] = (widget, enabled)
                if (name in DEFS):
                    Tooltip(lbl, DEFS[name])
                    Tooltip(widget, DEFS[name])
            self._r += 1

        def combo(var, values, width=12):
            return ttk.Combobox(cfgbox, textvariable=var, values=values, state="readonly", width=width)

        row("Label", ttk.Entry(cfgbox, textvariable=self.label_var, width=14), "label")
        strat_frame = ttk.Frame(cfgbox)
        for k, s in enumerate(STRATEGIES):
            ttk.Checkbutton(strat_frame, text=s, variable=self.strat_vars[s]).grid(
                row=k // 2, column=k % 2, sticky="w", padx=(0, 12))
        row("Strategies", strat_frame, "strategies")
        row("  Min bet (units)", combo(self.spreadmin_var, ["1", "2", "5"]), "spread_min", "readonly")
        row("  Max bet (units)", combo(self.spreadmax_var, ["4", "6", "8", "12", "16", "20", "40"]), "spread_max", "readonly")
        row("  Ramp start (true count)", combo(self.rampstart_var, ["0.0", "0.5", "1.0", "1.5", "2.0", "3.0"]), "ramp_start", "readonly")
        row("  Ramp slope (units / TC)", combo(self.spreadslope_var, ["1.0", "1.5", "2.0", "2.5", "3.0"]), "spread_slope", "readonly")
        row("  Wong out below (TC)", combo(self.wong_var, ["-2.0", "-1.0", "0.0", "1.0", "2.0"]), "wong_below", "readonly")
        row("Shuffle", combo(self.shuffle_var, ["random", "casino", "csm"]), "shuffle", "readonly")
        row("  Riffles (casino)", combo(self.riffles_var, ["1", "2", "3", "4", "5", "7"]), "shuffleRiffles", "readonly")
        row("  Strips (casino)", combo(self.strips_var, ["0", "1", "2"]), "shuffleStrips", "readonly")
        row("  Final cut (casino)", ttk.Checkbutton(cfgbox, variable=self.cut_var), "shuffleCut")
        row("Decks", combo(self.packs_var, ["1", "2", "4", "6", "8"]), "decks", "readonly")
        row("Penetration", combo(self.pen_var, ["0.50", "0.65", "0.75", "0.85"]), "penetration", "readonly")
        row("Blackjack pays", combo(self.bjp_var, ["1.5", "1.2"]), "blackjackPays", "readonly")
        row("Dealer hits soft 17", ttk.Checkbutton(cfgbox, variable=self.h17_var), "hitSoft17")
        row("Surrender", combo(self.surr_var, ["No surrender", "Early surrender", "Late surrender"], width=16), "surrender", "readonly")
        row("Double after split", ttk.Checkbutton(cfgbox, variable=self.das_var), "das")
        row("Dummy players", combo(self.dummies_var, ["0", "1", "2", "4", "6"]), "dummyPlayers", "readonly")
        row("Rounds / session", ttk.Entry(cfgbox, textvariable=self.rounds_var, width=14), "rounds")
        row("Seed", ttk.Entry(cfgbox, textvariable=self.seed_var, width=14), "seed")
        row("Trials (repeat runs)", ttk.Entry(cfgbox, textvariable=self.trials_var, width=14), "trials")
        row("Live heat (back-off)",
            ttk.Checkbutton(cfgbox, variable=self.heatlive_var, command=self.on_change), "heat_live")
        row("Live bankroll (Kelly)",
            ttk.Checkbutton(cfgbox, variable=self.bankrolllive_var, command=self.on_change), "bankroll_live")
        row("  Heat threshold", ttk.Entry(cfgbox, textvariable=self.heatthr_var, width=14), "heat_threshold")
        row("  Heat warm-up (hands)", ttk.Entry(cfgbox, textvariable=self.heatwarm_var, width=14), "heat_warmup")
        row("  Heat detect rate", ttk.Entry(cfgbox, textvariable=self.heatrate_var, width=14), "heat_rate")
        row("Bankroll units", ttk.Entry(cfgbox, textvariable=self.bunits_var, width=14), "bankroll_units")
        row("Ceiling samples", ttk.Entry(cfgbox, textvariable=self.ceil_var, width=14), "ceiling_samples")

        sweepbox = ttk.LabelFrame(left, text="2b.  Parameter sweep  (game only, optional)", padding=8)
        sweepbox.pack(side="top", fill="x", pady=(8, 0))
        ttk.Label(sweepbox, text="Sweep over").grid(row=0, column=0, sticky="w", pady=2)
        sweep_combo = ttk.Combobox(sweepbox, textvariable=self.sweep_var, state="readonly", width=18,
                                   values=["(none)"] + list(SWEEP_FIELDS))
        sweep_combo.grid(row=0, column=1, sticky="we", pady=2, padx=(8, 0))
        ttk.Label(sweepbox, text="Values").grid(row=1, column=0, sticky="w", pady=2)
        sweep_entry = ttk.Entry(sweepbox, textvariable=self.sweepvals_var, width=20)
        sweep_entry.grid(row=1, column=1, sticky="we", pady=2, padx=(8, 0))
        self._sweep_widgets = [sweep_combo, sweep_entry]
        Tooltip(sweep_combo, "Run the game once per value (comma-separated, below), holding everything "
                             "else fixed, and plot each strategy's edge against the swept parameter. "
                             "'(none)' = a normal single run. Takes about (number of values) x a single run.")
        self.sweep_var.trace_add("write", self._on_sweep_change)

        plotbox = ttk.LabelFrame(left, text="3.  Plots to show", padding=8)
        plotbox.pack(side="top", fill="x", pady=(8, 0))
        self.plot_vars = {}
        self.plot_cbs = {}
        for opt in PLOT_BUILDERS:
            v = tk.BooleanVar(value=(opt == "Edge vs true count"))
            cb = ttk.Checkbutton(plotbox, text=opt, variable=v)
            cb.pack(anchor="w")
            self.plot_vars[opt] = v
            self.plot_cbs[opt] = cb
        self.plothint_var = tk.StringVar()
        ttk.Label(plotbox, textvariable=self.plothint_var, foreground="#666",
                  justify="left").pack(anchor="w", pady=(6, 0))

        for var in (self.rounds_var, self.dummies_var, self.ceil_var, self.packs_var,
                    self.heatthr_var, self.heatwarm_var, self.heatrate_var, self.bunits_var,
                    self.sweepvals_var):
            var.trace_add("write", self.update_estimate)
        for v in self.strat_vars.values():
            v.trace_add("write", lambda *a: self.on_change())
        self.trials_var.trace_add("write", lambda *a: self.on_change())
        self.shuffle_var.trace_add("write", lambda *a: self.on_change())
        root.bind_all("<MouseWheel>", self._on_wheel)

        self.on_change()

    # --- reactive UI ---
    def available_plots(self):
        exp = self.exp_var.get()
        if (exp == "game" and self.sweep_var.get() in SWEEP_FIELDS):
            return []                              # a sweep draws its own edge-vs-parameter plot
        if (exp == "heat"):
            return ["Heat curve"]
        if (exp == "bankroll"):
            return ["Risk-of-ruin curve"]
        if (exp == "ceiling"):
            return []
        try:
            trials = int(self.trials_var.get())
        except ValueError:
            trials = 1
        if (trials > 1):
            plots = ["Edge vs true count", "Bankroll trajectory", "Result distribution"]
            if (self.heatlive_var.get() or self.bankrolllive_var.get()):
                plots.append("Survival histogram")
            return plots
        return ["Edge vs true count", "Bankroll trajectory"]

    def _on_sweep_change(self, *_):
        name = self.sweep_var.get()
        if (name in SWEEP_FIELDS):
            self.sweepvals_var.set(SWEEP_FIELDS[name][2])
        self.on_change()

    def on_change(self, *_):
        exp = self.exp_var.get()
        self.desc_var.set(DESCRIPTIONS.get(exp, ""))
        sweep_on = (exp == "game")
        for w in getattr(self, "_sweep_widgets", []):
            w.configure(state=("readonly" if (sweep_on and w is self._sweep_widgets[0]) else
                               ("normal" if sweep_on else "disabled")))
        if (not sweep_on and self.sweep_var.get() != "(none)"):
            self.sweep_var.set("(none)")
        strats = tuple(s for s in STRATEGIES if self.strat_vars[s].get())
        rel = relevant_controls(exp, self.heatlive_var.get(), self.bankrolllive_var.get(),
                                self.shuffle_var.get(), strats)
        for name, (w, enabled) in self.controls.items():
            on = name in rel
            if (name == "strategies"):
                for child in w.winfo_children():
                    child.configure(state=("normal" if on else "disabled"))
            else:
                w.configure(state=(enabled if on else "disabled"))
        # When the available plot set changes, default to showing all of them.
        avail = self.available_plots()
        changed = (avail != getattr(self, "_last_avail", None))
        self._last_avail = list(avail)
        for opt, cb in self.plot_cbs.items():
            on = opt in avail
            cb.config(state=("normal" if on else "disabled"))
            if (changed):
                self.plot_vars[opt].set(on)
            elif (not on and self.plot_vars[opt].get()):
                self.plot_vars[opt].set(False)
        if (avail and not any(self.plot_vars[o].get() for o in avail)):
            self.plot_vars[avail[0]].set(True)
        if (exp == "game" and self.sweep_var.get() in SWEEP_FIELDS):
            self.plothint_var.set("A parameter sweep draws its own plot (edge vs the swept "
                                  "value), so these don't apply.")
        else:
            self.plothint_var.set("Greyed-out plots need the matching mode: Result distribution\n"
                                  "needs Trials > 1; Survival needs Trials > 1 with live heat or bankroll.")
        self.update_estimate()

    def update_estimate(self, *_):
        try:
            cfg = self.build_config()
            secs = estimate_seconds(cfg)
            # a parameter sweep runs the game once per value
            if (cfg.experiment == "game" and self.sweep_var.get() in SWEEP_FIELDS):
                nvals = len([x for x in self.sweepvals_var.get().split(",") if x.strip()])
                if (nvals > 1):
                    secs *= nvals
            self.estimate_var.set("estimated runtime: " + human_time(secs))
        except Exception:
            self.estimate_var.set("estimated runtime: -")

    # --- config / run ---
    def build_config(self):
        strats = tuple(s for s in STRATEGIES if self.strat_vars[s].get()) or ("BASIC",)
        return Config(
            experiment=self.exp_var.get(),
            label=(self.label_var.get().strip() or "gui_run"),
            strategies=strats,
            spread_min=int(self.spreadmin_var.get()),
            spread_max=int(self.spreadmax_var.get()),
            ramp_start=float(self.rampstart_var.get()),
            spread_slope=float(self.spreadslope_var.get()),
            wong_below=float(self.wong_var.get()),
            shuffle=self.shuffle_var.get(),
            shuffleRiffles=int(self.riffles_var.get()),
            shuffleStrips=int(self.strips_var.get()),
            shuffleCut=self.cut_var.get(),
            numPacks=int(self.packs_var.get()),
            penetration=float(self.pen_var.get()),
            blackjackPays=float(self.bjp_var.get()),
            hitSoft17=self.h17_var.get(),
            surrender=(self.surr_var.get() != "No surrender"),
            earlySurrender=(self.surr_var.get() == "Early surrender"),
            doubleAfterSplit=self.das_var.get(),
            dummyPlayers=int(self.dummies_var.get()),
            rounds=int(self.rounds_var.get()),
            seed=int(self.seed_var.get()),
            trials=int(self.trials_var.get()),
            heat_live=self.heatlive_var.get(),
            bankroll_live=self.bankrolllive_var.get(),
            heat_threshold=float(self.heatthr_var.get()),
            heat_warmup=int(self.heatwarm_var.get()),
            heat_rate=float(self.heatrate_var.get()),
            bankroll_units=float(self.bunits_var.get()),
            ceiling_samples=int(self.ceil_var.get()),
        )

    def _cli_command(self):
        """The current settings as a reproducible 'python main.py ...' command.
        Emits strategies plus any field that differs from the Config default."""
        import dataclasses
        cfg = self.build_config()
        default = Config(experiment=cfg.experiment)
        parts = ["python main.py", cfg.experiment, "strategies=" + ",".join(cfg.strategies)]
        for f in dataclasses.fields(cfg):
            name = f.name
            if (name in ("experiment", "strategies", "label")):
                continue
            val = getattr(cfg, name)
            if (val == getattr(default, name)):
                continue
            if (isinstance(val, bool)):
                parts.append("%s=%s" % (name, "true" if val else "false"))
            else:
                parts.append("%s=%s" % (name, val))
        return " ".join(parts)

    def on_copy_cli(self):
        try:
            cmd = self._cli_command()
        except (ValueError, TypeError) as e:
            self.status.set("Couldn't build command: %s" % e)
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(cmd)
        except tk.TclError:
            pass
        self.results.clear()
        self._canvases = []
        t = tk.Text(self.results.inner, wrap="word", font=("Consolas", FONT_PT), height=4)
        t.insert("1.0", cmd)
        t.config(state="disabled")
        t.pack(fill="x", padx=2, pady=(2, 10))
        self.status.set("Copied CLI command to clipboard.")

    def _check_cancel(self):
        if (self._cancel.is_set()):
            raise Cancelled()

    def _progress(self, done, total, label):
        self._prog_val = 100.0 * done / max(1, total)
        self._prog_label = "%s %s / %s" % (label, format(done, ","), format(total, ","))

    def _poll(self):
        if (not self._running):
            return
        if (self._prog_label):                      # also covers the heat/bankroll calibration pass
            self.status.set("Running...  %s" % self._prog_label)
            if (self._run_is_game):
                self.progbar["value"] = self._prog_val
        self.root.after(150, self._poll)

    def on_cancel(self):
        self._cancel.set()
        self.status.set("Cancelling...")

    def on_run(self):
        try:
            cfg = self.build_config()
        except (ValueError, TypeError) as e:
            self.status.set("Bad input: %s" % e)
            return
        # Optional parameter sweep (game only): run once per value and plot edge vs value.
        self._sweep_field = None
        sweep_name = self.sweep_var.get()
        if (cfg.experiment == "game" and sweep_name in SWEEP_FIELDS):
            field, conv, _default = SWEEP_FIELDS[sweep_name]
            try:
                vals = [conv(x.strip()) for x in self.sweepvals_var.get().split(",") if x.strip()]
            except ValueError as e:
                self.status.set("Bad sweep values: %s" % e)
                return
            if (len(vals) < 2):
                self.status.set("Enter at least two sweep values (comma-separated).")
                return
            self._sweep_field = (sweep_name, field, vals)
        self._cancel.clear()
        self._run_label = cfg.label
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self._prog_val = 0.0
        self._prog_label = ""
        self._run_is_game = (cfg.experiment == "game")
        self._running = True
        if (self._run_is_game):
            self.progbar.config(mode="determinate")
            self.progbar["value"] = 0
        else:
            self.progbar.config(mode="indeterminate")
            self.progbar.start(12)
        self.status.set("Running...  (estimated %s)" % human_time(estimate_seconds(cfg)))
        self._poll()

        def worker():
            buf = io.StringIO()
            old = sys.stdout
            bundle = {}
            cancelled = False
            try:
                sys.stdout = buf
                if (self._sweep_field):
                    name, field, vals = self._sweep_field
                    xs, series = experiment.sweep_edges(cfg, field, vals,
                                                        cancel=self._check_cancel, progress=self._progress)
                    print("Edge (%%) vs %s:" % name)
                    import analysis as A
                    A.print_table([tuple([str(xs[i])] + ["%+.3f" % series[s][i] for s in series])
                                   for i in range(len(xs))], [name] + list(series))
                    bundle = {"_sweep": (name, xs, series)}
                else:
                    bundle = experiment.run(cfg, OUTDIR, save_plots=False,
                                            cancel=self._check_cancel, progress=self._progress) or {}
            except Cancelled:
                cancelled = True
            except Exception as e:
                buf.write("\nERROR: %s\n" % e)
            finally:
                sys.stdout = old
            text = buf.getvalue()
            self.root.after(0, lambda: self._finish(text, bundle, cancelled))

        threading.Thread(target=worker, daemon=True).start()

    def _finish(self, text, bundle, cancelled):
        self._running = False
        self.progbar.stop()
        self.progbar.config(mode="determinate")
        self.progbar["value"] = 0
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        n = self._render(text, bundle)
        saved = getattr(self, "_saved", 0)
        if (cancelled):
            self.status.set("Cancelled.")
        elif (n == 0):
            self.status.set("Done - see the tables above (no plot for this run).")
        else:
            msg = "Done. Showing %d plot%s." % (n, "" if n == 1 else "s")
            if (saved):
                msg += "  Saved %d to %s/." % (saved, OUTDIR)
            self.status.set(msg)

    # --- rendering ---
    def _figs_for(self, bundle):
        import analysis as A
        if (bundle.get("_sweep")):
            name, xs, series = bundle["_sweep"]
            fig = A.fig_param_sweep(name, xs, series)
            return [("Edge vs %s" % name, fig)] if fig is not None else []
        figs = []
        for opt, var in self.plot_vars.items():
            if (not var.get()):
                continue
            if (opt == "Edge vs true count"):
                rows_by = {}
                if (bundle.get("records")):                      # single run: every strategy from the log
                    for row in bundle.get("summary", []):
                        rows_by[row[0]] = A.edge_by_true_count(bundle["records"], row[0])
                elif (bundle.get("edge_rows")):                  # trials: pooled per strategy
                    rows_by = bundle["edge_rows"]
                fig = A.fig_edge_rows_multi(rows_by) if rows_by else None
            elif (opt == "Bankroll trajectory"):
                if (bundle.get("trajectories")):
                    fig = A.fig_trajectory_fan(bundle["trajectories"])
                elif (bundle.get("records")):
                    fig = A.fig_bankroll(bundle["records"])
                else:
                    fig = None
            else:
                builder, needs = PLOT_BUILDERS[opt]
                if (builder is None or any(not bundle.get(k) for k in needs)):
                    continue
                fig = getattr(A, builder)(*[bundle[k] for k in needs])
            if (fig is not None):
                figs.append((opt, fig))
        return figs

    def _render(self, text, bundle):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.results.clear()
        self._canvases = []
        inner = self.results.inner

        body = text or "(no output)"
        nlines = body.count("\n") + 1
        txt = tk.Text(inner, wrap="none", font=("Consolas", FONT_PT), height=min(max(nlines, 3), 24))
        txt.insert("1.0", body)
        txt.config(state="disabled")
        txt.pack(fill="x", padx=2, pady=(2, 10))

        figs = self._figs_for(bundle)
        save = self.saveplots_var.get()
        if (save):
            os.makedirs(OUTDIR, exist_ok=True)
        self._saved = 0
        for title, fig in figs:
            ttk.Label(inner, text=title, font=("Segoe UI", FONT_PT + 1, "bold")).pack(anchor="w", padx=2)
            canvas = FigureCanvasTkAgg(fig, master=inner)
            canvas.get_tk_widget().pack(fill="x", padx=2, pady=(0, 14))
            canvas.draw()
            self._canvases.append(canvas)
            if (save):
                path = os.path.join(OUTDIR, "%s_%s.png" % (_slug(self._run_label), _slug(title)))
                try:
                    fig.savefig(path, dpi=120, bbox_inches="tight")
                    self._saved += 1
                except Exception:
                    pass

        self.root.update_idletasks()
        self.results.canvas.yview_moveto(0.0)
        return len(figs)

    def _on_wheel(self, e):
        w = self.root.winfo_containing(e.x_root, e.y_root)
        target = self.results.canvas
        while w is not None:
            if (w is self.leftscroll.canvas or w is self.leftscroll.inner):
                target = self.leftscroll.canvas
                break
            if (w is self.results.canvas or w is self.results.inner):
                target = self.results.canvas
                break
            w = getattr(w, "master", None)
        target.yview_scroll(int(-e.delta / 120), "units")


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(size=FONT_PT)
            except tk.TclError:
                pass
    except tk.TclError:
        pass
    root.geometry("1320x900")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
