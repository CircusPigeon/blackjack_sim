import math
import random
from deck import Deck, eor_tags
from dealer import Dealer
from guest import Guest
from play import Play
from config import Config
from analysis import RECORD_COLUMNS

DEFAULT_RULES = {
    "hitSoft17": True,      # dealer hits soft 17 (H17) vs stands (S17)
    "blackjackPays": 1.5,   # 3:2 = 1.5, 6:5 = 1.2
    "surrender": True,      # late surrender offered
    "maxHands": 4,          # cap on hands after splitting
}

COUNTERS = ("COUNT", "TRACK", "ORACLE", "HIOPT2", "ZEN", "OMEGA2",
            "COUNT0", "COUNTX", "WONG")   # advantage strategies heat/bankroll apply to
LEVEL2 = ("HIOPT2", "ZEN", "OMEGA2")      # level-2 counts: bet and play on their own count


class Blackjack:
    def __init__(self, config=None, verbose=False, record=False):
        self.config = config if config is not None else Config()
        self.verbose = verbose
        self.recording = record
        self.rules = dict(DEFAULT_RULES)
        self.rules.update(self.config.rules())

        # Tracked guests (recorded) and untracked dummy bystanders (consume
        # cards only). All of them play and settle; only the tracked ones are
        # written to the record log.
        self.guests = []
        for idx, strat in enumerate(self.config.strategies):
            guest = Guest(idx + 1, strat)
            guest.rules = self.rules
            self._applySpread(guest)
            self.guests.append(guest)
        self.dummies = []
        for k in range(self.config.dummyPlayers):
            dummy = Guest(101 + k, self.config.dummyStrategy)
            dummy.rules = self.rules
            self._applySpread(dummy)
            self.dummies.append(dummy)
        self.numTracked = len(self.guests)
        self.numPlaying = len(self.guests) + len(self.dummies)

        self.dealer = Dealer(hitSoft17=self.rules["hitSoft17"])
        self.players = self.guests + self.dummies + [self.dealer]

        track = "TRACK" in self.config.strategies
        self.deck = Deck(self.config.numPacks, self.config.penetration,
                         self.config.make_shuffler(), track=track,
                         eor=eor_tags(self.rules["hitSoft17"], self.rules["surrender"]))
        self.numRound = 0
        self.numReshuffles = 0
        self.holeCounted = True
        self.records = {c: [] for c in RECORD_COLUMNS} if record else None

        # Live heat (casino back-off) and bankroll (Kelly + ruin) layers.
        self.heatLive = self.config.heat_live
        self.bankrollLive = self.config.bankroll_live
        self.heatAlpha = 0.04
        self.heatWidth = self.config.heat_width
        self.heatBase = self.config.heat_rate
        self.heatWarmup = self.config.heat_warmup
        self.act = [True] * self.numPlaying
        if (self.bankrollLive):
            for g in self.guests:
                if (g.strategy in COUNTERS):
                    g.bankrollMode = True
                    g.unit = 1
                    g.money = float(self.config.bankroll_units)
                    g.startMoney = float(self.config.bankroll_units)
                    g.kellyFrac = self.config.kelly_fraction
                    g.pivot = float(self.config.ramp_start)
                    g.minBet = 1.0
                    g.maxBet = 0.25 * self.config.bankroll_units
                    g.ruinLevel = self.config.ruin_frac * self.config.bankroll_units

    def _applySpread(self, guest):
        # Config-driven bet ramp for the advantage strategies (1-to-N spread).
        guest.minUnits = self.config.spread_min
        guest.maxUnits = self.config.spread_max
        guest.rampStart = float(self.config.ramp_start)
        guest.slope = float(self.config.spread_slope)
        guest.wongBelow = float(self.config.wong_below)

    def log(self, msg):
        if (self.verbose):
            print(msg)

    def _emit(self, **row):
        for c in RECORD_COLUMNS:
            self.records[c].append(row[c])

    def run(self):
        self.numRound += 1
        if (self.deck.needsReshuffle()):
            self.deck.reshuffle()
            self.numReshuffles += 1
            self.log("==== Reshuffling shoe (cut card reached) ====")
        self.deck.beginRound()
        tc = self.deck.getTrueCount()
        self.log("Round " + str(self.numRound) + "  | running count "
                 + str(self.deck.getRunningCount()) + ", true count " + str(round(tc, 2)))

        # Who is still at the table this round (heroes leave when barred/ruined).
        self.act = [not self.players[i].out for i in range(self.numPlaying)]

        # Per-round snapshots for the record log (tracked guests only).
        moneyStart = [self.players[i].getMoney() for i in range(self.numTracked)]
        wageredStart = [self.players[i].totalWagered for i in range(self.numTracked)]
        initBet = [0] * self.numTracked

        # Bets (every active player at the table, tracked and dummy)
        for player in self.players:
            player.clearHand()
        for i in range(self.numPlaying):
            player = self.players[i]
            if (not self.act[i]):
                player.bet = 0
                player.bets = [0]
                player.insuranceBet = 0
                continue
            if (player.strategy == "TRACK"):
                signal = self.deck.predictedTrueCount()
            elif (player.strategy == "ORACLE"):
                signal = self.deck.getEorTrueCount()
            elif (player.strategy in LEVEL2):
                signal = self.deck.getSystemTrueCount(player.strategy)
            else:
                signal = tc
            player.calculateBet(signal)
            if (player.getBet() == 0):
                # Wonged out: sits this round out (no cards, no wager) but keeps
                # watching the count. act=False skips deal/play/settle below.
                self.act[i] = False
                player.bets = [0]
                player.insuranceBet = 0
                self.log("  " + player.getName() + " [" + player.strategy
                         + "] wongs out (TC " + str(round(signal, 2)) + ")")
                continue
            player.bets = [player.getBet()]
            player.insuranceBet = 0
            player.totalWagered += player.getBet()
            player.handsPlayed += 1
            if (i < self.numTracked):
                initBet[i] = player.getBet()
            if (self.heatLive and player.strategy in COUNTERS):
                self._casinoWatch(player, tc)
            self.log("  " + player.getName() + " [" + player.strategy + "] money "
                     + str(player.getMoney()) + ", bet " + str(player.getBet()))

        # Deal two cards to each active hand. The dealer's first card is the
        # hole card and is dealt without being counted until it is revealed.
        for idx, player in enumerate(self.players):
            if (player is self.dealer):
                card = self.deck.pullTopCard(counted=False)
                player.addCard(0, card)
                self.holeCounted = False
                self.log("    Dealer dealt a hole card")
            elif (self.act[idx]):
                self.deal(player, 0)
        for idx, player in enumerate(self.players):
            if (player is self.dealer):
                self.deal(player, 0)
            elif (self.act[idx]):
                self.deal(player, 0)
        for i in range(self.numPlaying):
            if (self.act[i]):
                self.log(self.players[i].handString(0))

        upcard = self.dealer.getShowcard()
        self.log("  Dealer shows " + str(upcard))
        if (upcard == 1):
            self.offerInsurance(tc)
        # Early surrender is decided before the dealer peeks, so a surrendered hand
        # only loses half even when the dealer turns out to have a blackjack.
        if (self.rules["surrender"] and self.rules.get("earlySurrender")):
            for i in range(self.numPlaying):
                if (self.act[i]):
                    self._offerEarlySurrender(self.players[i])

        dealerNatural = self.dealer.blackjack(0)
        if (dealerNatural):
            self.revealHole()
            self.log("  Dealer has blackjack!")
        else:
            for i in range(self.numPlaying):
                if (self.act[i]):
                    self.makePlay(self.players[i])
            if (self.anyLiveHand()):
                self.revealHole()
                self.makePlay(self.dealer)
                self.log("  Dealer finishes on " + str(self.dealer.getTotal(0))
                         + (" (bust)" if self.dealer.bust(0) else ""))

        self.settle(dealerNatural)
        self.log("********")

        # Ruin check (takes effect from the next round).
        for g in self.guests:
            if (g.bankrollMode and not g.out and g.money <= g.ruinLevel):
                g.ruined = True
                g.out = True

        if (self.recording):
            for i in range(self.numTracked):
                p = self.players[i]
                self._emit(round=self.numRound, strategy=p.strategy,
                           true_count=round(tc, 3), bet=initBet[i],
                           wagered=p.totalWagered - wageredStart[i],
                           result=p.getMoney() - moneyStart[i],
                           bankroll=p.getMoney())

    def revealHole(self):
        # Turn the hole card face up and fold it into the running count. If all
        # players bust, this is never called, so the counter never sees it.
        if (not self.holeCounted):
            self.deck.applyCount(self.dealer.hand[0][0])
            self.holeCounted = True
            self.log("  Dealer reveals hole card " + str(self.dealer.hand[0][0]))

    def anyLiveHand(self):
        for gi in range(self.numPlaying):
            if (not self.act[gi]):
                continue
            player = self.players[gi]
            for hi in range(len(player.hand)):
                if (not player.bust(hi) and not player.surrendered[hi]):
                    return True
        return False

    def _casinoWatch(self, guest, tc):
        # The pit tracks an EWMA regression of bet on true count; a steep slope
        # gets the hero backed off (barred), more likely the steeper it is.
        a = self.heatAlpha
        guest.hb = (1 - a) * guest.hb + a * guest.bet
        guest.hc = (1 - a) * guest.hc + a * tc
        guest.hbc = (1 - a) * guest.hbc + a * guest.bet * tc
        guest.hcc = (1 - a) * guest.hcc + a * tc * tc
        guest.hn += 1
        cov = guest.hbc - guest.hb * guest.hc
        varc = guest.hcc - guest.hc * guest.hc
        if (varc < 1e-6):
            varc = 1e-6
        slope = cov / varc
        if (guest.hn > self.heatWarmup):
            p = self.heatBase / (1.0 + math.exp(-(slope - self.config.heat_threshold) / self.heatWidth))
            if (random.random() < p):
                guest.barred = True
                guest.out = True

    def deal(self, player, i):
        card = self.deck.pullTopCard()
        player.addCard(i, card)
        self.log("    " + player.getName() + " dealt a " + str(card))

    def transferMoney(self, player, amount):
        player.updateMoney(amount)
        self.dealer.updateMoney(-amount)

    def _playSignal(self, player):
        # The true count a strategy keys its PLAY decisions (deviations,
        # insurance) on: level-2 counters use their own count, everyone else
        # (including the dealer, who has no strategy) the Hi-Lo count.
        if (getattr(player, "strategy", None) in LEVEL2):
            return self.deck.getSystemTrueCount(player.strategy)
        return self.deck.getTrueCount()

    def offerInsurance(self, tc):
        for i in range(self.numPlaying):
            if (not self.act[i]):
                continue
            player = self.players[i]
            if (player.wantsInsurance(self._playSignal(player))):
                ins = player.getBet() // 2
                player.insuranceBet = ins
                player.totalWagered += ins
                self.log("  " + player.getName() + " takes insurance (" + str(ins) + ")")

    def makePlay(self, player):
        upcard = self.dealer.getShowcard()
        h = 0
        while (h < len(player.hand)):
            self.playHand(player, h, upcard)
            h += 1

    def _offerEarlySurrender(self, player):
        # One initial two-card hand, decided before the dealer peeks. A SURRENDER
        # from the player's strategy retires the hand (it loses half at settle,
        # which checks surrender before the dealer's blackjack).
        upcard = self.dealer.getShowcard()
        play = player.getPlay(0, upcard, self._playSignal(player), False, False, True)
        if (play == Play.SURRENDER.value):
            player.surrendered[0] = True
            player.handDone[0] = True
            self.log("  " + player.getName() + " early-surrenders vs " + str(upcard))

    def playHand(self, player, i, upcard):
        while (True):
            if (player.handDone[i] or player.bust(i) or player.blackjack(i)):
                break
            tc = self._playSignal(player)
            das = self.rules.get("doubleAfterSplit", True)
            # No-DAS: a hand formed by a split (more than one hand on the felt) can't double.
            canDouble = (len(player.hand[i]) == 2) and (das or len(player.hand) == 1)
            canSplit = (player.isPair(i) and len(player.hand) < self.rules["maxHands"])
            # Late surrender is offered during play; early surrender is decided
            # before the peek (see _offerEarlySurrender), so it's not offered here.
            canSurrender = (self.rules["surrender"] and not self.rules.get("earlySurrender")
                            and len(player.hand) == 1 and len(player.hand[i]) == 2)
            play = player.getPlay(i, upcard, tc, canDouble, canSplit, canSurrender)
            if (play == Play.DOUBLE.value and canDouble):
                self.doubleDown(player, i)
                break
            elif (play == Play.SPLIT.value and canSplit):
                self.doSplit(player, i)
            elif (play == Play.SURRENDER.value and canSurrender):
                player.surrendered[i] = True
                player.handDone[i] = True
                self.log("  " + player.getName() + " surrenders hand " + str(i + 1))
                break
            elif (play == Play.HIT.value):
                self.hit(player, i)
            elif (play == Play.STAND.value):
                self.log("  " + player.getName() + " stands hand " + str(i + 1)
                         + " on " + str(player.getTotal(i)))
                break
            else:
                # Unknown / illegal action: fall back to a safe basic line.
                if (player.getTotal(i) < 17):
                    self.hit(player, i)
                else:
                    break

    def hit(self, player, i):
        self.deal(player, i)
        if (player.bust(i)):
            self.log("  " + player.getName() + " hand " + str(i + 1)
                     + " busts on " + str(player.getTotal(i)))

    def doubleDown(self, player, i):
        player.totalWagered += player.bets[i]
        player.bets[i] *= 2
        self.deal(player, i)
        self.log("  " + player.getName() + " doubles hand " + str(i + 1) + " to "
                 + str(player.getTotal(i)) + (" (bust)" if player.bust(i) else ""))

    def doSplit(self, player, i):
        card = player.hand[i].pop()
        newIndex = len(player.hand)
        player.hand.append([card])
        player.handDone.append(False)
        player.surrendered.append(False)
        player.bets.append(player.bets[i])
        player.totalWagered += player.bets[i]
        self.deal(player, i)
        self.deal(player, newIndex)
        self.log("  " + player.getName() + " splits " + str(card) + "s into hands "
                 + str(i + 1) + " and " + str(newIndex + 1))
        if (card == 1):
            # Split aces get one card each and no further action.
            player.handDone[i] = True
            player.handDone[newIndex] = True

    def settle(self, dealerNatural):
        dealerTotal = self.dealer.getTotal(0)
        dealerBust = self.dealer.bust(0)
        bjPays = self.rules["blackjackPays"]
        for gi in range(self.numPlaying):
            if (not self.act[gi]):
                continue
            player = self.players[gi]

            if (player.insuranceBet > 0):
                if (dealerNatural):
                    self.transferMoney(player, 2 * player.insuranceBet)
                    self.log("  " + player.getName() + " insurance wins +"
                             + str(2 * player.insuranceBet))
                else:
                    self.transferMoney(player, -player.insuranceBet)
                    self.log("  " + player.getName() + " insurance loses -"
                             + str(player.insuranceBet))

            nHands = len(player.hand)
            for hi in range(nHands):
                bet = player.bets[hi]
                isNatural = (nHands == 1 and len(player.hand[hi]) == 2 and player.blackjack(hi))
                if (player.surrendered[hi]):
                    half = bet // 2
                    self.transferMoney(player, -half)
                    result = "surrenders -" + str(half)
                elif (player.bust(hi)):
                    self.transferMoney(player, -bet)
                    result = "loses (bust) -" + str(bet)
                elif (isNatural and dealerNatural):
                    result = "pushes (both blackjack)"
                elif (isNatural):
                    win = int(bjPays * bet)
                    self.transferMoney(player, win)
                    result = "blackjack +" + str(win)
                elif (dealerNatural):
                    self.transferMoney(player, -bet)
                    result = "loses to dealer BJ -" + str(bet)
                elif (dealerBust or player.getTotal(hi) > dealerTotal):
                    self.transferMoney(player, bet)
                    result = "wins +" + str(bet)
                elif (player.getTotal(hi) < dealerTotal):
                    self.transferMoney(player, -bet)
                    result = "loses -" + str(bet)
                else:
                    result = "pushes"
                self.log("  " + player.getName() + " hand " + str(hi + 1)
                         + " (" + str(player.getTotal(hi)) + ") " + result)

    def report(self):
        print("After " + str(self.numRound) + " rounds:")
        for i in range(self.numTracked):
            p = self.players[i]
            print("  " + p.getName() + " [" + p.strategy + "]"
                  + "  wagered " + str(p.totalWagered)
                  + ", profit " + str(p.getProfit())
                  + ", edge " + ("%+.3f" % (p.getEdge() * 100)) + "%")
