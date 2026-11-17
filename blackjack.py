"""Pure Blackjack game logic for the Fly Brain Casino (no Streamlit deps).

Run `python blackjack.py` to execute the self-check.
"""
import numpy as np

CARD_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}
SUITS = ["♠", "♥", "♦", "♣"]


class CasinoGame:
    """Six-deck shoe, reshuffled automatically when it runs low."""

    def __init__(self):
        self.reset_shoe()

    def reset_shoe(self):
        self.deck = []
        for _ in range(6):
            for rank in range(1, 14):
                for suit in SUITS:
                    self.deck.append((rank, suit))
        np.random.shuffle(self.deck)

    def draw_card(self):
        if len(self.deck) < 20:
            self.reset_shoe()
        return self.deck.pop()

    @staticmethod
    def calc_hand(cards):
        """Returns (best_sum, is_usable_ace)."""
        val = 0
        aces = 0
        for rank, _ in cards:
            if rank == 1:
                aces += 1
                val += 11
            elif rank in (11, 12, 13):
                val += 10
            else:
                val += rank

        while val > 21 and aces > 0:
            val -= 10
            aces -= 1

        return val, aces > 0

    @staticmethod
    def card_display(card):
        rank, suit = card
        name = CARD_NAMES.get(rank, str(rank))
        return name, suit, suit in ("♥", "♦")


def judge(player_total, fly_total):
    """Head-to-head outcome (no dealer). Returns (player_status, fly_status, headline)."""
    if player_total > 21 and fly_total > 21:
        return "Bust", "Bust", "Both busted — push"
    if player_total > 21:
        return "Bust", "Win", "Fly wins — you busted"
    if fly_total > 21:
        return "Win", "Bust", "You win — the fly busted"
    if player_total > fly_total:
        return "Win", "Lost", "You win"
    if fly_total > player_total:
        return "Lost", "Win", "The fly wins"
    return "Push", "Push", f"Push at {player_total}"


def shape_reward(outcome, final_score):
    """Dopamine shaping for a finished Gymnasium Blackjack hand.

    Win +1, push 0, loss -3*|21 - score|/21: losing at 20 or a small bust
    stings a little, while standing pat on 12 is the most expensive common
    outcome — passivity is punished hardest. The 3x amplification keeps the
    dopamine signal strong enough to learn; unamplified partial credit
    collapses toward always-stand.
    """
    if outcome > 0:
        return 1.0
    if outcome == 0:
        return 0.0
    return -3.0 * abs(21 - final_score) / 21.0


if __name__ == "__main__":
    calc = CasinoGame.calc_hand

    assert calc([(1, "♠"), (13, "♥")]) == (21, True)                    # A + K
    assert calc([(1, "♠"), (5, "♥")]) == (16, True)                     # soft 16
    assert calc([(1, "♠"), (1, "♥"), (9, "♦")]) == (21, True)           # two aces, soft 21
    assert calc([(1, "♠"), (1, "♥"), (9, "♦"), (5, "♣")]) == (16, False)  # aces demoted
    assert calc([(13, "♠"), (12, "♥"), (2, "♦")]) == (22, False)       # hard bust

    assert judge(20, 22) == ("Win", "Bust", "You win — the fly busted")
    assert judge(22, 22) == ("Bust", "Bust", "Both busted — push")
    assert judge(22, 15) == ("Bust", "Win", "Fly wins — you busted")
    assert judge(20, 20) == ("Push", "Push", "Push at 20")
    assert judge(21, 18) == ("Win", "Lost", "You win")
    assert judge(18, 21) == ("Lost", "Win", "The fly wins")

    shoe = CasinoGame()
    draws = [shoe.draw_card() for _ in range(500)]  # forces at least one reshuffle
    assert all(1 <= r <= 13 for r, _ in draws)

    sr = shape_reward
    assert sr(1, 20) == 1.0                      # win pays flat
    assert sr(0, 20) == 0.0                      # push stays neutral
    assert sr(-1, 20) == -3 / 21                 # lost at 20: mild
    assert sr(-1, 12) == -27 / 21                # passive low stand: harsh
    assert sr(-1, 23) == -6 / 21                 # bust graded automatically
    assert sr(-1, 12) < sr(-1, 20) < 0           # passive low stand hurts most
    assert sr(-1, 26) < sr(-1, 22) < 0           # bigger bust hurts more
    assert sr(-1, 23) > sr(-1, 12)               # even a bust beats passive standing

    print("blackjack.py self-check passed.")
