import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fly Brain Casino: Play Against the Fly",
    page_icon="🪰",
    layout="wide",
)

CACHE_FILE = "real_malecns_weights.npz"

# -----------------------------------------------------------------------------
# Offline Biological Weight Loader
# -----------------------------------------------------------------------------
@st.cache_data
def load_biological_weights():
    """Loads the pre-extracted male-cns:v1.0 synapse matrices from disk."""
    if not os.path.exists(CACHE_FILE):
        st.error(
            f"Weights file '{CACHE_FILE}' not found! "
            "Please ensure 'real_malecns_weights.npz' is placed in the project directory."
        )
        st.stop()

    data = np.load(CACHE_FILE)
    return {
        "W_pn_kc": data["W_pn_kc"],
        "W_kc_mbon": data["W_kc_mbon"],
        "raw_synapses": int(data["raw_synapses"]),
        "n_kc": int(data["n_kc"]),
    }


connectome_data = load_biological_weights()

# -----------------------------------------------------------------------------
# Drosophila Mushroom Body Agent
# -----------------------------------------------------------------------------
class MushroomBodyAgent:
    def __init__(self, W_pn_kc, W_kc_mbon, lr=0.006, trace_decay=0.85, sparsity=0.10):
        self.W_pn_kc = W_pn_kc.copy()
        self.W_kc_mbon = W_kc_mbon.copy()
        self.W_initial = W_kc_mbon.copy()
        self.n_kc, self.n_pn = self.W_pn_kc.shape
        self.n_mbon = self.W_kc_mbon.shape[0]
        self.k_sparse = max(5, int(self.n_kc * sparsity))
        self.lr = lr
        self.trace_decay = trace_decay
        self.eligibility_trace = np.zeros_like(self.W_kc_mbon)

    def encode(self, obs):
        p_sum, d_card, ace = obs
        return np.array([(p_sum - 4.0) / 17.0, d_card / 10.0, 1.0 if ace else 0.0], dtype=np.float64)

    def get_kc_activity(self, pn_rates):
        raw = self.W_pn_kc @ pn_rates
        out = np.zeros(self.n_kc, dtype=np.float64)
        top_k = np.argpartition(raw, -self.k_sparse)[-self.k_sparse:]
        out[top_k] = np.maximum(0.0, raw[top_k])
        return out

    def select_action(self, obs, temp=0.1):
        pn = self.encode(obs)
        kc = self.get_kc_activity(pn)
        drive = self.W_kc_mbon @ kc

        shifted = (drive - np.max(drive)) / max(temp, 1e-4)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        action = int(np.random.choice(self.n_mbon, p=probs))

        self.eligibility_trace *= self.trace_decay
        post = np.zeros(self.n_mbon, dtype=np.float64)
        post[action] = 1.0
        self.eligibility_trace += np.outer(post, kc)
        return action

    def deliver_dopamine(self, reward):
        self.W_kc_mbon += self.lr * self.eligibility_trace * reward
        np.clip(self.W_kc_mbon, 0.001, 3.0, out=self.W_kc_mbon)
        self.eligibility_trace.fill(0.0)

    def get_action_deterministic(self, obs):
        pn = self.encode(obs)
        kc = self.get_kc_activity(pn)
        drives = self.W_kc_mbon @ kc
        return int(np.argmax(drives)), drives


# -----------------------------------------------------------------------------
# Casino Deck Engine
# -----------------------------------------------------------------------------
class CasinoGame:
    CARD_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}
    SUITS = ["♠", "♥", "♦", "♣"]

    def __init__(self):
        self.reset_shoe()

    def reset_shoe(self):
        self.deck = []
        for _ in range(6):
            for rank in range(1, 14):
                for suit in self.SUITS:
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
            elif rank in [11, 12, 13]:
                val += 10
            else:
                val += rank

        while val > 21 and aces > 0:
            val -= 10
            aces -= 1

        usable_ace = aces > 0
        return val, usable_ace

    @staticmethod
    def card_display(card):
        rank, suit = card
        name = CasinoGame.CARD_NAMES.get(rank, str(rank))
        is_red = suit in ["♥", "♦"]
        return name, suit, is_red


# -----------------------------------------------------------------------------
# Minified CSS & HTML Helpers
# -----------------------------------------------------------------------------
CASINO_CSS = (
    "<style>"
    ".casino-table{background:radial-gradient(circle,#0e5e2e 0%,#06381a 80%,#03210f 100%);border:12px solid #5a3818;border-radius:40px;padding:30px;color:white;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;box-shadow:0 16px 36px rgba(0,0,0,0.6),inset 0 0 50px rgba(0,0,0,0.4);margin-bottom:25px;}"
    ".table-header{text-align:center;border-bottom:2px dashed rgba(255,255,255,0.25);padding-bottom:12px;margin-bottom:20px;font-size:1.1rem;letter-spacing:2px;text-transform:uppercase;color:#e2b755;font-weight:700;}"
    ".player-slot{background:rgba(0,0,0,0.28);border:1px solid rgba(255,255,255,0.15);border-radius:18px;padding:16px;min-height:220px;}"
    ".slot-title{font-size:1.15rem;font-weight:bold;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;}"
    ".cards-container{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;}"
    ".poker-card{background:#ffffff;border-radius:8px;width:60px;height:88px;display:flex;flex-direction:column;justify-content:space-between;padding:6px;font-weight:bold;font-size:1.05rem;box-shadow:2px 4px 10px rgba(0,0,0,0.4);border:1px solid #d4d4d4;user-select:none;}"
    ".card-hidden{background:repeating-linear-gradient(45deg,#a72828,#a72828 8px,#7c1a1a 8px,#7c1a1a 16px);border:2px solid #ffffff;}"
    ".score-badge{display:inline-block;background:#e2b755;color:#1a1a1a;padding:3px 10px;border-radius:12px;font-size:0.9rem;font-weight:800;}"
    ".status-pill{padding:4px 12px;border-radius:8px;font-weight:bold;font-size:0.85rem;text-transform:uppercase;}"
    ".pill-active{background:#2980b9;color:white;}"
    ".pill-stand{background:#7f8c8d;color:white;}"
    ".pill-bust{background:#c0392b;color:white;}"
    ".pill-win{background:#27ae60;color:white;}"
    ".pill-push{background:#f39c12;color:white;}"
    "</style>"
)


def render_card_html(card, hide=False):
    if hide:
        return '<div class="poker-card card-hidden"></div>'
    name, suit, is_red = CasinoGame.card_display(card)
    color_style = "color:#c0392b;" if is_red else "color:#2c3e50;"
    return f'<div class="poker-card" style="{color_style}"><div style="font-size:0.85rem;line-height:1;">{name}</div><div style="text-align:center;font-size:1.4rem;line-height:1;">{suit}</div><div style="font-size:0.85rem;line-height:1;text-align:right;">{name}</div></div>'


def render_casino_table(game_state):
    dealer_cards = game_state["dealer_cards"]
    dealer_hide = not game_state["round_over"]
    dealer_html = "".join([render_card_html(c, hide=(i == 1 and dealer_hide)) for i, c in enumerate(dealer_cards)])
    dealer_val, _ = CasinoGame.calc_hand(dealer_cards if not dealer_hide else [dealer_cards[0]])
    dealer_disp = str(dealer_val) if not dealer_hide else f"{dealer_val} + ?"

    player_cards = game_state["player_cards"]
    player_html = "".join([render_card_html(c) for c in player_cards])
    p_val, _ = CasinoGame.calc_hand(player_cards)

    fly_cards = game_state["fly_cards"]
    fly_html = "".join([render_card_html(c) for c in fly_cards])
    fly_val, _ = CasinoGame.calc_hand(fly_cards)

    def status_badge(status):
        cls = "pill-active"
        if status in ["Win", "Blackjack!"]:
            cls = "pill-win"
        elif status in ["Bust", "Lost"]:
            cls = "pill-bust"
        elif status == "Stood":
            cls = "pill-stand"
        elif status == "Push":
            cls = "pill-push"
        return f'<span class="status-pill {cls}">{status}</span>'

    table_html = f'{CASINO_CSS}<div class="casino-table"><div class="table-header">Janelia male-cns:v1.0 Fruit Fly Casino Table</div><div style="max-width: 450px; margin: 0 auto 25px auto;"><div class="player-slot" style="text-align: center;"><div class="slot-title"><span>🤵 Automated Dealer</span><span class="score-badge">Total: {dealer_disp}</span></div><div class="cards-container" style="justify-content: center;">{dealer_html}</div><div>{status_badge(game_state["dealer_status"])}</div></div></div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;"><div class="player-slot"><div class="slot-title"><span>👤 You (Human)</span><span class="score-badge">Total: {p_val}</span></div><div class="cards-container">{player_html}</div><div>Status: {status_badge(game_state["player_status"])}</div></div><div class="player-slot"><div class="slot-title"><span>🪰 Fruit Fly Brain (MBON Drives)</span><span class="score-badge">Total: {fly_val}</span></div><div class="cards-container">{fly_html}</div><div>Status: {status_badge(game_state["fly_status"])}</div><div style="margin-top: 10px; font-size: 0.85rem; color: #ced6e0;">Fly Action Log: <b>{game_state["fly_log"]}</b></div></div></div></div>'
    st.markdown(table_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
st.title("🪰 Play Blackjack Head-to-Head Against the Fly Brain")
st.caption("Powered by actual synapse counts from the Janelia Male CNS (`male-cns:v1.0`) connectome.")

# -----------------------------------------------------------------------------
# Sidebar: Biological Specs & Training Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Biological Specs")
st.sidebar.metric("Kenyon Cells (KCs)", connectome_data["n_kc"])
st.sidebar.metric("Verified Synapses", connectome_data["raw_synapses"])

st.sidebar.header("Training Configuration")
training_episodes = st.sidebar.slider(
    "Training Episodes",
    min_value=5_000,
    max_value=60_000,
    value=25_000,
    step=5_000,
    help="Number of simulated Blackjack hands for dopaminergic reinforcement learning.",
)
learning_rate = st.sidebar.slider(
    "Dopaminergic Learning Rate (η)",
    min_value=0.001,
    max_value=0.020,
    value=0.006,
    step=0.001,
    format="%.3f",
    help="Magnitude of weight adjustment triggered by dopamine bursts (reward).",
)
sparsity_ratio = st.sidebar.slider(
    "APL Inhibition Sparsity",
    min_value=0.05,
    max_value=0.25,
    value=0.10,
    step=0.01,
    help="Fraction of Kenyon Cells active per state (winner-take-all feedback inhibition).",
)

# Initialize Session State
if "casino" not in st.session_state:
    st.session_state.casino = CasinoGame()

if "agent" not in st.session_state:
    st.session_state.agent = MushroomBodyAgent(
        connectome_data["W_pn_kc"],
        connectome_data["W_kc_mbon"],
        lr=learning_rate,
        sparsity=sparsity_ratio,
    )
    st.session_state.trained = False

agent = st.session_state.agent
agent.lr = learning_rate
agent.k_sparse = max(5, int(agent.n_kc * sparsity_ratio))

st.sidebar.metric("Trained Status", "✅ Pre-Trained" if st.session_state.trained else "⚠️ Untrained Baseline")

if st.sidebar.button("Train Fly Brain", use_container_width=True):
    progress_bar = st.sidebar.progress(0.0)
    with st.spinner("Simulating dopaminergic RL with Gymnasium Blackjack-v1..."):
        env = gym.make("Blackjack-v1")
        initial_temp = 0.20
        min_temp = 0.02
        for ep in range(1, training_episodes + 1):
            temp = max(min_temp, initial_temp * (1.0 - (ep / training_episodes)))
            obs, _ = env.reset()
            done = False
            while not done:
                act = agent.select_action(obs, temp=temp)
                obs, rew, term, trunc, _ = env.step(act)
                done = term or trunc
            agent.deliver_dopamine(rew)

            if ep % (training_episodes // 10) == 0:
                progress_bar.progress(ep / training_episodes)

        env.close()
        st.session_state.trained = True
    st.sidebar.success(f"Trained on {training_episodes:,} hands!")

if st.sidebar.button("Reset Synapses to Baseline", use_container_width=True):
    agent.W_kc_mbon = connectome_data["W_kc_mbon"].copy()
    agent.eligibility_trace.fill(0.0)
    st.session_state.trained = False
    st.sidebar.info("Synapses restored to baseline anatomical state.")

# -----------------------------------------------------------------------------
# Casino State Management
# -----------------------------------------------------------------------------
if "game_state" not in st.session_state:
    st.session_state.game_state = None


def start_new_round():
    casino = st.session_state.casino
    player_cards = [casino.draw_card(), casino.draw_card()]
    fly_cards = [casino.draw_card(), casino.draw_card()]
    dealer_cards = [casino.draw_card(), casino.draw_card()]

    p_val, _ = CasinoGame.calc_hand(player_cards)
    fly_val, _ = CasinoGame.calc_hand(fly_cards)

    st.session_state.game_state = {
        "player_cards": player_cards,
        "fly_cards": fly_cards,
        "dealer_cards": dealer_cards,
        "player_status": "Hitting..." if p_val < 21 else "Blackjack!",
        "fly_status": "Waiting...",
        "dealer_status": "Dealing...",
        "fly_log": "Observing hole card...",
        "player_done": (p_val >= 21),
        "fly_done": False,
        "round_over": False,
    }


def step_fly_turn():
    """Fly executes decisions using empirical Mushroom Body readout."""
    gs = st.session_state.game_state
    casino = st.session_state.casino

    dealer_upcard = gs["dealer_cards"][0][0]
    dealer_val = 10 if dealer_upcard in [11, 12, 13] else dealer_upcard

    fly_actions = []
    while True:
        fly_val, usable_ace = CasinoGame.calc_hand(gs["fly_cards"])
        if fly_val > 21:
            gs["fly_status"] = "Bust"
            fly_actions.append(f"Busted with {fly_val}")
            break
        if fly_val == 21:
            gs["fly_status"] = "Stood"
            fly_actions.append("Stood on 21")
            break

        # Query Mushroom Body readout
        action, drives = agent.get_action_deterministic((fly_val, dealer_val, usable_ace))
        # drives[0] = MBON03 (Stand), drives[1] = MBON01 (Hit)
        if action == 1:
            card = casino.draw_card()
            gs["fly_cards"].append(card)
            fly_actions.append(f"Hit (Drive {drives[1]:.2f} > {drives[0]:.2f})")
        else:
            gs["fly_status"] = "Stood"
            fly_actions.append(f"Stood (Drive {drives[0]:.2f} >= {drives[1]:.2f})")
            break

    gs["fly_done"] = True
    gs["fly_log"] = " ➔ ".join(fly_actions)


def resolve_dealer_and_results():
    gs = st.session_state.game_state
    casino = st.session_state.casino

    # Dealer draws until sum >= 17
    while True:
        d_val, _ = CasinoGame.calc_hand(gs["dealer_cards"])
        if d_val < 17:
            gs["dealer_cards"].append(casino.draw_card())
        else:
            break

    dealer_final, _ = CasinoGame.calc_hand(gs["dealer_cards"])
    if dealer_final > 21:
        gs["dealer_status"] = "Bust"
    else:
        gs["dealer_status"] = f"Finished {dealer_final}"

    # Evaluate Human
    p_final, _ = CasinoGame.calc_hand(gs["player_cards"])
    if p_final > 21:
        gs["player_status"] = "Bust"
    elif dealer_final > 21 or p_final > dealer_final:
        gs["player_status"] = "Win"
    elif p_final == dealer_final:
        gs["player_status"] = "Push"
    else:
        gs["player_status"] = "Lost"

    # Evaluate Fly
    fly_final, _ = CasinoGame.calc_hand(gs["fly_cards"])
    if fly_final > 21:
        gs["fly_status"] = "Bust"
    elif dealer_final > 21 or fly_final > dealer_final:
        gs["fly_status"] = "Win"
    elif fly_final == dealer_final:
        gs["fly_status"] = "Push"
    else:
        gs["fly_status"] = "Lost"

    gs["round_over"] = True


# -----------------------------------------------------------------------------
# Casino Play UI
# -----------------------------------------------------------------------------
if st.session_state.game_state is None:
    start_new_round()

gs = st.session_state.game_state
render_casino_table(gs)

# Control Buttons
col_ctrl1, col_ctrl2, col_ctrl3, _ = st.columns([1, 1, 1, 3])

if not gs["player_done"]:
    with col_ctrl1:
        if st.button("🟢 HIT (Take Card)", use_container_width=True):
            gs["player_cards"].append(st.session_state.casino.draw_card())
            pval, _ = CasinoGame.calc_hand(gs["player_cards"])
            if pval > 21:
                gs["player_status"] = "Bust"
                gs["player_done"] = True
                step_fly_turn()
                resolve_dealer_and_results()
            st.rerun()

    with col_ctrl2:
        if st.button("🛑 STAND (Hold)", use_container_width=True):
            gs["player_status"] = "Stood"
            gs["player_done"] = True
            step_fly_turn()
            resolve_dealer_and_results()
            st.rerun()
else:
    with col_ctrl3:
        if st.button("🔄 Deal Next Hand", use_container_width=True):
            start_new_round()
            st.rerun()

# -----------------------------------------------------------------------------
# Biological Diagnostic Expander
# -----------------------------------------------------------------------------
with st.expander("🔬 Live Fly Brain Readout & Circuit Diagnostics"):
    st.write(
        "Inspect the live membrane potential drives of the Kenyon Cells (KCs) "
        "and Mushroom Body Output Neurons (MBONs) for the current hand."
    )
    if gs:
        fly_v, fly_ace = CasinoGame.calc_hand(gs["fly_cards"])
        dealer_up = gs["dealer_cards"][0][0]
        dealer_v = 10 if dealer_up in [11, 12, 13] else dealer_up

        act, drives = agent.get_action_deterministic((fly_v, dealer_v, fly_ace))
        pn_r = agent.encode((fly_v, dealer_v, fly_ace))
        kc_act = agent.get_kc_activity(pn_r)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Fly Hand Sum", fly_v)
        c2.metric("MBON03 Drive (Stand)", f"{drives[0]:.4f}")
        c3.metric("MBON01 Drive (Hit)", f"{drives[1]:.4f}")

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.stem(np.where(kc_act > 0)[0], kc_act[kc_act > 0], markerfmt=" ", basefmt="k-")
        ax.set_title(f"Sparse Firing Distribution of Kenyon Cells (Top-{agent.k_sparse} active)")
        ax.set_xlabel("Kenyon Cell Index")
        ax.set_ylabel("Activation Rate")
        st.pyplot(fig)