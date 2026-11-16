# 🪰 Fly Brain Casino

Play blackjack head-to-head against a fruit fly mushroom-body circuit wired from real synapse counts (Janelia `male-cns:v1.0` connectome): ~2,250 Kenyon cells, projection-neuron sensory expansion, and a two-MBON hit/stand readout trained with dopaminergic reinforcement learning.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rules

- No dealer: you and the fly each get two cards, and the better hand ≤ 21 wins the showdown.
- Every time you hit, the fly takes one live Mushroom Body action (watch its action log grow turn by turn).
- Untrained, the fly mostly freezes — hit **Train Fly Brain** in the sidebar to give it dopamine-driven practice first.

## Re-extract weights

`real_malecns_weights.npz` ships with the repo. To rebuild it from neuPrint:

```bash
python main.py   # prompts for a neuPrint token, or set NEUPRINT_APPLICATION_CREDENTIALS
```

Data: [male-cns:v1.0](https://neuprint.janelia.org) connectome, Janelia Research Campus.
