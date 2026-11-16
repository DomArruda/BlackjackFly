"""One-time connectome extractor for the Fly Brain Casino.

Queries the Janelia male-cns:v1.0 neuPrint server for Kenyon Cell / MBON /
Projection Neuron synapse counts and writes real_malecns_weights.npz, which
app.py loads at startup.

Usage:
    set NEUPRINT_APPLICATION_CREDENTIALS=<your token>   (or you will be prompted)
    python main.py
"""
import os

import numpy as np
from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC

CACHE_FILE = "real_malecns_weights.npz"


def get_token():
    tok = (
        os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN")
    )
    return tok or input("Paste your neuPrint auth token (neuprint.janelia.org → Account): ").strip()


def extract_weights(auth_token):
    client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=auth_token)

    # 1. Kenyon Cells -> MBON readout (rows land as MBON03 Stand, MBON01 Hit)
    _, kc_mbon_edges = fetch_adjacencies(NC(type="KC.*"), NC(type="MBON(01|03).*"))
    kc_mbon_pivot = kc_mbon_edges.pivot_table(
        index="bodyId_post", columns="bodyId_pre", values="weight", fill_value=0
    )

    unique_mbons = kc_mbon_pivot.index.tolist()
    if len(unique_mbons) < 2:
        raise RuntimeError("Found fewer than 2 MBON partners for criteria.")

    chosen_mbons = unique_mbons[:2]
    W_kc_mbon_raw = kc_mbon_pivot.loc[chosen_mbons].to_numpy(dtype=np.float64)
    active_kc_body_ids = kc_mbon_pivot.columns.tolist()

    # 2. Projection Neurons for sensory expansion
    _, pn_kc_edges = fetch_adjacencies(NC(type=".*PN.*"), NC(bodyId=active_kc_body_ids))
    pn_kc_pivot = pn_kc_edges.pivot_table(
        index="bodyId_post", columns="bodyId_pre", values="weight", fill_value=0
    )
    pn_kc_pivot = pn_kc_pivot.reindex(index=active_kc_body_ids, fill_value=0)
    pn_weights = pn_kc_pivot.to_numpy(dtype=np.float64)

    # 3. Group PNs into 3 channels: [Hand Sum, Dealer Upcard, Usable Ace]
    chunks = np.array_split(np.arange(pn_weights.shape[1]), 3)
    W_pn_kc_raw = np.column_stack([pn_weights[:, chunk].sum(axis=1) for chunk in chunks])

    # 4. Normalize empirical synapse counts to biological conductances
    W_pn_kc = W_pn_kc_raw / np.maximum(1.0, np.linalg.norm(W_pn_kc_raw, axis=1, keepdims=True))

    non_zeros = W_kc_mbon_raw[W_kc_mbon_raw > 0]
    p95 = np.percentile(non_zeros, 95) if non_zeros.size > 0 else 1.0
    W_kc_mbon = (W_kc_mbon_raw / p95) * 0.5

    return {
        "W_pn_kc": W_pn_kc,
        "W_kc_mbon": W_kc_mbon,
        "raw_synapses": int(np.count_nonzero(W_kc_mbon_raw)),
        "n_kc": len(active_kc_body_ids),
    }


def main():
    weights = extract_weights(get_token())
    np.savez_compressed(CACHE_FILE, **weights)
    print(
        f"Wrote {CACHE_FILE}: {weights['n_kc']} KCs, "
        f"{weights['raw_synapses']} verified KC->MBON synapses."
    )


if __name__ == "__main__":
    main()
