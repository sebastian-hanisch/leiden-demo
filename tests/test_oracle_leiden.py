"""Unabhängige Orakel für den Leiden-Kern: Ähnlichkeitsgraph (sklearn-Nachbarn + networkx),
Modularität (networkx), CPM (Paar-Summe von Hand), Aggregations-Invarianz, Lokal-Optimalität
des lokalen Verschiebens (jede Einzelverschiebung per VOLLER Qualitätsfunktion nachgerechnet,
nicht per Gewinnformel), Zusammenhang der Communities (networkx) und Rand-Index (sklearn)."""

import numpy as np
import pytest

from ld_algorithm import (
    aggregate_graph, build_similarity_graph, cpm_quality, local_moving, modularity, refine, run,
)
from ld_evaluation import rand_index
from ld_scenario import generate_instance

nx = pytest.importorskip("networkx")
neighbors = pytest.importorskip("sklearn.neighbors")
metrics = pytest.importorskip("sklearn.metrics")

CASES = [
    dict(n_points=30 + 3 * i, k=2 + i % 5, spread=0.1 + 0.07 * (i % 6), density_imbalance=0.2 * (i % 5),
         bridge_strength=0.15 * (i % 4), shape="blobs" if i % 3 else "moons", seed=i, n_neighbors=3 + i % 8)
    for i in range(10)
]


def _graph(c):
    inst = generate_instance(**{k: v for k, v in c.items() if k != "n_neighbors"})
    return inst, inst.as_array(), build_similarity_graph(inst.as_array(), c["n_neighbors"])


def _cpm_by_pairs(a, labels, gamma, node_weights=None):
    n = len(a)
    nw = np.ones(n) if node_weights is None else node_weights
    total = 0.0
    for c in set(labels.tolist()):
        idx = [i for i in range(n) if labels[i] == c]
        e = sum(a[i, j] for i in idx for j in idx if i < j) + sum(a[i, i] / 2 for i in idx)
        s = sum(nw[i] for i in idx)
        total += e - gamma * s * (s - 1) / 2
    return total


def _quality(a, labels, gamma, qf, nw=None):
    return modularity(a, labels, gamma) if qf == "modularity" else cpm_quality(a, labels, gamma, nw)


def test_similarity_graph_and_modularity_against_networkx():
    rng = np.random.default_rng(0)
    for c in CASES:
        _, x, a = _graph(c)
        n = len(x)
        d, ix = neighbors.NearestNeighbors(n_neighbors=c["n_neighbors"] + 1).fit(x).kneighbors(x)
        edges = {}
        for i in range(n):
            for j in ix[i, 1:]:
                edges[(min(i, j), max(i, j))] = float(np.linalg.norm(x[i] - x[j]))
        sigma = max(np.median(list(edges.values())), 1e-6)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for (p, q), dist in edges.items():
            g.add_edge(p, q, weight=float(np.exp(-dist ** 2 / (2 * sigma ** 2))))
        assert np.allclose(a, nx.to_numpy_array(g, nodelist=range(n), weight="weight"), atol=1e-12)
        gamma = float(rng.uniform(0.3, 4.0))
        labels = rng.integers(0, int(rng.integers(1, 6)), size=n)
        comms = [set(np.where(labels == k)[0].tolist()) for k in set(labels.tolist())]
        expected = nx.community.modularity(g, comms, weight="weight", resolution=gamma)
        assert modularity(a, labels, gamma) == pytest.approx(expected, abs=1e-10)


def test_cpm_and_aggregation_invariance():
    rng = np.random.default_rng(1)
    for c in CASES:
        _, x, a = _graph(c)
        n = len(x)
        labels = rng.integers(0, 4, size=n)
        for gamma in (0.0, 0.001, 0.05, 0.4):
            assert cpm_quality(a, labels, gamma) == pytest.approx(_cpm_by_pairs(a, labels, gamma), abs=1e-9)
        fine = rng.integers(0, int(rng.integers(2, n // 2 + 2)), size=n)
        agg, fine_r, nw = aggregate_graph(a, fine)
        coarse = rng.integers(0, int(rng.integers(1, len(agg) + 1)), size=len(agg))
        for qf, gamma in (("modularity", 1.3), ("cpm", 0.05)):
            assert _quality(agg, coarse, gamma, qf, nw) == pytest.approx(_quality(a, coarse[fine_r], gamma, qf), abs=1e-8)


@pytest.mark.parametrize("qf,gamma", [("modularity", 1.0), ("cpm", 0.05)])
def test_local_moving_result_is_a_local_optimum_under_full_quality_function(qf, gamma):
    for c in CASES[:6]:
        _, x, a = _graph(c)
        agg, _, nw = aggregate_graph(a, np.arange(len(a)) // 2)  # gewichteter Graph mit Selbstschleifen
        for graph, weights in ((a, None), (agg, nw)):
            lab = local_moving(graph, gamma, np.random.default_rng(c["seed"]), quality_function=qf, node_weights=weights)
            q0 = _quality(graph, lab, gamma, qf, weights)
            for i in range(len(graph)):
                for cand in {lab[j] for j in np.nonzero(graph[i])[0] if j != i} - {lab[i]}:
                    moved = lab.copy()
                    moved[i] = cand
                    assert _quality(graph, moved, gamma, qf, weights) <= q0 + 1e-9
            ref = refine(graph, lab, gamma, np.random.default_rng(1), quality_function=qf, node_weights=weights)
            assert all(len(set(lab[ref == k])) == 1 for k in set(ref.tolist()))  # Verfeinerung nur INNERHALB


@pytest.mark.parametrize("qf,gamma", [("modularity", 1.0), ("cpm", 0.05)])
def test_run_passes_monotone_quality_recomputed_and_communities_connected(qf, gamma):
    for c in CASES:
        _, x, a = _graph(c)
        res = run(x, c["n_neighbors"], gamma, c["seed"], quality_function=qf)
        prev = -np.inf
        for p in res.passes:
            labs = np.array(p.partition)
            q = _quality(a, labs, gamma, qf)
            assert p.quality == pytest.approx(q, abs=1e-9)
            assert p.n_communities == len(set(labs.tolist()))
            assert q >= prev - 1e-9
            prev = q
        labs = np.array(res.final_labels)
        g = nx.from_numpy_array(a)
        for k in set(labs.tolist()):
            assert nx.is_connected(g.subgraph(np.where(labs == k)[0].tolist()))


def test_rand_index_against_sklearn():
    rng = np.random.default_rng(2)
    for _ in range(40):
        n = int(rng.integers(5, 40))
        true, pred = rng.integers(-1, 3, size=n), rng.integers(0, 4, size=n)
        m = true != -1
        if m.sum() >= 2:
            assert rand_index(true, pred) == pytest.approx(metrics.rand_score(true[m], pred[m]))
