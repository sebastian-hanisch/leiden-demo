import numpy as np
import pytest

from ld_algorithm import (
    _pairwise_agreement,
    build_similarity_graph,
    community_is_connected,
    consensus_clustering,
    cpm_quality,
    local_moving,
    modularity,
    refine,
    run,
    run_local_moving_only,
)
from ld_evaluation import rand_index
from ld_scenario import generate_instance


def test_hand_computed_modularity_two_triangles():
    """Zwei Dreiecke {0,1,2} und {3,4,5}, verbunden durch die Bruecke (2,3). Von Hand:
    2m=14, Community A={0,1,2}: 3 interne Kanten -> Summe ueber (i,j)-Paare = 6,
    Gradsumme=7, Beitrag = 6 - 7^2/14 = 2.5. Community B symmetrisch identisch: 2.5.
    Q = (2.5+2.5)/14 = 0.357142857..."""
    adjacency = np.zeros((6, 6))
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    labels = [0, 0, 0, 1, 1, 1]
    assert modularity(adjacency, labels, resolution=1.0) == pytest.approx(5.0 / 14.0)

    # Die Ein-Cluster-Partition (alles verschmolzen) hat nachweislich niedrigere Modularitaet.
    assert modularity(adjacency, [0, 0, 0, 0, 0, 0], resolution=1.0) == pytest.approx(0.0)


def test_hand_computed_cpm_two_triangles():
    """Gleiches Beispiel wie test_hand_computed_modularity_two_triangles oben: zwei
    Dreiecke {0,1,2}/{3,4,5}, verbunden durch die Bruecke (2,3). Von Hand: jede Community
    hat e_C=3 (3 interne Kanten) und n_C=3 -> C(3,2)=3, also H_CPM(gamma=1) = (3-3)+(3-3)
    = 0. Die Ein-Cluster-Partition (e_C=7, n_C=6 -> C(6,2)=15) hat H_CPM(gamma=1) = 7-15
    = -8, nachweislich schlechter."""
    adjacency = np.zeros((6, 6))
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    labels = [0, 0, 0, 1, 1, 1]
    assert cpm_quality(adjacency, labels, resolution=1.0) == pytest.approx(0.0)
    assert cpm_quality(adjacency, [0, 0, 0, 0, 0, 0], resolution=1.0) == pytest.approx(-8.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cpm_matches_leidenalg_reference_implementation(seed):
    """Unabhaengiger Kreuzvergleich der CPM-Qualitaetsfunktion gegen leidenalg's
    natives CPMVertexPartition - analog zum Modularitaets-Kreuzvergleich unten."""
    ig = pytest.importorskip("igraph")
    la = pytest.importorskip("leidenalg")

    instance = generate_instance(n_points=90, k=4, spread=0.15, shape="blobs", seed=seed)
    data = instance.as_array()
    adjacency = build_similarity_graph(data, n_neighbors=8)

    ours = np.array(
        run(data, n_neighbors=8, resolution=0.002, seed=seed, quality_function="cpm").final_labels
    )

    n = len(adjacency)
    edges, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                edges.append((i, j))
                weights.append(float(adjacency[i, j]))
    graph = ig.Graph(n=n, edges=edges)
    graph.es["weight"] = weights
    partition = la.find_partition(
        graph, la.CPMVertexPartition, weights="weight", resolution_parameter=0.002, seed=seed
    )
    theirs = np.array(partition.membership)

    iu = np.triu_indices(n, k=1)
    same_ours = (ours[:, None] == ours[None, :])[iu]
    same_theirs = (theirs[:, None] == theirs[None, :])[iu]
    agreement = float(np.mean(same_ours == same_theirs))
    assert agreement > 0.85, f"seed {seed}: agreement {agreement}"


def test_matches_networkx_modularity_formula():
    """Exakter Zahlen-Kreuzvergleich gegen networkx.algorithms.community.quality.modularity
    - deterministisch, da reine Q-Berechnung fuer eine gegebene Partition keine
    Algorithmus-Zufaelligkeit enthaelt."""
    nx = pytest.importorskip("networkx")

    instance = generate_instance(n_points=60, k=3, spread=0.15, shape="blobs", seed=1)
    adjacency = build_similarity_graph(instance.as_array(), n_neighbors=6)
    labels = np.array(instance.true_labels)

    ours = modularity(adjacency, labels, resolution=1.0)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(adjacency)))
    for i in range(len(adjacency)):
        for j in range(i + 1, len(adjacency)):
            if adjacency[i, j] > 0:
                graph.add_edge(i, j, weight=adjacency[i, j])
    communities = [set(np.where(labels == c)[0].tolist()) for c in sorted(set(labels.tolist()))]
    theirs = nx.algorithms.community.quality.modularity(graph, communities, weight="weight")

    assert ours == pytest.approx(theirs, rel=1e-9)


def test_modularity_never_decreases_across_passes():
    """Struktur-Invariante: jede der drei Leiden-Phasen ist so konstruiert, dass Q
    niemals sinkt - Analogon zu divisive-demos SSE-Monotonie/kmeans-demos
    Inertia-Monotonie, hier in die andere Richtung."""
    for seed in range(5):
        instance = generate_instance(n_points=90, k=4, spread=0.2, shape="blobs", seed=seed)
        result = run(instance.as_array(), n_neighbors=8, resolution=1.0, seed=seed)
        q_values = [p.quality for p in result.passes]
        for i in range(len(q_values) - 1):
            assert q_values[i + 1] >= q_values[i] - 1e-9


def test_cpm_never_decreases_across_passes():
    """Dieselbe Monotonie-Invariante wie oben, aber fuer CPM statt Modularitaet -
    unabhaengig von der Qualitaetsfunktion muss jede Phase H nur erhoehen oder
    gleichlassen koennen."""
    for seed in range(5):
        instance = generate_instance(n_points=90, k=4, spread=0.2, shape="blobs", seed=seed)
        result = run(instance.as_array(), n_neighbors=8, resolution=0.002, seed=seed, quality_function="cpm")
        q_values = [p.quality for p in result.passes]
        for i in range(len(q_values) - 1):
            assert q_values[i + 1] >= q_values[i] - 1e-9


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cpm_never_decreases_across_passes_with_multilevel_aggregation(seed):
    """Regressionstest fuer einen echten Bug: die urspruengliche CPM-Gewinnformel im
    lokalen Verschieben/Verfeinern liess den Faktor node_weights[i] (die "Groesse" eines
    ggf. aggregierten Knotens) im Kosten-Term weg (resolution * comm_size[c] statt
    resolution * comm_size[c] * node_weights[i]) - auf einem Szenario, das mehrere
    Aggregationsebenen durchlaeuft (k=20 kleine Gruppen), fuehrte das dazu, dass Pass 1
    eine SCHLECHTERE CPM-Qualitaet erreichte als Pass 0 (17.4 -> 2.45 in der
    urspruenglichen manuellen Reproduktion) - ein voellig falsches Verhalten fuer einen
    Algorithmus, der nachweislich monoton verbessern MUSS. Braucht ein Szenario, das
    tatsaechlich >=2 Passes durchlaeuft, sonst wuerde der Bug (der erst bei aggregierten,
    also nicht-Basisgraph-Knoten auftritt) gar nicht ausgeloest."""
    instance = generate_instance(n_points=60, k=20, spread=0.05, shape="blobs", seed=seed)
    result = run(instance.as_array(), n_neighbors=4, resolution=0.6, seed=seed, quality_function="cpm")
    assert len(result.passes) >= 2, "Testszenario muss mehrere Aggregationsebenen durchlaufen"
    q_values = [p.quality for p in result.passes]
    for i in range(len(q_values) - 1):
        assert q_values[i + 1] >= q_values[i] - 1e-9


def test_pairwise_agreement_identical_partitions_is_one():
    """Zwei identische Label-Zuweisungen (auch bei unterschiedlicher Label-Nummerierung
    der Gruppen, solange die Paar-Zugehoerigkeit gleich ist) muessen Uebereinstimmung
    1.0 ergeben."""
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 0, 1, 1, 2, 2])
    assert _pairwise_agreement([a, b]) == pytest.approx(1.0)


def test_pairwise_agreement_disjoint_partitions_is_low():
    """Eine Ein-Cluster-Partition (alle Punktpaare 'gleich') gegen die feinstmoegliche
    Partition (alle Punktpaare 'verschieden') teilen kein einziges Paar-Urteil -
    Uebereinstimmung muss 0.0 sein."""
    all_together = np.array([0, 0, 0, 0])
    all_apart = np.array([0, 1, 2, 3])
    assert _pairwise_agreement([all_together, all_apart]) == pytest.approx(0.0)


def test_pairwise_agreement_single_run_is_perfect():
    """Randfall: weniger als zwei Laeufe koennen sich nicht widersprechen, per
    Definition 1.0."""
    assert _pairwise_agreement([np.array([0, 1, 2])]) == 1.0


def test_consensus_clustering_reproducible_given_seed():
    """Konsensus-Clustering muss bei gleichem Seed exakt dasselbe Ergebnis liefern -
    wie jede andere Zufallsroutine dieser Demo (siehe COMPARISON_SEED-Konvention)."""
    instance = generate_instance(n_points=90, k=4, spread=0.2, shape="blobs", seed=1)
    data = instance.as_array()
    first = consensus_clustering(data, n_neighbors=8, resolution=1.0, seed=42, n_runs=5, max_rounds=3)
    second = consensus_clustering(data, n_neighbors=8, resolution=1.0, seed=42, n_runs=5, max_rounds=3)
    assert first.final_labels == second.final_labels
    assert first.n_communities == second.n_communities


def test_consensus_clustering_converges_on_seed_sensitive_preset():
    """Das eigens dafuer gebaute Preset 'Ergebnis haengt vom Zufall ab (Konsensus hilft)'
    muss innerhalb von CONSENSUS_MAX_ROUNDS tatsaechlich konvergieren (Konsensus-Matrix
    wird 'scharf') und dabei eine gueltige, nicht-triviale Partition liefern - sonst waere
    das Sicherheitslimit max_rounds in der Praxis die einzige Bremse, nicht echte
    Konvergenz."""
    instance = generate_instance(n_points=150, k=6, spread=0.3, shape="blobs", seed=3)
    result = consensus_clustering(
        instance.as_array(), n_neighbors=8, resolution=1.0, seed=1, n_runs=15, max_rounds=6
    )
    assert result.converged
    assert result.n_rounds <= 6
    assert result.n_communities >= 2
    assert 0.0 <= result.single_run_agreement <= 1.0
    rand_index_value = rand_index(instance.true_labels, result.final_labels)
    assert rand_index_value > 0.7


def test_consensus_clustering_works_with_cpm():
    """Konsensus-Clustering haengt nur von den zurueckgegebenen Partitionen ab, nicht von
    der internen Gewinnformel - muss also mit CPM genauso funktionieren wie mit
    Modularitaet (keine Sonderbehandlung im Code, aber ein echter Test schadet nicht)."""
    instance = generate_instance(n_points=60, k=4, spread=0.15, shape="blobs", seed=2)
    result = consensus_clustering(
        instance.as_array(), n_neighbors=6, resolution=0.01, seed=1,
        quality_function="cpm", n_runs=5, max_rounds=3,
    )
    assert result.n_communities >= 1
    assert len(result.final_labels) == 60


def test_all_communities_are_connected():
    """Kerneigenschaft von Leiden (dank Verfeinerung): jede zurueckgegebene Community
    induziert einen zusammenhaengenden Teilgraphen - ueber mehrere Zufallsszenarien
    getestet."""
    for seed in range(8):
        instance = generate_instance(n_points=100, k=5, spread=0.25, shape="blobs", seed=seed)
        data = instance.as_array()
        adjacency = build_similarity_graph(data, n_neighbors=6)
        result = run(data, n_neighbors=6, resolution=1.0, seed=seed)
        labels = np.array(result.final_labels)
        for c in set(labels.tolist()):
            idx = np.where(labels == c)[0]
            assert community_is_connected(adjacency, idx), f"seed {seed}, community {c} disconnected"


def test_refinement_can_split_a_local_moving_community():
    """Zeigt, dass die Verfeinerungsphase tatsaechlich etwas bewirkt (kein Blindgang):
    auf einem echten Szenario liefert `refine` eine ANDERE (feinere) Partition als
    `local_moving` allein - lokales Verschieben allein kann ein zu grobes, aus mehreren
    schwach verbundenen Teilen bestehendes 'Community'-Label liefern, das die
    Verfeinerung anschliessend aufspaltet."""
    instance = generate_instance(n_points=120, k=4, spread=0.12, shape="blobs", seed=1)
    adjacency = build_similarity_graph(instance.as_array(), n_neighbors=8)
    rng = np.random.default_rng(1)

    local_labels = local_moving(adjacency, resolution=1.0, rng=rng)
    refined_labels = refine(adjacency, local_labels, resolution=1.0, rng=rng)

    assert not np.array_equal(np.unique(local_labels, return_inverse=True)[1],
                               np.unique(refined_labels, return_inverse=True)[1])
    assert len(set(refined_labels.tolist())) >= len(set(local_labels.tolist()))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_leidenalg_reference_implementation(seed):
    """Unabhaengiger Kreuzvergleich gegen leidenalg/python-igraph (die
    Referenzimplementierung des Algorithmus selbst). Toleranzbasiert (wie bei
    hdbscan-demo/spectral-demo), da Knoten-Verarbeitungsreihenfolge und
    Tie-Breaking zwischen Implementierungen abweichen koennen."""
    ig = pytest.importorskip("igraph")
    la = pytest.importorskip("leidenalg")

    instance = generate_instance(n_points=90, k=4, spread=0.15, shape="blobs", seed=seed)
    data = instance.as_array()
    adjacency = build_similarity_graph(data, n_neighbors=8)

    ours = np.array(run(data, n_neighbors=8, resolution=1.0, seed=seed).final_labels)

    n = len(adjacency)
    edges, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                edges.append((i, j))
                weights.append(float(adjacency[i, j]))
    graph = ig.Graph(n=n, edges=edges)
    graph.es["weight"] = weights
    partition = la.find_partition(graph, la.ModularityVertexPartition, weights="weight", seed=seed)
    theirs = np.array(partition.membership)

    iu = np.triu_indices(n, k=1)
    same_ours = (ours[:, None] == ours[None, :])[iu]
    same_theirs = (theirs[:, None] == theirs[None, :])[iu]
    agreement = float(np.mean(same_ours == same_theirs))
    assert agreement > 0.85, f"seed {seed}: agreement {agreement}"


def test_no_refinement_ablation_matches_local_moving_semantics():
    """Die testeigene Ablation (`run_local_moving_only`, Louvain-Verhalten ohne
    Verfeinerung) ist NICHT Teil der App - sie dient hier lediglich als Kontrastfolie:
    ohne Verfeinerung ist keine Zusammenhangsgarantie mehr bewiesen (auch wenn sie auf
    den in dieser Demo gezeigten Szenarien meist zufaellig haelt) - der volle Leiden-
    Algorithmus mit Verfeinerung (`run`) ist die einzige Variante mit dieser Garantie
    (siehe test_all_communities_are_connected)."""
    instance = generate_instance(n_points=90, k=4, spread=0.2, shape="blobs", seed=1)
    ablation = run_local_moving_only(instance.as_array(), n_neighbors=8, resolution=1.0, seed=1)
    assert ablation.n_communities >= 1
