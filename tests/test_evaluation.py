import pytest

import ld_constants as C
from ld_algorithm import run
from ld_evaluation import kmeans_across_k, kmeans_baseline_labels, rand_index
from ld_scenario import generate_instance


def test_rand_index_is_one_for_identical_partitions():
    assert rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_rand_index_is_one_up_to_relabeling():
    assert rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)


def test_rand_index_penalizes_disagreement():
    assert rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.6


def test_rand_index_excludes_true_label_minus_one():
    """Bruecken-/Ausreisserpunkte (true_label -1, siehe ld_scenario.py::_add_bridge)
    haben keine echte Gruppenzugehoerigkeit und muessen ausgeschlossen werden - wie in
    hdbscan-demo."""
    with_bridge = rand_index([0, 0, 1, 1, -1], [0, 0, 1, 1, 2])
    without_bridge = rand_index([0, 0, 1, 1], [0, 0, 1, 1])
    assert with_bridge == without_bridge == pytest.approx(1.0)


def test_kmeans_baseline_returns_k_distinct_labels():
    instance = generate_instance(n_points=60, k=3, spread=0.15, shape="blobs", seed=1)
    labels = kmeans_baseline_labels(instance.as_array(), k=3, seed=1)
    assert len(set(labels.tolist())) == 3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_leiden_finds_correct_k_without_any_k_parameter(seed):
    """Kern-Nachweis 1: bei klar getrennten Gruppen findet Leiden die wahre
    Gruppenzahl UND eine (fast) perfekte Partition - ganz ohne dass irgendwo ein
    Ziel-k uebergeben wurde."""
    instance = generate_instance(n_points=120, k=4, spread=0.12, shape="blobs", seed=seed)
    result = run(instance.as_array(), n_neighbors=8, resolution=1.0, seed=seed)
    ri = rand_index(instance.true_labels, result.final_labels)
    assert abs(result.n_communities - instance.k) <= 2
    assert ri > 0.9


def test_leiden_finds_correct_k_with_many_groups():
    """Bleibt robust auch bei deutlich hoeherer wahrer Gruppenzahl."""
    instance = generate_instance(n_points=200, k=8, spread=0.1, shape="blobs", seed=2)
    result = run(instance.as_array(), n_neighbors=8, resolution=1.0, seed=2)
    ri = rand_index(instance.true_labels, result.final_labels)
    assert result.n_communities == 8
    assert ri > 0.95


def test_resolution_limit_merges_small_well_separated_clusters():
    """Kern-Nachweis 2 (die ehrliche Botschaft der Demo): bei vielen kleinen, aber klar
    getrennten Gruppen verschmilzt Standard-Modularitaet (gamma=1) einige davon - die
    gefundene Gruppenzahl liegt deutlich unter der wahren, obwohl die Trennung objektiv
    klar ist."""
    instance = generate_instance(n_points=60, k=20, spread=0.05, shape="blobs", seed=1)
    result = run(instance.as_array(), n_neighbors=4, resolution=1.0, seed=1)
    assert result.n_communities < 15  # deutlich unter den wahren 20


def test_higher_resolution_partially_mitigates_but_does_not_fully_fix():
    """Ein hoeherer Aufloesungsparameter hilft, behebt das Limit aber nicht
    vollstaendig - kein Free Lunch."""
    instance = generate_instance(n_points=60, k=20, spread=0.05, shape="blobs", seed=1)
    low_res = run(instance.as_array(), n_neighbors=4, resolution=1.0, seed=1)
    high_res = run(instance.as_array(), n_neighbors=4, resolution=3.0, seed=1)
    assert high_res.n_communities > low_res.n_communities
    assert high_res.n_communities < 20  # immer noch nicht die volle wahre Gruppenzahl


def test_kmeans_across_k_shows_sensitivity_to_wrong_k():
    """Kontrast zur 'kein k nötig'-Kernaussage: die k-Means-Referenz haengt sichtbar
    von der Wahl des Ziel-k ab."""
    instance = generate_instance(n_points=120, k=4, spread=0.12, shape="blobs", seed=1)
    scores = kmeans_across_k(instance.as_array(), instance.true_labels, [2, 4, 8], seed=1)
    assert scores[4] > 0.9
    assert scores[2] < scores[4]


def _run_preset_as_app_would(preset_name):
    p = C.PRESETS[preset_name]
    instance = generate_instance(
        p["n_points"], p["k"], p["spread"], p["shape"], p["seed"],
        density_imbalance=p["density_imbalance"], bridge_strength=p["bridge_strength"],
    )
    result = run(instance.as_array(), p["n_neighbors"], p["resolution"], p["seed"])
    ri = rand_index(instance.true_labels, result.final_labels)
    return instance, p, result, ri


def test_simple_preset_matches_actual_app_behavior():
    instance, _, result, ri = _run_preset_as_app_would("Einfaches Beispiel")
    assert result.n_communities == instance.k
    assert ri > 0.95


def test_many_groups_preset_matches_actual_app_behavior():
    instance, _, result, ri = _run_preset_as_app_would("Kein k nötig - viele Gruppen")
    assert result.n_communities == instance.k
    assert ri > 0.95


def test_resolution_limit_preset_matches_actual_app_behavior():
    instance, p, result, ri = _run_preset_as_app_would("Auflösungsgrenze")
    assert p["resolution"] == 1.0
    assert result.n_communities < instance.k - 5


def test_resolution_compromise_preset_matches_actual_app_behavior():
    _, low_p, low_result, _ = _run_preset_as_app_would("Auflösungsgrenze")
    instance, high_p, high_result, _ = _run_preset_as_app_would("Auflösungsparameter als Kompromiss")
    assert high_p["resolution"] > low_p["resolution"]
    assert high_result.n_communities > low_result.n_communities
    assert high_result.n_communities < instance.k


def test_combined_hardcase_preset_survives_density_imbalance_and_bridge_much_better_than_naive_methods():
    """Kern-Nachweis: anders als DBSCAN (Dichte-Ungleichgewicht) und Single-Linkage
    (Bruecken-Chaining) verschmilzt Leiden bei diesem Haertefall KEINE zwei echten
    Gruppen faelschlich - der Rand-Index bleibt hoch. Es ist aber nicht perfekt immun:
    die Bruecke selbst kann eine eigene kleine Community bilden, daher found_k typischer-
    weise etwas UEBER dem wahren k statt exakt gleich (siehe PRESET_HELP in app.py)."""
    instance, p, result, ri = _run_preset_as_app_would("Kombinierter Härtefall (Dichte + Brücke)")
    assert p["density_imbalance"] > 0 and p["bridge_strength"] > 0
    assert ri > 0.95
    assert result.n_communities >= instance.k
