import numpy as np

from ld_scenario import generate_instance


def test_point_count_matches_request():
    instance = generate_instance(n_points=100, k=4, spread=0.15, shape="blobs", seed=1)
    assert instance.n_points == 100


def test_reproducible_given_same_seed():
    kwargs = dict(n_points=80, k=5, spread=0.15, shape="blobs", seed=42)
    a = generate_instance(**kwargs)
    b = generate_instance(**kwargs)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_moons_k_equals_two_produces_two_true_groups():
    instance = generate_instance(n_points=150, k=2, spread=0.08, shape="moons", seed=2)
    labels = np.array(instance.true_labels)
    assert set(labels.tolist()) == {0, 1}


def test_moons_generalizes_to_k_greater_than_two():
    instance = generate_instance(n_points=150, k=4, spread=0.08, shape="moons", seed=2)
    labels = np.array(instance.true_labels)
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_supports_high_k_for_resolution_limit_scenarios():
    instance = generate_instance(n_points=60, k=20, spread=0.05, shape="blobs", seed=1)
    labels = np.array(instance.true_labels)
    assert len(set(labels.tolist())) == 20
    counts = np.bincount(labels)
    assert counts.min() >= 1
