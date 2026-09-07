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


def test_default_density_imbalance_and_bridge_strength_are_zero():
    instance = generate_instance(n_points=100, k=4, spread=0.15, shape="blobs", seed=1)
    assert -1 not in instance.true_labels


def test_density_imbalance_lowers_local_density_of_group_zero():
    instance = generate_instance(
        n_points=150, k=3, spread=0.2, shape="blobs", seed=3, density_imbalance=0.9
    )
    points = np.array(instance.points)
    labels = np.array(instance.true_labels)

    def mean_nn_distance(group_points):
        d = np.sqrt(((group_points[:, None, :] - group_points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1).mean()

    nn_group0 = mean_nn_distance(points[labels == 0])
    nn_others = np.mean([mean_nn_distance(points[labels == i]) for i in (1, 2)])
    assert nn_group0 > nn_others * 1.5


def test_density_imbalance_works_for_moons_too():
    instance = generate_instance(
        n_points=200, k=2, spread=0.1, shape="moons", seed=7, density_imbalance=0.9
    )
    points = np.array(instance.points)
    labels = np.array(instance.true_labels)

    def mean_nn_distance(group_points):
        d = np.sqrt(((group_points[:, None, :] - group_points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1).mean()

    nn_group0 = mean_nn_distance(points[labels == 0])
    nn_group1 = mean_nn_distance(points[labels == 1])
    assert nn_group0 > nn_group1 * 1.5


def test_no_bridge_by_default_strength_zero():
    instance = generate_instance(n_points=100, k=4, spread=0.15, shape="blobs", seed=1, bridge_strength=0.0)
    assert -1 not in instance.true_labels


def test_bridge_points_lie_between_the_first_two_group_centers():
    instance = generate_instance(
        n_points=100, k=4, spread=0.15, shape="blobs", seed=2, bridge_strength=1.0
    )
    points = np.array(instance.points)
    labels = np.array(instance.true_labels)
    bridge_points = points[labels == -1]

    center_a = points[labels == 0].mean(axis=0)
    center_b = points[labels == 1].mean(axis=0)
    direction = center_b - center_a
    t = ((bridge_points - center_a) @ direction) / (direction @ direction)

    assert bridge_points.shape[0] > 0
    assert np.mean((t >= -0.15) & (t <= 1.15)) > 0.9


def test_bridge_works_for_moons_too():
    instance = generate_instance(
        n_points=100, k=2, spread=0.1, shape="moons", seed=2, bridge_strength=1.0
    )
    assert instance.true_labels.count(-1) == 80  # 100% von MAX_BRIDGE_POINTS (80)
