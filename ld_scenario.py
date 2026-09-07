"""Zufällige 2D-Punktwolken für die Leiden-Demo: entweder k Gauß-Cluster auf einem Ring
("blobs", wie kmeans-demo/spectral-demo) oder k nicht-konvexe Halbkreis-Bögen ("moons") -
bei k=2 das klassische Zwei-Halbmonde-Beispiel, bei k>2 wie Blütenblätter auf einem Ring
angeordnet (dieselbe Generalisierung wie in dbscan-demo/spectral-demo, hier frisch
nachgebaut - kein Cross-Repo-Import). Anders als spectral-demo erlaubt `k` hier deutlich
höhere Werte (bis 20), das Vehikel für die Auflösungslimit-Szenarien.

Zusätzlich dieselben Konstruktionsparameter wie hdbscan-demo: density_imbalance
(dbscan-demo-Mechanik, Gruppe 0 diffuser bei gleicher Punktzahl) UND bridge_strength
(agglomerative-demo-Mechanik, Punktbrücke zwischen den ersten beiden Gruppen) - beide
unabhängig einstellbar, um zu zeigen, ob Leidens kNN-Graph-Ansatz auf dieselbe Weise wie
DBSCAN/Single-Linkage an diesen beiden klassischen Härtefällen scheitert oder nicht."""

from dataclasses import dataclass

import numpy as np

from ld_constants import ARC_RADIUS, ARC_RING_RADIUS, MAX_BRIDGE_POINTS, RING_RADIUS

MIN_STD_FRACTION = 0.05


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple
    true_labels: tuple  # Gruppenindex, oder -1 fuer Brueckenpunkte (gehoeren zu keiner Gruppe)
    shape: str
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _cluster_stds(k, density_imbalance, base_std):
    """Wie dbscan-demo/hdbscan-demo: Gruppe 0 wird mit wachsendem density_imbalance
    diffuser (groessere Streuung bei gleicher Punktzahl, also geringere lokale Dichte),
    die uebrigen entsprechend enger."""
    stds = np.full(k, base_std)
    if k > 1:
        stds[0] *= 1 + density_imbalance
        stds[1:] *= max(1 - 0.6 * density_imbalance, MIN_STD_FRACTION)
    return stds


def _generate_blobs(n_points, k, spread, density_imbalance, rng):
    """Fester Ring-Radius, UNABHAENGIG von k - wie in allen Geschwister-Demos (kmeans-demo,
    spectral-demo etc. skalieren ihren Blob-Ring-Radius ebenfalls nicht mit k). Hier ist
    das aber besonders wichtig, bewusst so belassen und nicht "korrigiert": bei hohem k
    ruecken die Gruppen dadurch enger zusammen, was ueberhaupt erst den fuer diese Demo
    zentralen Auflösungslimit-Effekt ermoeglicht (siehe project-memory) - waeren die
    Gruppen immer proportional weiter auseinander, koennte Modularitaetsoptimierung sie
    nie faelschlich verschmelzen. (Die MOONS-Variante unten skaliert ihren Ring-Radius bei
    k>2 dagegen mit k - dort sind die Elemente ganze Bögen statt Punkte, die bei hohem k
    sonst überlappen würden, siehe `_generate_moons`.)"""
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS
    stds = _cluster_stds(k, density_imbalance, base_std)

    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()

    points_per_group, labels_per_group = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=stds[i], size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))
    points = np.concatenate(points_per_group, axis=0)
    labels = np.concatenate(labels_per_group, axis=0)
    return points, labels, centers


def _generate_moons(n_points, k, spread, density_imbalance, rng):
    """k=2: klassisches Zwei-Halbmonde-Beispiel. k>2: k Halbkreis-Bögen wie
    Blütenblätter auf einem Ring, konkave Seite zum Zentrum. density_imbalance wirkt wie
    bei "blobs": Gruppe 0 wird diffuser, die uebrigen entsprechend enger."""
    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()
    base_noise_std = max(spread, MIN_STD_FRACTION) * ARC_RADIUS * 0.3
    noise_stds = _cluster_stds(k, density_imbalance, base_noise_std)

    if k == 2:
        t1 = rng.uniform(0, np.pi, counts[0])
        x1, y1 = ARC_RADIUS * np.cos(t1), ARC_RADIUS * np.sin(t1)
        t2 = rng.uniform(0, np.pi, counts[1])
        x2, y2 = ARC_RADIUS * (1 - np.cos(t2)), ARC_RADIUS * (0.5 - np.sin(t2))
        pts1 = np.stack([x1, y1], axis=1) + rng.normal(scale=noise_stds[0], size=(counts[0], 2))
        pts2 = np.stack([x2, y2], axis=1) + rng.normal(scale=noise_stds[1], size=(counts[1], 2))
        points = np.concatenate([pts1, pts2], axis=0)
        labels = np.concatenate([np.zeros(counts[0], dtype=int), np.ones(counts[1], dtype=int)])
        centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
        return points, labels, centers

    ring_radius = ARC_RING_RADIUS * max(1.0, k / 6.0)
    layout_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.1, 0.1, size=k)
    arc_centers = np.stack(
        [ring_radius * np.cos(layout_angles), ring_radius * np.sin(layout_angles)], axis=1
    )

    points_per_group, labels_per_group = [], []
    for i in range(k):
        t = rng.uniform(0, np.pi, counts[i])
        local_x, local_y = ARC_RADIUS * np.cos(t), ARC_RADIUS * np.sin(t)
        rot = layout_angles[i] + np.pi
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rx = cos_r * local_x - sin_r * local_y
        ry = sin_r * local_x + cos_r * local_y
        pts = np.stack([rx, ry], axis=1) + arc_centers[i] + rng.normal(scale=noise_stds[i], size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))
    points = np.concatenate(points_per_group, axis=0)
    labels = np.concatenate(labels_per_group, axis=0)
    centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
    return points, labels, centers


def _add_bridge(points, labels, center_a, center_b, bridge_strength, rng):
    """Wie agglomerative-demo/hdbscan-demo: zusaetzliche Punkte entlang der vollen
    Verbindungslinie zwischen zwei Zentren - das Vehikel, um zu pruefen, ob eine duenne
    Punktbruecke Leidens Graph-Community-Erkennung genauso faelschlich verschmelzen laesst
    wie Single-Linkage-Chaining."""
    n_bridge = int(round(bridge_strength * MAX_BRIDGE_POINTS))
    if n_bridge <= 0:
        return points, labels
    t = rng.uniform(0.0, 1.0, n_bridge)
    base = center_a[None, :] + t[:, None] * (center_b - center_a)[None, :]
    jitter_std = 0.08 * RING_RADIUS
    bridge_points = base + rng.normal(scale=jitter_std, size=base.shape)
    bridge_labels = np.full(n_bridge, -1)
    return np.concatenate([points, bridge_points]), np.concatenate([labels, bridge_labels])


def generate_instance(n_points, k, spread, shape, seed, density_imbalance=0.0, bridge_strength=0.0):
    """spread steuert je nach shape entweder die Cluster-Streuung ("blobs") oder das
    Rauschen um die ideale Bogen-Kurve ("moons"). k wirkt bei BEIDEN Formen und darf hier
    (anders als bei spectral-demo) auch hohe Werte annehmen - die Ring-/Bogen-Anordnung
    skaliert ihren Radius mit k mit, damit auch viele Gruppen sichtbar getrennt bleiben.
    density_imbalance und bridge_strength wirken bei BEIDEN Formen, wie in hdbscan-demo."""
    rng = np.random.default_rng(seed)

    if shape == "moons":
        points, labels, centers = _generate_moons(n_points, k, spread, density_imbalance, rng)
    else:
        points, labels, centers = _generate_blobs(n_points, k, spread, density_imbalance, rng)

    if k >= 2 and bridge_strength > 0:
        points, labels = _add_bridge(points, labels, centers[0], centers[1], bridge_strength, rng)

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        shape=shape,
        k=k,
    )
