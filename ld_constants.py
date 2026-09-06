"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die Leiden-Algorithmus-
(Modularitätsoptimierungs-)Demo."""

DEFAULT_N_POINTS = 120
DEFAULT_K = 4
DEFAULT_SPREAD = 0.12
DEFAULT_SEED = 1
DEFAULT_SHAPE = "blobs"
DEFAULT_N_NEIGHBORS = 8
DEFAULT_RESOLUTION = 1.0

N_POINTS_MIN, N_POINTS_MAX = 30, 300
K_MIN, K_MAX = 2, 20
SPREAD_MIN, SPREAD_MAX = 0.05, 0.9
N_NEIGHBORS_MIN, N_NEIGHBORS_MAX = 2, 30
RESOLUTION_MIN, RESOLUTION_MAX = 0.3, 4.0

SHAPES = ("blobs", "moons")
SHAPE_LABELS = {"blobs": "Gruppen (Blobs)", "moons": "Halbmonde"}

RING_RADIUS = 3.0
ARC_RADIUS = 2.5
ARC_RING_RADIUS = 7.5

# Sicherheitsgrenze fuer die naive O(n^2)-Aehnlichkeitsberechnung.
N_POINTS_HARD_MAX = 300

# Fester Vergleichs-Seed fuer den Methodenvergleich (kmeans-Baseline), unabhaengig vom
# Szenario-Seed (Lehre aus gmm-/dpmm-/spectral-/divisive-demo: Vergleichs-Randomness nie
# an den Szenario-Seed koppeln).
COMPARISON_SEED = 1

PRESETS = {
    "Einfaches Beispiel": {
        "n_points": 120, "k": 4, "spread": 0.12, "shape": "blobs",
        "n_neighbors": 8, "resolution": 1.0, "seed": 1,
    },
    "Kein k nötig - viele Gruppen": {
        "n_points": 200, "k": 8, "spread": 0.1, "shape": "blobs",
        "n_neighbors": 8, "resolution": 1.0, "seed": 2,
    },
    "Auflösungsgrenze": {
        "n_points": 60, "k": 20, "spread": 0.05, "shape": "blobs",
        "n_neighbors": 4, "resolution": 1.0, "seed": 1,
    },
    "Auflösungsparameter als Kompromiss": {
        "n_points": 60, "k": 20, "spread": 0.05, "shape": "blobs",
        "n_neighbors": 4, "resolution": 3.0, "seed": 1,
    },
}
