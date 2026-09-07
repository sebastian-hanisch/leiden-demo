"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die Leiden-Algorithmus-
(Modularitätsoptimierungs-)Demo."""

DEFAULT_N_POINTS = 120
DEFAULT_K = 4
DEFAULT_SPREAD = 0.12
DEFAULT_DENSITY_IMBALANCE = 0.0
DEFAULT_BRIDGE_STRENGTH = 0.0
DEFAULT_SEED = 1
DEFAULT_SHAPE = "blobs"
DEFAULT_N_NEIGHBORS = 8
DEFAULT_QUALITY_FUNCTION = "modularity"
DEFAULT_RESOLUTION = 1.0
DEFAULT_CPM_RESOLUTION = 0.002

N_POINTS_MIN, N_POINTS_MAX = 30, 300
K_MIN, K_MAX = 2, 20
SPREAD_MIN, SPREAD_MAX = 0.05, 0.9
DENSITY_IMBALANCE_MIN, DENSITY_IMBALANCE_MAX = 0.0, 1.0
BRIDGE_STRENGTH_MIN, BRIDGE_STRENGTH_MAX = 0.0, 1.0
# N_NEIGHBORS darf bis knapp an N_POINTS_MAX-1 heranreichen (ein vollstaendiger Graph) -
# die Aehnlichkeitsberechnung ist ohnehin schon O(n^2) unabhaengig von n_neighbors
# (dichte n x n-Matrix), und lokales Verschieben/Verfeinern bleibt selbst bei n_neighbors
# nahe N_POINTS_MAX empirisch weit unter einer spuerbaren Verzoegerung (~0.14s bei
# n_points=300, n_neighbors=299 - gemessen, nicht geschaetzt).
N_NEIGHBORS_MIN, N_NEIGHBORS_MAX = 2, 299
RESOLUTION_MIN, RESOLUTION_MAX = 0.3, 4.0
# CPMs resolution_parameter hat eine VOELLIG andere Skala als Modularitaets-gamma: es
# wird direkt gegen Kantengewichte verglichen (hier Gauss-Kernel-Werte in [0,1], typischer
# Median ~0.5-0.6), nicht gegen ein von der Graphgroesse abhaengiges Nullmodell. Empirisch
# ermittelt (gegen leidenalg.CPMVertexPartition, nicht geschaetzt): ~0.0005-0.004 liefert
# auf den Standard-Szenarien dieser Demo dieselbe "richtige" Partition wie Modularitaet
# bei gamma=1.0; ab ~0.5-0.6 loest sich sogar das Aufloesungslimit-Szenario (k=20) korrekt
# auf, was Modularitaet bei KEINEM gamma schafft (siehe Preset "Aufloesungslimit richtig
# behoben (CPM)"). 1.0 zersplittert bereits fast alles in Singletons.
CPM_RESOLUTION_MIN, CPM_RESOLUTION_MAX = 0.0, 1.0

SHAPES = ("blobs", "moons")
SHAPE_LABELS = {"blobs": "Gruppen (Blobs)", "moons": "Halbmonde"}

QUALITY_FUNCTIONS = ("modularity", "cpm")
QUALITY_FUNCTION_LABELS = {
    "modularity": "Modularität (Standard)",
    "cpm": "CPM (Constant Potts Model)",
}

RING_RADIUS = 3.0
ARC_RADIUS = 2.5
ARC_RING_RADIUS = 7.5
MAX_BRIDGE_POINTS = 80

# Sicherheitsgrenze fuer die naive O(n^2)-Aehnlichkeitsberechnung.
N_POINTS_HARD_MAX = 300

# Fester Vergleichs-Seed fuer den Methodenvergleich (kmeans-Baseline), unabhaengig vom
# Szenario-Seed (Lehre aus gmm-/dpmm-/spectral-/divisive-demo: Vergleichs-Randomness nie
# an den Szenario-Seed koppeln).
COMPARISON_SEED = 1

PRESETS = {
    "Einfaches Beispiel": {
        "n_points": 120, "k": 4, "spread": 0.12, "density_imbalance": 0.0, "bridge_strength": 0.0,
        "shape": "blobs", "n_neighbors": 8, "quality_function": "modularity", "resolution": 1.0, "seed": 1,
    },
    "Kein k nötig - viele Gruppen": {
        "n_points": 200, "k": 8, "spread": 0.1, "density_imbalance": 0.0, "bridge_strength": 0.0,
        "shape": "blobs", "n_neighbors": 8, "quality_function": "modularity", "resolution": 1.0, "seed": 2,
    },
    "Auflösungsgrenze": {
        "n_points": 60, "k": 20, "spread": 0.05, "density_imbalance": 0.0, "bridge_strength": 0.0,
        "shape": "blobs", "n_neighbors": 4, "quality_function": "modularity", "resolution": 1.0, "seed": 1,
    },
    "Auflösungsparameter als Kompromiss": {
        "n_points": 60, "k": 20, "spread": 0.05, "density_imbalance": 0.0, "bridge_strength": 0.0,
        "shape": "blobs", "n_neighbors": 4, "quality_function": "modularity", "resolution": 3.0, "seed": 1,
    },
    "Kombinierter Härtefall (Dichte + Brücke)": {
        "n_points": 120, "k": 4, "spread": 0.12, "density_imbalance": 0.6, "bridge_strength": 0.3,
        "shape": "blobs", "n_neighbors": 8, "quality_function": "modularity", "resolution": 1.0, "seed": 1,
    },
    "Auflösungslimit richtig behoben (CPM)": {
        "n_points": 60, "k": 20, "spread": 0.05, "density_imbalance": 0.0, "bridge_strength": 0.0,
        "shape": "blobs", "n_neighbors": 4, "quality_function": "cpm", "resolution": 0.5, "seed": 1,
    },
}
