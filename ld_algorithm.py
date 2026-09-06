"""Leiden-Algorithmus (Traag, Waltman & van Eck, 2019) from scratch:
Ähnlichkeitsgraph (identischer Aufbau wie spectral-demo, frisch nachgebaut) →
wiederholte Pässe aus lokalem Verschieben, Verfeinerung und Aggregation, bis die
Modularität nicht mehr steigt. Bewusst ohne igraph/leidenalg zur Laufzeit implementiert.
`leidenalg`/`python-igraph` dienen in tests/ nur als unabhängiger Kreuzvergleich,
`networkx` nur für einen exakten Modularitäts-Formel-Kreuzvergleich."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pass:
    level: int  # 0-indexiert, ein Eintrag je vollstaendigem Verschieben+Verfeinern+Aggregieren-Pass
    partition: tuple  # Labels fuer ALLE urspruenglichen Punkte (0..k-1, fortlaufend), Stand nach diesem Pass
    modularity: float  # Q_gamma des Gesamtgraphen bei dieser Partition
    n_communities: int


@dataclass(frozen=True)
class RunResult:
    passes: tuple  # chronologische Pass-Eintraege
    resolution: float

    @property
    def final_labels(self):
        return self.passes[-1].partition

    @property
    def final_modularity(self):
        return self.passes[-1].modularity

    @property
    def n_communities(self):
        return self.passes[-1].n_communities


def build_similarity_graph(data, n_neighbors):
    """k-naechste-Nachbarn-Graph (ohne Selbsteinschluss), symmetrisiert ueber die
    "ODER"-Regel, Gauss-Kernel-Gewichte mit Median-Distanz-Bandbreite - identischer
    Aufbau wie spectral-demo/sp_algorithm.py."""
    n = len(data)
    dist = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))

    order = np.argsort(dist, axis=1)
    neighbor_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        nn = order[i, 1 : n_neighbors + 1]
        neighbor_mask[i, nn] = True
    adjacency = neighbor_mask | neighbor_mask.T
    np.fill_diagonal(adjacency, False)

    retained_distances = dist[adjacency]
    sigma = np.median(retained_distances) if retained_distances.size > 0 else 1.0
    sigma = max(sigma, 1e-6)

    weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    similarity = np.where(adjacency, weights, 0.0)
    return similarity


def modularity(adjacency, labels, resolution=1.0):
    """Q_gamma = (1/2m) * sum_ij [A_ij - gamma*k_i*k_j/(2m)] * delta(c_i,c_j), exakt wie
    networkx.algorithms.community.quality.modularity (siehe Kreuzvergleich-Test)."""
    adjacency = np.asarray(adjacency, dtype=float)
    labels = np.asarray(labels)
    degrees = adjacency.sum(axis=1)
    m2 = degrees.sum()
    if m2 <= 0:
        return 0.0

    total = 0.0
    for c in set(labels.tolist()):
        idx = np.where(labels == c)[0]
        internal = adjacency[np.ix_(idx, idx)].sum()
        deg_sum = degrees[idx].sum()
        total += internal - resolution * (deg_sum ** 2) / m2
    return float(total / m2)


def _relabel(labels):
    labels = np.asarray(labels)
    unique = sorted(set(labels.tolist()))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[l] for l in labels])


def local_moving(adjacency, resolution, rng, max_sweeps=100):
    """Phase 1: ausgehend von Singleton-Communities verschiebt jeder Knoten sich gierig
    in die Nachbar-Community mit dem groessten Modularitaetsgewinn
    ($k_{i,\\text{in}}(C) - \\gamma \\cdot \\Sigma_{\\text{tot}}(C) \\cdot k_i / 2m$,
    die Standard-Louvain-Gewinnformel), bis eine volle Runde ueber alle Knoten keine
    Verschiebung mehr bringt."""
    n = len(adjacency)
    degrees = adjacency.sum(axis=1)
    m2 = degrees.sum()
    if m2 <= 0:
        return np.arange(n)

    neighbor_lists = []
    for i in range(n):
        nz = np.nonzero(adjacency[i])[0]
        neighbor_lists.append([(int(j), float(adjacency[i, j])) for j in nz if j != i])

    community = np.arange(n)
    comm_degree_sum = degrees.copy()

    order = list(range(n))
    for _sweep in range(max_sweeps):
        improved = False
        rng.shuffle(order)
        for i in order:
            current_comm = community[i]
            comm_degree_sum[current_comm] -= degrees[i]

            neighbor_weights = {}
            for j, w in neighbor_lists[i]:
                c = community[j]
                neighbor_weights[c] = neighbor_weights.get(c, 0.0) + w

            best_comm = current_comm
            best_gain = neighbor_weights.get(current_comm, 0.0) - resolution * comm_degree_sum[current_comm] * degrees[i] / m2

            for c, w_ic in neighbor_weights.items():
                gain = w_ic - resolution * comm_degree_sum[c] * degrees[i] / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = c

            comm_degree_sum[best_comm] += degrees[i]
            if best_comm != current_comm:
                community[i] = best_comm
                improved = True
        if not improved:
            break

    return community


def refine(adjacency, local_labels, resolution, rng):
    """Phase 2 (der entscheidende Unterschied zu Louvain): innerhalb jeder in Phase 1
    gefundenen Community wird - ausgehend von Singletons, aber mit der GLEICHEN
    global normalisierten Modularitaets-Gewinnformel wie Phase 1 (nicht neu abgeleitet
    aus dem isolierten Teilgraphen - das wuerde durch eine ganz andere Nullmodell-
    Normierung systematisch zu viel aufsplitten) - erneut lokal verschoben, aber nur
    zwischen Kandidaten INNERHALB derselben Phase-1-Community. Da dabei ausschliesslich
    ueber tatsaechlich vorhandene Kanten gemergt wird, ist jede entstehende verfeinerte
    Community GARANTIERT zusammenhaengend (jeder Knoten ist ueber eine Kette von
    Kanten-Merges mit jedem anderen Knoten seiner Community verbunden) - das behebt
    Louvains bekannten Fehler, nach Aggregation eine nicht-zusammenhaengende Community
    liefern zu koennen (siehe tests/test_algorithm.py: `test_all_communities_are_connected`
    beweist das fuer den vollen Algorithmus direkt, `test_refinement_can_split_a_local_moving_community`
    zeigt, dass diese Phase auf echten Szenarien tatsaechlich etwas veraendert)."""
    n = len(adjacency)
    degrees = adjacency.sum(axis=1)
    m2 = degrees.sum()
    if m2 <= 0:
        return np.arange(n)

    refined = np.arange(n)
    comm_degree_sum = degrees.copy()

    for comm in sorted(set(local_labels.tolist())):
        idx = np.where(local_labels == comm)[0]
        if len(idx) <= 1:
            continue
        idx_set = set(int(i) for i in idx)
        order = list(idx)
        rng.shuffle(order)

        for _sweep in range(100):
            improved = False
            rng.shuffle(order)
            for i in order:
                current = refined[i]
                comm_degree_sum[current] -= degrees[i]

                neighbor_weights = {}
                for j in np.nonzero(adjacency[i])[0]:
                    j = int(j)
                    if j != i and j in idx_set:
                        c = refined[j]
                        neighbor_weights[c] = neighbor_weights.get(c, 0.0) + adjacency[i, j]

                best = current
                best_gain = neighbor_weights.get(current, 0.0) - resolution * comm_degree_sum[current] * degrees[i] / m2
                for c, w in neighbor_weights.items():
                    gain = w - resolution * comm_degree_sum[c] * degrees[i] / m2
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best = c

                comm_degree_sum[best] += degrees[i]
                if best != current:
                    refined[i] = best
                    improved = True
            if not improved:
                break

    return refined


def aggregate_graph(adjacency, labels):
    """Phase 3: baut den reduzierten Graphen - ein Knoten je Community, Kantengewicht =
    Summe der Inter-Community-Gewichte (Selbstschleifen = doppelte interne Kantensumme,
    konsistent mit der ueber ALLE (i,j)-Paare summierenden Modularitaetsformel)."""
    labels = _relabel(labels)
    m = int(labels.max()) + 1
    new_adjacency = np.zeros((m, m))
    n = len(adjacency)
    for i in range(n):
        ci = labels[i]
        row = adjacency[i]
        nz = np.nonzero(row)[0]
        for j in nz:
            new_adjacency[ci, labels[j]] += row[j]
    return new_adjacency, labels


def community_is_connected(adjacency, indices):
    """Prueft per Breitensuche, ob der von `indices` induzierte Teilgraph zusammenhaengend
    ist - die Kerneigenschaft, die Leidens Verfeinerungsphase garantiert und Louvains
    reines lokales Verschieben (ohne Verfeinerung) verletzen kann."""
    indices = list(indices)
    if len(indices) <= 1:
        return True
    idx_set = set(indices)
    visited = {indices[0]}
    stack = [indices[0]]
    while stack:
        node = stack.pop()
        for neighbor in np.nonzero(adjacency[node])[0]:
            neighbor = int(neighbor)
            if neighbor in idx_set and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited == idx_set


def _run_passes(base_adjacency, resolution, seed, use_refinement, max_levels=20):
    n = len(base_adjacency)
    rng = np.random.default_rng(seed)

    current_membership = np.arange(n)
    current_adjacency = base_adjacency.copy()
    passes = []

    for level in range(max_levels):
        local_labels = local_moving(current_adjacency, resolution, rng)
        working_labels = refine(current_adjacency, local_labels, resolution, rng) if use_refinement else local_labels
        working_labels = _relabel(working_labels)

        new_membership = _relabel(working_labels[current_membership])
        q = modularity(base_adjacency, new_membership, resolution)
        passes.append(
            Pass(level, tuple(int(x) for x in new_membership), q, int(new_membership.max()) + 1)
        )

        n_communities_this_level = int(working_labels.max()) + 1
        if n_communities_this_level == len(current_adjacency):
            break

        current_adjacency, _ = aggregate_graph(current_adjacency, working_labels)
        current_membership = new_membership
        if len(current_adjacency) == 1:
            break

    return tuple(passes)


def run(data, n_neighbors, resolution, seed, max_levels=20):
    """Fuehrt den vollstaendigen Leiden-Algorithmus (mit Verfeinerung) auf dem aus den
    Rohdaten gebauten Aehnlichkeitsgraphen aus - Pass fuer Pass protokolliert, damit die
    App den Fortschritt Level fuer Level durchblaettern kann. Braucht anders als
    spectral-demo/k-Means/GMM/... KEINE Ziel-Clusteranzahl irgendwo."""
    data = np.asarray(data, dtype=float)
    base_adjacency = build_similarity_graph(data, n_neighbors)
    passes = _run_passes(base_adjacency, resolution, seed, use_refinement=True, max_levels=max_levels)
    return RunResult(passes=passes, resolution=resolution)


def run_local_moving_only(data, n_neighbors, resolution, seed, max_levels=20):
    """Testeigene Ablation: lokales Verschieben OHNE Verfeinerung (Louvains Verhalten
    nachgebaut) - ausschliesslich als Kontrastfolie in tests/ (zeigt, dass die
    Verfeinerung nicht Teil des noetigen Minimalpfads ist, sondern eine bewusste
    Zusatzgarantie), kein Teil des in der App gezeigten Algorithmus."""
    data = np.asarray(data, dtype=float)
    base_adjacency = build_similarity_graph(data, n_neighbors)
    passes = _run_passes(base_adjacency, resolution, seed, use_refinement=False, max_levels=max_levels)
    return RunResult(passes=passes, resolution=resolution)
