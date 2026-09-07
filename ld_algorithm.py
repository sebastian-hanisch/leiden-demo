"""Leiden-Algorithmus (Traag, Waltman & van Eck, 2019) from scratch:
Ähnlichkeitsgraph (identischer Aufbau wie spectral-demo, frisch nachgebaut) →
wiederholte Pässe aus lokalem Verschieben, Verfeinerung und Aggregation, bis die
Qualitätsfunktion nicht mehr steigt. Zwei austauschbare Qualitätsfunktionen, dieselbe
Leiden-Maschinerie fuer beide:

- **Modularität** (Standard, Newman 2004): vergleicht Kantendichte je Community gegen ein
  Nullmodell, das von der GESAMTEN Graphgröße abhängt (k_i*k_j/2m) - das ist exakt die
  Ursache des Auflösungslimits (Fortunato & Barthélemy, 2007), das diese Demo an anderer
  Stelle zeigt: kleine, klar getrennte Communities werden verschmolzen, weil ihr Beitrag
  zum globalen Nullmodell verschwindend klein ist, unabhängig davon, wie klar getrennt sie
  tatsächlich sind.
- **CPM** (Constant Potts Model, Traag, Van Dooren & Nesterov, 2011, "Narrow scope for
  resolution-limit-free community detection", Physical Review E 84, 016114): vergleicht
  Kantendichte je Community gegen einen FESTEN Schwellenwert `resolution` je Knotenpaar,
  unabhängig von der Gesamtgraphgröße - dadurch nachweislich frei vom Auflösungslimit
  (nicht nur gelindert wie bei Modularität mit hohem γ). Der Preis: `resolution` hat eine
  völlig andere Skala als Modularitäts-γ (hier typischerweise 0.0-1.0 statt 0.3-4.0, siehe
  ld_constants.py) und muss auf die Kantengewichts-Skala des Graphen abgestimmt werden -
  kein Free Lunch, nur ein anderer Kompromiss.

Bewusst ohne igraph/leidenalg zur Laufzeit implementiert. `leidenalg`/`python-igraph`
dienen in tests/ nur als unabhängiger Kreuzvergleich (fuer BEIDE Qualitätsfunktionen,
leidenalg unterstuetzt CPM nativ ueber `CPMVertexPartition`), `networkx` nur für einen
exakten Modularitäts-Formel-Kreuzvergleich (networkx kennt kein CPM).

Zusätzlich **Konsensus-Clustering** (Lancichinetti & Fortunato, 2012, "Consensus
clustering in complex networks", Scientific Reports 2, 336): lokales Verschieben
besucht Knoten in zufälliger Reihenfolge, wodurch verschiedene Seeds - besonders bei
dichten oder stark überlappenden Szenarien - unterschiedliche (wenn auch ähnlich gute)
lokale Optima liefern können. `consensus_clustering` führt Leiden `n_runs`-mal
unabhängig aus, baut daraus eine Konsensus-Matrix (Anteil der Läufe, in denen zwei
Punkte zusammen landeten) und behandelt diese Matrix als NEUEN gewichteten Graphen für
eine erneute Runde - wiederholt, bis die Matrix "kristallklar" wird (jeder Eintrag 0
oder 1, d.h. alle Läufe liefern exakt dieselbe Partition). Ersetzt die Abhängigkeit vom
Zufall der Besuchsreihenfolge durch eine explizite Mehrheitsentscheidung über viele
unabhängige Läufe - funktioniert mit BEIDEN Qualitätsfunktionen, da es nur auf den
jeweils zurückgegebenen Partitionen aufbaut, nicht auf deren internen Details."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pass:
    level: int  # 0-indexiert, ein Eintrag je vollstaendigem Verschieben+Verfeinern+Aggregieren-Pass
    partition: tuple  # Labels fuer ALLE urspruenglichen Punkte (0..k-1, fortlaufend), Stand nach diesem Pass
    quality: float  # Wert der GEWAEHLTEN Qualitaetsfunktion (Modularitaet ODER CPM) bei dieser Partition
    n_communities: int


@dataclass(frozen=True)
class RunResult:
    passes: tuple  # chronologische Pass-Eintraege
    resolution: float
    quality_function: str = "modularity"

    @property
    def final_labels(self):
        return self.passes[-1].partition

    @property
    def final_quality(self):
        return self.passes[-1].quality

    @property
    def n_communities(self):
        return self.passes[-1].n_communities


@dataclass(frozen=True)
class ConsensusResult:
    final_labels: tuple  # Labels (0..k-1, fortlaufend) der finalen Konsensus-Partition
    n_communities: int
    n_rounds: int  # Anzahl Konsensus-Runden bis zur Konvergenz (inkl. der allerersten)
    converged: bool  # True, wenn eine "kristallklare" (nur 0/1) Konsensus-Matrix erreicht wurde
    single_run_agreement: float  # mittlere paarweise Punktpaar-Uebereinstimmung der n_runs
    # unabhaengigen EINZEL-Laeufe VOR jeder Konsensus-Bildung - misst, wie stark das
    # Ergebnis ohne Konsensus vom Zufalls-Seed abhaengt (1.0 = alle Laeufe identisch)


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


def cpm_quality(adjacency, labels, resolution=1.0, node_weights=None):
    """H_CPM = sum_c [e_c - gamma * n_c*(n_c-1)/2] (Traag, Van Dooren & Nesterov, 2011).
    `e_c` ist die Summe der INTERNEN Kantengewichte einer Community (jede Kante einfach
    gezaehlt), `n_c` ihre Knotenanzahl. Anders als bei Modularitaet skaliert der
    Referenzwert NICHT mit der Gesamtgraphgroesse - das macht CPM nachweislich frei vom
    Aufloesungslimit. `node_weights` (Standard: 1 je Knoten) erlaubt es, einen aggregierten
    Knoten (der mehrere urspruengliche Punkte vertritt) korrekt mit seiner tatsaechlichen
    Groesse statt mit 1 zu gewichten - siehe `aggregate_graph`."""
    adjacency = np.asarray(adjacency, dtype=float)
    labels = np.asarray(labels)
    n = len(adjacency)
    if node_weights is None:
        node_weights = np.ones(n)
    node_weights = np.asarray(node_weights, dtype=float)

    total = 0.0
    for c in set(labels.tolist()):
        idx = np.where(labels == c)[0]
        internal_weight = adjacency[np.ix_(idx, idx)].sum() / 2.0
        nc = node_weights[idx].sum()
        total += internal_weight - resolution * nc * (nc - 1) / 2.0
    return float(total)


def quality(adjacency, labels, resolution=1.0, quality_function="modularity"):
    """Dispatcher: wertet die GEWAEHLTE Qualitaetsfunktion auf der Basis-Granularität
    (urspruengliche Punkte, node_weights=1) aus - fuer die Zwischenzustaende waehrend
    der Aggregation siehe die node_weights-Parameter von local_moving/refine direkt."""
    if quality_function == "cpm":
        return cpm_quality(adjacency, labels, resolution)
    return modularity(adjacency, labels, resolution)


def _relabel(labels):
    labels = np.asarray(labels)
    unique = sorted(set(labels.tolist()))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[l] for l in labels])


def local_moving(adjacency, resolution, rng, quality_function="modularity", node_weights=None, max_sweeps=100):
    """Phase 1: ausgehend von Singleton-Communities verschiebt jeder Knoten sich gierig
    in die Nachbar-Community mit dem groessten Qualitaetsgewinn, bis eine volle Runde
    ueber alle Knoten keine Verschiebung mehr bringt. Bei Modularitaet ist das die
    Standard-Louvain-Gewinnformel
    ($k_{i,\\text{in}}(C) - \\gamma \\cdot \\Sigma_{\\text{tot}}(C) \\cdot k_i / 2m$); bei
    CPM die entsprechende Herleitung fuer H_CPM
    ($w_{i,\\text{in}}(C) - \\gamma \\cdot n_C \\cdot s_i$, wobei $s_i$ die "Groesse" von
    Knoten $i$ ist - 1 fuer einen urspruenglichen Punkt, sonst die Anzahl urspruenglicher
    Punkte, die ein aggregierter Knoten vertritt: einen Knoten der Groesse $s_i$ in eine
    Community der Groesse $n_C$ einzufuegen erhoeht die Paar-Strafe um
    $\\binom{n_C+s_i}{2} - \\binom{n_C}{2} = n_C \\cdot s_i + \\binom{s_i}{2}$, wobei der
    von $C$ unabhaengige zweite Term beim Vergleich der Communities herausfaellt)."""
    n = len(adjacency)
    degrees = adjacency.sum(axis=1)
    m2 = degrees.sum()
    if quality_function != "cpm" and m2 <= 0:
        return np.arange(n)
    if node_weights is None:
        node_weights = np.ones(n)
    node_weights = np.asarray(node_weights, dtype=float)

    neighbor_lists = []
    for i in range(n):
        nz = np.nonzero(adjacency[i])[0]
        neighbor_lists.append([(int(j), float(adjacency[i, j])) for j in nz if j != i])

    community = np.arange(n)
    comm_degree_sum = degrees.copy()
    comm_size = node_weights.copy()

    def null_cost(c, i):
        if quality_function == "cpm":
            return resolution * comm_size[c] * node_weights[i]
        return resolution * comm_degree_sum[c] * degrees[i] / m2

    order = list(range(n))
    for _sweep in range(max_sweeps):
        improved = False
        rng.shuffle(order)
        for i in order:
            current_comm = community[i]
            comm_degree_sum[current_comm] -= degrees[i]
            comm_size[current_comm] -= node_weights[i]

            neighbor_weights = {}
            for j, w in neighbor_lists[i]:
                c = community[j]
                neighbor_weights[c] = neighbor_weights.get(c, 0.0) + w

            best_comm = current_comm
            best_gain = neighbor_weights.get(current_comm, 0.0) - null_cost(current_comm, i)

            for c, w_ic in neighbor_weights.items():
                gain = w_ic - null_cost(c, i)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = c

            comm_degree_sum[best_comm] += degrees[i]
            comm_size[best_comm] += node_weights[i]
            if best_comm != current_comm:
                community[i] = best_comm
                improved = True
        if not improved:
            break

    return community


def refine(adjacency, local_labels, resolution, rng, quality_function="modularity", node_weights=None):
    """Phase 2 (der entscheidende Unterschied zu Louvain): innerhalb jeder in Phase 1
    gefundenen Community wird - ausgehend von Singletons, aber mit der GLEICHEN
    global normalisierten Gewinnformel wie Phase 1 (nicht neu abgeleitet aus dem
    isolierten Teilgraphen - das wuerde durch eine ganz andere Nullmodell-Normierung
    systematisch zu viel aufsplitten) - erneut lokal verschoben, aber nur zwischen
    Kandidaten INNERHALB derselben Phase-1-Community. Da dabei ausschliesslich ueber
    tatsaechlich vorhandene Kanten gemergt wird, ist jede entstehende verfeinerte
    Community GARANTIERT zusammenhaengend (jeder Knoten ist ueber eine Kette von
    Kanten-Merges mit jedem anderen Knoten seiner Community verbunden) - das behebt
    Louvains bekannten Fehler, nach Aggregation eine nicht-zusammenhaengende Community
    liefern zu koennen (siehe tests/test_algorithm.py: `test_all_communities_are_connected`
    beweist das fuer den vollen Algorithmus direkt, `test_refinement_can_split_a_local_moving_community`
    zeigt, dass diese Phase auf echten Szenarien tatsaechlich etwas veraendert)."""
    n = len(adjacency)
    degrees = adjacency.sum(axis=1)
    m2 = degrees.sum()
    if quality_function != "cpm" and m2 <= 0:
        return np.arange(n)
    if node_weights is None:
        node_weights = np.ones(n)
    node_weights = np.asarray(node_weights, dtype=float)

    refined = np.arange(n)
    comm_degree_sum = degrees.copy()
    comm_size = node_weights.copy()

    def null_cost(c, i):
        if quality_function == "cpm":
            return resolution * comm_size[c] * node_weights[i]
        return resolution * comm_degree_sum[c] * degrees[i] / m2

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
                comm_size[current] -= node_weights[i]

                neighbor_weights = {}
                for j in np.nonzero(adjacency[i])[0]:
                    j = int(j)
                    if j != i and j in idx_set:
                        c = refined[j]
                        neighbor_weights[c] = neighbor_weights.get(c, 0.0) + adjacency[i, j]

                best = current
                best_gain = neighbor_weights.get(current, 0.0) - null_cost(current, i)
                for c, w in neighbor_weights.items():
                    gain = w - null_cost(c, i)
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best = c

                comm_degree_sum[best] += degrees[i]
                comm_size[best] += node_weights[i]
                if best != current:
                    refined[i] = best
                    improved = True
            if not improved:
                break

    return refined


def aggregate_graph(adjacency, labels, node_weights=None):
    """Phase 3: baut den reduzierten Graphen - ein Knoten je Community, Kantengewicht =
    Summe der Inter-Community-Gewichte (Selbstschleifen = doppelte interne Kantensumme,
    konsistent mit der ueber ALLE (i,j)-Paare summierenden Modularitaetsformel). Gibt
    zusaetzlich die aggregierten `node_weights` zurueck (Summe je Community) - fuer
    Modularitaet ungenutzt (die aggregiert bereits korrekt ueber die Kantengewichte
    selbst), aber fuer CPM ZWINGEND: ohne sie wuerde jeder aggregierte Knoten faelschlich
    wieder mit Groesse 1 in die naechste Ebene starten, statt mit der tatsaechlichen
    Anzahl urspruenglicher Punkte, die er vertritt."""
    labels = _relabel(labels)
    m = int(labels.max()) + 1
    new_adjacency = np.zeros((m, m))
    n = len(adjacency)
    if node_weights is None:
        node_weights = np.ones(n)
    node_weights = np.asarray(node_weights, dtype=float)
    new_node_weights = np.zeros(m)
    for i in range(n):
        ci = labels[i]
        new_node_weights[ci] += node_weights[i]
        row = adjacency[i]
        nz = np.nonzero(row)[0]
        for j in nz:
            new_adjacency[ci, labels[j]] += row[j]
    return new_adjacency, labels, new_node_weights


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


def _run_passes(base_adjacency, resolution, seed, use_refinement, quality_function="modularity", max_levels=20):
    n = len(base_adjacency)
    rng = np.random.default_rng(seed)

    current_membership = np.arange(n)
    current_adjacency = base_adjacency.copy()
    current_node_weights = np.ones(n)
    passes = []

    for level in range(max_levels):
        local_labels = local_moving(
            current_adjacency, resolution, rng, quality_function=quality_function, node_weights=current_node_weights
        )
        working_labels = (
            refine(
                current_adjacency, local_labels, resolution, rng,
                quality_function=quality_function, node_weights=current_node_weights,
            )
            if use_refinement else local_labels
        )
        working_labels = _relabel(working_labels)

        new_membership = _relabel(working_labels[current_membership])
        q = quality(base_adjacency, new_membership, resolution, quality_function)
        passes.append(
            Pass(level, tuple(int(x) for x in new_membership), q, int(new_membership.max()) + 1)
        )

        n_communities_this_level = int(working_labels.max()) + 1
        if n_communities_this_level == len(current_adjacency):
            break

        current_adjacency, _, current_node_weights = aggregate_graph(
            current_adjacency, working_labels, current_node_weights
        )
        current_membership = new_membership
        if len(current_adjacency) == 1:
            break

    return tuple(passes)


def run(data, n_neighbors, resolution, seed, quality_function="modularity", max_levels=20):
    """Fuehrt den vollstaendigen Leiden-Algorithmus (mit Verfeinerung) auf dem aus den
    Rohdaten gebauten Aehnlichkeitsgraphen aus - Pass fuer Pass protokolliert, damit die
    App den Fortschritt Level fuer Level durchblaettern kann. Braucht anders als
    spectral-demo/k-Means/GMM/... KEINE Ziel-Clusteranzahl irgendwo. `quality_function`
    waehlt zwischen "modularity" (Standard) und "cpm" (Constant Potts Model, siehe
    Modul-Docstring) - dieselbe Maschinerie optimiert in beiden Faellen."""
    data = np.asarray(data, dtype=float)
    base_adjacency = build_similarity_graph(data, n_neighbors)
    passes = _run_passes(
        base_adjacency, resolution, seed, use_refinement=True, quality_function=quality_function, max_levels=max_levels
    )
    return RunResult(passes=passes, resolution=resolution, quality_function=quality_function)


def run_local_moving_only(data, n_neighbors, resolution, seed, quality_function="modularity", max_levels=20):
    """Testeigene Ablation: lokales Verschieben OHNE Verfeinerung (Louvains Verhalten
    nachgebaut) - ausschliesslich als Kontrastfolie in tests/ (zeigt, dass die
    Verfeinerung nicht Teil des noetigen Minimalpfads ist, sondern eine bewusste
    Zusatzgarantie), kein Teil des in der App gezeigten Algorithmus."""
    data = np.asarray(data, dtype=float)
    base_adjacency = build_similarity_graph(data, n_neighbors)
    passes = _run_passes(
        base_adjacency, resolution, seed, use_refinement=False, quality_function=quality_function, max_levels=max_levels
    )
    return RunResult(passes=passes, resolution=resolution, quality_function=quality_function)


def _pairwise_agreement(label_sets):
    """Mittlere paarweise Punktpaar-Uebereinstimmung ueber ALLE Paare von Partitionen in
    `label_sets` (wie ein Rand-Index zwischen je zwei Läufen statt gegen eine
    Ground-Truth) - 1.0 bedeutet, alle Läufe liefern exakt dieselbe Partition (bis auf
    Umbenennung der Labels), niedriger bedeutet echte Seed-Abhängigkeit."""
    label_sets = [np.asarray(l) for l in label_sets]
    n = len(label_sets[0])
    if n < 2 or len(label_sets) < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    same_matrices = [(labels[:, None] == labels[None, :])[iu] for labels in label_sets]
    agreements = []
    for a in range(len(same_matrices)):
        for b in range(a + 1, len(same_matrices)):
            agreements.append(np.mean(same_matrices[a] == same_matrices[b]))
    return float(np.mean(agreements))


def consensus_clustering(
    data, n_neighbors, resolution, seed, quality_function="modularity", n_runs=20, max_rounds=8
):
    """Konsensus-Clustering (Lancichinetti & Fortunato, 2012) - siehe Modul-Docstring für
    die Grundidee. Ablauf:

    1. `n_runs` unabhängige Leiden-Läufe auf dem Basis-Ähnlichkeitsgraphen (oder, ab der
       zweiten Runde, auf der Konsensus-Matrix der vorigen Runde).
    2. Konsensus-Matrix bauen: Eintrag (i,j) = Anteil der Läufe, in denen i und j in
       derselben Community landeten.
    3. Ist die Matrix bereits "kristallklar" (nur 0/1-Einträge) - fertig, alle Läufe
       stimmen exakt überein.
    4. Sonst: die Konsensus-Matrix selbst wird zum neuen gewichteten Graphen für die
       nächste Runde (Schritt 1), bis zu `max_rounds`-mal.

    Braucht `_run_passes` direkt (nicht `run`), da ab Runde 2 die "Kanten" bereits eine
    Konsensus-Matrix sind, kein aus Rohdaten gebauter Ähnlichkeitsgraph mehr."""
    data = np.asarray(data, dtype=float)
    base_adjacency = build_similarity_graph(data, n_neighbors)
    n = len(data)

    seed_rng = np.random.default_rng(seed)
    current_adjacency = base_adjacency
    current_labels = None
    single_run_agreement = None
    converged = False
    n_rounds = 0

    for round_idx in range(max_rounds):
        n_rounds = round_idx + 1
        run_seeds = seed_rng.integers(0, 2**31 - 1, size=n_runs)
        current_labels = []
        for s in run_seeds:
            passes = _run_passes(
                current_adjacency, resolution, int(s), use_refinement=True, quality_function=quality_function
            )
            current_labels.append(np.asarray(passes[-1].partition))

        if round_idx == 0:
            single_run_agreement = _pairwise_agreement(current_labels)

        consensus = np.zeros((n, n))
        for labels in current_labels:
            consensus += (labels[:, None] == labels[None, :]).astype(float)
        consensus /= n_runs
        np.fill_diagonal(consensus, 0.0)

        if np.all((consensus == 0.0) | (consensus == 1.0)):
            converged = True
            break

        current_adjacency = consensus

    final_labels = _relabel(current_labels[0])
    return ConsensusResult(
        final_labels=tuple(int(x) for x in final_labels),
        n_communities=int(final_labels.max()) + 1,
        n_rounds=n_rounds,
        converged=converged,
        single_run_agreement=single_run_agreement,
    )
