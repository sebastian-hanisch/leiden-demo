"""Rand-Index (from scratch) sowie eine kleine, frische k-Means-Referenzimplementierung
(bewusst NICHT aus einer anderen Demo importiert) - das Vehikel fuer die
"kein k nötig"-Kernaussage: k-Means braucht ein Ziel-k und scheitert bei falscher Wahl,
Leiden nie."""

import numpy as np


def rand_index(true_labels, pred_labels):
    """Anteil der Punktpaare, bei denen beide Partitionen uebereinstimmen (entweder
    beide im selben Cluster oder beide in unterschiedlichen). Punkte mit true_label -1
    (Bruecken-/Ausreisserpunkte ohne echte Gruppenzugehoerigkeit, siehe
    ld_scenario.py::_add_bridge) werden ausgeschlossen, wie in hdbscan-demo."""
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    mask = true_arr != -1
    true_arr, pred_arr = true_arr[mask], pred_arr[mask]
    n = len(true_arr)
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    same_true = (true_arr[:, None] == true_arr[None, :])[iu]
    same_pred = (pred_arr[:, None] == pred_arr[None, :])[iu]
    return float(np.mean(same_true == same_pred))


def _kmeans_once(data, k, rng, max_iter=100):
    n = len(data)
    idx = rng.choice(n, size=k, replace=False)
    centers = data[idx].copy()

    labels = None
    for _iteration in range(max_iter):
        dists = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = data[mask].mean(axis=0)
    inertia = float(((data - centers[labels]) ** 2).sum())
    return labels, inertia


def kmeans_baseline_labels(data, k, seed, n_restarts=5, max_iter=100):
    """Kleines, frisches Lloyd's (kein Cross-Repo-Import) auf den rohen 2D-Koordinaten,
    mit ein paar internen Neustarts (behaelt den Lauf mit niedrigster Inertia) - die
    einfachste denkbare "man muss k vorher kennen"-Referenz, im Gegensatz zu Leiden, das
    ganz ohne Ziel-k auskommt."""
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(data)
    k = min(k, n)

    best_labels, best_inertia = None, np.inf
    for _ in range(n_restarts):
        labels, inertia = _kmeans_once(data, k, rng, max_iter)
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def kmeans_across_k(data, true_labels, k_values, seed):
    """Rand-Index der k-Means-Referenz fuer mehrere Ziel-k-Werte - zeigt live, wie stark
    das Ergebnis von der (bei Leiden gar nicht noetigen) Wahl von k abhaengt."""
    scores = {}
    for k in k_values:
        labels = kmeans_baseline_labels(data, k, seed)
        scores[k] = rand_index(true_labels, labels)
    return scores
