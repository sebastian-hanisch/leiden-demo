"""Plotly-Visualisierungen: Ähnlichkeitsgraph (Punkte + Kanten, Stil wie spectral-demo),
Modularitäts-über-Pässe-Kurve (neu für diese Demo), Punktwolke, Kleinmultiples
(verschiedene Auflösungsparameter) und Methoden-Vergleichsdiagramm (Leiden vs. k-Means
bei verschiedenen Ziel-k)."""

import numpy as np

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _cluster_color(index, n_clusters):
    if n_clusters <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
    hue = (index * 360.0 / n_clusters) % 360
    return f"hsl({hue:.1f}, 65%, 50%)"


def _axis_range(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.1 or 1.0
    pady = (ymax - ymin) * 0.1 or 1.0
    return [xmin - padx, xmax + padx], [ymin - pady, ymax + pady]


def _cluster_traces(data, labels, legend):
    import plotly.graph_objects as go

    traces = []
    cluster_ids = sorted(set(labels.tolist()))
    n_clusters = len(cluster_ids)
    show_legend = legend and n_clusters <= 12
    for index, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = _cluster_color(index, n_clusters)
        traces.append(
            go.Scatter(
                x=data[mask, 0], y=data[mask, 1], mode="markers", name=f"Cluster {cid + 1}",
                showlegend=show_legend,
                marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
                hoverinfo="skip",
            )
        )
    return traces


def _build_scatter(data, labels, height, legend):
    import plotly.graph_objects as go

    fig = go.Figure()
    for trace in _cluster_traces(data, np.asarray(labels), legend):
        fig.add_trace(trace)

    xr, yr = _axis_range(data)
    layout_kwargs = dict(
        template="plotly_white", height=height,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        showlegend=legend,
        margin=dict(t=40 if legend else 5, l=10 if legend else 5, r=10 if legend else 5, b=10 if legend else 5),
    )
    if legend:
        layout_kwargs["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(**layout_kwargs)
    return fig


def build_scatter_figure(data, labels):
    return _build_scatter(data, labels, height=460, legend=True)


def build_mini_scatter_figure(data, labels):
    return _build_scatter(data, labels, height=220, legend=False)


def build_graph_figure(data, adjacency, labels):
    """Zeichnet den Ähnlichkeitsgraphen mit Kanten, eingefärbt nach der aktuellen
    Community-Zuordnung - macht sichtbar, welche Kanten innerhalb (kräftig) und
    zwischen (blass) Communities verlaufen."""
    import plotly.graph_objects as go

    n = len(data)
    labels = np.asarray(labels)
    edge_x, edge_y, edge_opacity = [], [], []
    max_weight = adjacency.max() if adjacency.max() > 0 else 1.0
    for i in range(n):
        for j in range(i + 1, n):
            w = adjacency[i, j]
            if w > 0:
                edge_x += [data[i, 0], data[j, 0], None]
                edge_y += [data[i, 1], data[j, 1], None]
                edge_opacity.append(w / max_weight)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines", line=dict(color="#9aa6ba", width=1),
            opacity=0.4, hoverinfo="skip", showlegend=False,
        )
    )
    for trace in _cluster_traces(data, labels, legend=False):
        fig.add_trace(trace)

    xr, yr = _axis_range(data)
    fig.update_layout(
        template="plotly_white", height=420,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        margin=dict(t=10, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_modularity_curve_figure(passes, y_label="Modularität Q"):
    """Qualitaetsfunktion (Modularitaet ODER CPM, je nach `y_label`) gegen Pass-Nummer -
    beweisbar monoton steigend (jede Phase kann Q nur erhoehen oder gleich lassen),
    Analogon zu divisive-demos SSE-Kurve, nur in die andere Richtung."""
    import plotly.graph_objects as go

    levels = [p.level for p in passes]
    values = [p.quality for p in passes]
    fig = go.Figure(
        go.Scatter(x=levels, y=values, mode="lines+markers", line=dict(color="#1f77b4"))
    )
    fig.update_layout(
        template="plotly_white", height=260,
        xaxis=dict(title="Pass (Level)", fixedrange=True, dtick=1),
        yaxis=dict(title=y_label, fixedrange=True),
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_kmeans_across_k_chart(scores, leiden_score, leiden_k):
    """Balkendiagramm: k-Means-Referenz-Rand-Index bei mehreren Ziel-k-Werten (Regler
    muss geraten werden) gegen eine einzelne Referenzlinie fuer Leiden (kein k noetig,
    findet `leiden_k` automatisch)."""
    import plotly.graph_objects as go

    ks = sorted(scores.keys())
    labels = [f"k-Means\n(k={k})" for k in ks]
    values = [scores[k] for k in ks]
    colors = ["#d68a2e"] * len(ks)

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.add_hline(
        y=leiden_score, line=dict(color="#2ca02c", dash="dash"),
        annotation_text=f"Leiden (automatisch k={leiden_k})", annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white", height=300,
        yaxis=dict(title="Rand-Index", range=[0, 1.05], fixedrange=True),
        xaxis=dict(fixedrange=True), margin=dict(t=30, l=10, r=10, b=10), showlegend=False,
    )
    return fig
