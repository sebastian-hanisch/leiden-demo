"""Leiden-Algorithmus (Modularitätsoptimierung) für automatische Depot-Gruppierung -
interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der "Konzepte"-Reihe, Fortsetzung von spectral-demo (wie dpmm-demo
gmm-demo fortsetzte): spectral-demo braucht am Ende immer noch k-Means auf der
Spektral-Einbettung, die Zielclusterzahl k muss also weiterhin vorab feststehen. Der
Leiden-Algorithmus (Traag, Waltman & van Eck, 2019) - der heutige De-facto-Standard der
Netzwerk-Community-Detection - behebt genau das ueber Modularitaetsoptimierung, mit
einem voellig anderen Mechanismus als HDBSCAN (Dichte) oder DPMM (Bayesianische
Nichtparametrik): gieriges lokales Verschieben + zusammenhangsgarantierende Verfeinerung
+ hierarchische Graph-Aggregation, ganz ohne Eigenzerlegung.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ld_constants as C
from ld_algorithm import build_similarity_graph, run
from ld_evaluation import kmeans_across_k, rand_index
from ld_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ld_scenario import generate_instance
from ld_visualization import (
    build_graph_figure,
    build_kmeans_across_k_chart,
    build_modularity_curve_figure,
    build_mini_scatter_figure,
    build_scatter_figure,
)

st.set_page_config(page_title="Leiden-Algorithmus – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, spread, density_imbalance, bridge_strength, shape, n_neighbors, resolution, seed):
    instance = generate_instance(
        n_points, k, spread, shape, seed, density_imbalance=density_imbalance, bridge_strength=bridge_strength
    )
    result = run(instance.as_array(), n_neighbors, resolution, seed)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_mini_run(instance, n_neighbors, resolution, seed):
    return run(instance.as_array(), n_neighbors, resolution, seed)


@st.cache_data(show_spinner=False)
def _compute_kmeans_comparison(instance, seed, leiden_k):
    k_values = sorted({max(2, leiden_k - 2), max(2, leiden_k - 1), leiden_k, leiden_k + 1, leiden_k + 2})
    return kmeans_across_k(instance.as_array(), instance.true_labels, k_values, seed)


st.title("🕸️ Leiden-Algorithmus: automatische Depot-Gruppierung ohne Ziel-k")

st.markdown(
    """
Wie viele Verteilzentren sollten aus einem Netz von Standorten entstehen? Jede bisherige
Demo dieser Zeile musste diese Zahl irgendwo vorgeben - **der Leiden-Algorithmus
nicht**. Er optimiert direkt die **Modularität** eines Ähnlichkeitsgraphen (wie viel
dichter sind die Kanten innerhalb einer Gruppe als in einem zufälligen Vergleichsgraphen
mit denselben Gradsummen erwartet?) und findet die Gruppenanzahl dabei automatisch -
ganz ohne k-Parameter irgendwo in dieser App. Der heutige De-facto-Standard der
Netzwerk-Community-Detection, Nachfolger des älteren Louvain-Algorithmus.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere "
    "Verfahren vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - "
    "**ein** Verfahren an einem wachsenden Beispiel: Fortsetzung von spectral-demo, "
    "das dieselbe verbleibende Schwäche offen benennt, die hier behoben wird."
)

with st.expander("So funktioniert der Leiden-Algorithmus", expanded=True):
    st.markdown(
        """
Auf demselben Ähnlichkeitsgraphen wie spectral-demo (k-nächste-Nachbarn, Gauß-Kernel-
Gewichte) wiederholt der Algorithmus einen Pass aus drei Phasen, bis die Modularität
nicht mehr steigt:

- **Lokales Verschieben**: ausgehend von Singleton-Communities verschiebt sich jeder
  Knoten gierig in die Nachbar-Community mit dem größten Modularitätsgewinn.
- **Verfeinerung**: innerhalb jeder so gefundenen Community wird erneut lokal verschoben,
  aber beschränkt auf tatsächlich vorhandene Kanten - das garantiert, dass jede
  zurückgegebene Community **zusammenhängend** bleibt (der ältere Louvain-Algorithmus
  kann ohne diesen Schritt nachweislich nicht-zusammenhängende Communities liefern).
- **Aggregation**: die gefundenen Communities werden zu einem neuen, kleineren Graphen
  zusammengefasst, und der ganze Pass beginnt von vorn.

Ein **Auflösungsparameter** γ steuert dabei, wie leicht eine Trennung "lohnt" - der eine
method-eigene Regler dieser Demo.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel": "Klar getrennte, moderate Gruppenzahl - Leiden findet die korrekte Partition UND die korrekte Gruppenzahl vollautomatisch.",
    "Kein k nötig - viele Gruppen": "Deutlich höhere wahre Gruppenzahl, weiterhin klar getrennt - 'kein k nötig' bleibt robust.",
    "Auflösungsgrenze": "Viele kleine, eng gepackte Gruppen - Standard-Modularität (γ=1) verschmilzt einige davon trotz klarer Trennung.",
    "Auflösungsparameter als Kompromiss": "Dieselbe Szenerie mit höherem γ - hilft, behebt das Auflösungslimit aber nicht vollständig.",
    "Kombinierter Härtefall (Dichte + Brücke)": "Dieselben zwei Härtefälle, an denen DBSCAN bzw. Single-Linkage-Chaining scheitern - Leiden übersteht beide deutlich besser (Rand-Index meist >0.95), ist aber nicht perfekt immun: die Brückenpunkte selbst bilden gelegentlich eine eigene kleine Community, statt zwei echte Gruppen fälschlich zu verschmelzen.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach "
    "kopieren, um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Standorte", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider("Anzahl wahrer Gruppen", *bounds("k_slider"), key="k_slider")
    spread = st.slider(
        "Streuung", *bounds("spread_slider"), key="spread_slider", step=0.01,
        help="Klein = Gruppen klar getrennt. Groß = Gruppen überlappen sich spürbar.",
    )
    density_imbalance = st.slider(
        "Dichte-Ungleichgewicht", *bounds("density_imbalance_slider"), key="density_imbalance_slider",
        step=0.05,
        help="0 = alle Gruppen gleich dicht. 1 = eine Gruppe wird deutlich lockerer/diffuser "
        "als die übrigen, bei gleicher Punktzahl - derselbe Härtefall wie in dbscan-demo/"
        "hdbscan-demo.",
    )
    bridge_strength = st.slider(
        "Brücken-Stärke (zwischen den ersten beiden Gruppen)", *bounds("bridge_strength_slider"),
        key="bridge_strength_slider", step=0.02,
        help="0 = keine Brücke. Höher = mehr verbindende Punkte zwischen Gruppe 1 und 2 - "
        "derselbe Härtefall wie in agglomerative-demo/hdbscan-demo (dort Single-Linkage-"
        "Chaining).",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Punktwolken-Form**")
    shape = st.radio(
        "Form", options=C.SHAPES, key="shape_radio", format_func=lambda s: C.SHAPE_LABELS[s],
        help="„Gruppen“: runde, konvexe Cluster. „Halbmonde“: nicht-konvexe Bögen - Dichte-"
        "Ungleichgewicht und Brücken-Stärke wirken auf beide Formen.",
    )
    n_neighbors = st.slider(
        "Nachbarn je Punkt (Graph-Konstruktion)", *bounds("n_neighbors_slider"), key="n_neighbors_slider",
    )

    st.markdown("**Leiden-Parameter**")
    resolution = st.slider(
        "Auflösungsparameter γ", *bounds("resolution_slider"), key="resolution_slider", step=0.1,
        help="1.0 = klassische Modularität. Höher = mehr, kleinere Communities werden "
        "bevorzugt - kann das Auflösungslimit lindern, aber um den Preis, andernorts zu "
        "übersplitten.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Standorte.",
    )

sync_query_params(n_points, k, spread, density_imbalance, bridge_strength, seed, shape, n_neighbors, resolution)

with st.spinner("Führe den Leiden-Algorithmus aus..."):
    instance, result = _compute_run(
        int(n_points), int(k), spread, density_imbalance, bridge_strength, shape,
        int(n_neighbors), resolution, int(seed)
    )

max_step = len(result.passes) - 1
run_key = (n_points, k, spread, density_imbalance, bridge_strength, shape, n_neighbors, resolution, seed)
if "ld_step" not in st.session_state or st.session_state.get("ld_step_owner") != run_key:
    st.session_state["ld_step"] = max_step
    st.session_state["ld_step_owner"] = run_key

st.markdown("## 🎯 Leiden-Algorithmus in Aktion")

if max_step == 0:
    step = 0
    st.caption("Bereits nach dem ersten Pass konvergiert - kein Regler nötig.")
else:
    step = st.slider(
        "Pass (Level)", 0, max_step, key="ld_step",
        help="Ein Pass = lokales Verschieben + Verfeinerung + Aggregation. Reglerposition "
        "steht standardmäßig auf dem letzten (konvergierten) Pass - frei verschiebbar, um "
        "den Fortschritt Level für Level zu erkunden.",
    )
current_pass = result.passes[step]
base_adjacency = build_similarity_graph(instance.as_array(), int(n_neighbors))

graph_col, scatter_col = st.columns(2)
graph_col.plotly_chart(
    build_graph_figure(instance.as_array(), base_adjacency, current_pass.partition),
    width="stretch", key=f"graph_{step}",
)
scatter_col.plotly_chart(
    build_scatter_figure(instance.as_array(), current_pass.partition),
    width="stretch", key=f"scatter_{step}",
)

lm1, lm2, lm3 = st.columns(3)
lm1.metric("Pässe bisher", f"{step + 1}/{max_step + 1}")
lm2.metric("Gefundene Gruppenanzahl", current_pass.n_communities)
lm3.metric("Modularität Q", f"{current_pass.modularity:.4f}")

st.plotly_chart(build_modularity_curve_figure(result.passes[: step + 1]), width="stretch", key="q_curve")

st.markdown("**Und mit anderem Auflösungsparameter?**")
st.caption("Gleiche Standorte wie oben - nur γ unterscheidet sich.")
example_resolutions = [0.5, 1.0, 2.0, 3.5]
example_cols = st.columns(len(example_resolutions))
for col, example_res in zip(example_cols, example_resolutions):
    with col:
        example_result = _compute_mini_run(instance, int(n_neighbors), example_res, int(seed))
        st.plotly_chart(
            build_mini_scatter_figure(instance.as_array(), example_result.final_labels),
            width="stretch", key=f"mini_{example_res}",
        )
        st.caption(f"γ={example_res} ({example_result.n_communities} Gruppen)")

st.markdown("---")

st.subheader("📐 Kein k nötig - aber wo ist der Haken?")
st.markdown(
    """
Live für Ihr aktuelles Szenario berechnet: der **Rand-Index** (Anteil der Punktpaare,
bei denen die berechnete Aufteilung mit der tatsächlichen Gruppenzugehörigkeit
übereinstimmt) einer k-Means-Referenz bei mehreren möglichen Ziel-k-Werten, gegen eine
einzelne Referenzlinie für Leiden - das ganz ohne jeden k-Parameter auskommt:
"""
)

leiden_score = rand_index(instance.true_labels, result.final_labels)
kmeans_scores = _compute_kmeans_comparison(instance, C.COMPARISON_SEED, result.n_communities)
st.plotly_chart(
    build_kmeans_across_k_chart(kmeans_scores, leiden_score, result.n_communities),
    width="stretch", key="kmeans_across_k",
)

true_k = instance.k
gap_from_true = result.n_communities - true_k
if gap_from_true == 0:
    st.success(
        f"✅ Leiden findet die wahre Gruppenzahl ({true_k}) UND einen Rand-Index von "
        f"{leiden_score:.2f} - ganz ohne dass irgendwo ein k eingestellt wurde. Jede "
        f"k-Means-artige Methode bräuchte genau diese Zahl vorab."
    )
elif gap_from_true < 0:
    st.warning(
        f"⚠️ Das Auflösungslimit der Modularität in Aktion: Leiden findet hier "
        f"{result.n_communities} Gruppen statt der wahren {true_k} - viele kleine, klar "
        f"getrennte Gruppen werden trotz objektiv klarer Trennung verschmolzen, weil sie "
        f"klein relativ zur Gesamtgraphgröße sind. 'Kein k nötig' ist kein Free Lunch: "
        f"probieren Sie einen höheren Auflösungsparameter γ in der Seitenleiste - er "
        f"hilft, behebt das Limit aber nicht vollständig."
    )
else:
    st.warning(
        f"⚠️ Leiden findet hier {result.n_communities} Gruppen statt der wahren {true_k} "
        f"- mehr, nicht weniger: bei Dichte-Ungleichgewicht oder einer Punktbrücke "
        f"(Seitenleiste) kann ein dünn verbundener Bereich zu einer eigenen kleinen "
        f"Community werden, statt zwei echte Gruppen fälschlich zu verschmelzen (der "
        f"Rand-Index bleibt trotzdem meist hoch, siehe oben - anders als bei DBSCAN oder "
        f"Single-Linkage-Chaining). Ein niedrigerer Auflösungsparameter γ fasst solche "
        f"kleinen Community-Fragmente eher wieder zusammen."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modularität mit Auflösungsparameter** $\gamma$ (Reichardt-Bornholdt-Generalisierung,
$\gamma=1$ ist die klassische Modularität):

$$
Q_\gamma = \frac{1}{2m}\sum_{ij}\left[A_{ij} - \gamma\frac{k_i k_j}{2m}\right]\delta(c_i,c_j)
$$

wobei $A$ die gewichtete Adjazenzmatrix, $k_i$ der Grad von Knoten $i$, $m$ die
Gesamtkantengewichtsumme und $\delta(c_i,c_j)$ 1 ist, wenn $i,j$ derselben Community
angehören.

**Leiden-Algorithmus** (Traag, Waltman & van Eck, 2019), wiederholt bis Konvergenz:

1. **Lokales Verschieben**: jeder Knoten wandert gierig in die Nachbar-Community mit
   größtem Gewinn $k_{i,\text{in}}(C) - \gamma \cdot \Sigma_{\text{tot}}(C) \cdot k_i / 2m$.
2. **Verfeinerung**: innerhalb jeder gefundenen Community erneut lokal verschieben,
   beschränkt auf tatsächlich vorhandene Kanten - garantiert Zusammenhang. Der ältere
   Louvain-Algorithmus (2008) überspringt diesen Schritt und kann deshalb nach
   Aggregation nicht-zusammenhängende Communities liefern.
3. **Aggregation**: Communities werden zu Knoten eines neuen, kleineren Graphen.

**Auflösungslimit** (Fortunato & Barthélemy, 2007): Modularitätsoptimierung kann
Communities, die kleiner als $\mathcal{O}(\sqrt{2m})$ relativ zum Gesamtgraphen sind,
prinzipiell nicht zuverlässig auflösen - selbst bei objektiv perfekter Trennung kann das
Verschmelzen zweier solcher Gruppen die Modularität erhöhen. Ein höheres $\gamma$
verschiebt diese Grenze, hebt sie aber nicht auf.

**Ehrlicher Kompromiss**: kein Ziel-k mehr nötig (anders als bei jeder k-Means-artigen
Demo dieser Reihe), aber der Auflösungsparameter γ ist ein neuer freier Parameter mit
eigener, gut dokumentierter Schwäche.

Implementiert in `ld_algorithm.py` (Leiden, Modularität) und `ld_evaluation.py`
(Rand-Index, k-Means-Referenz).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
