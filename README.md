# Leiden-Algorithmus für automatische Depot-Gruppierung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-leiden-demo.streamlit.app/)**

Zehntes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **Fortsetzung von [spectral-demo](../spectral-demo)**
(wie dpmm-demo gmm-demo fortsetzte, nicht wie divisive-demo ein Kontrast):
spectral-demo benennt im Mathe-Abschnitt offen die eigene verbleibende Schwäche - die
Clusterzahl k muss weiterhin vorab feststehen, genau wie bei k-Means. Diese Demo behebt
**genau das**, mit einem völlig anderen Mechanismus als HDBSCAN (Dichte) oder DPMM
(Bayesianische Nichtparametrik): **Modularitätsoptimierung via des Leiden-Algorithmus**
(Traag, Waltman & van Eck, 2019), dem heutigen De-facto-Standard der
Netzwerk-Community-Detection (Nachfolger von Louvain, 2008). Schließt damit die
"kein k nötig"-Symmetrie über alle drei unabhängigen Äste der Clustering-Linie:

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo   (kein k: über Dichte)
              agglomerative-demo ──────────┘
kmeans-demo → gmm-demo → dpmm-demo             (kein k: über Bayesianische Nichtparametrik)
kmeans-demo → spectral-demo → leiden-demo       (kein k: über Modularitätsoptimierung)
kmeans-demo ─┐
             ├──> divisive-demo
agglomerative-demo ─┘
```

Arbeitet auf demselben Ähnlichkeitsgraphen wie spectral-demo (KNN + Gauß-Kernel-
Gewichte, frisch nachgebaut, kein Cross-Import), aber ohne jede Eigenzerlegung - ein
völlig anderes algorithmisches Paradigma (gieriges lokales Optimieren + hierarchische
Graph-Aggregation statt linearer Algebra) auf identischer Eingabe. **Ehrliche
Kernbotschaft**: Modularitätsoptimierung braucht kein k, hat aber ihre eigene, gut
dokumentierte Schwäche - das **Auflösungslimit** (Fortunato & Barthélemy, 2007):
Cluster, die klein relativ zur Gesamtgraphgröße sind, werden auch bei objektiv klarer
Trennung fälschlich verschmolzen. Kein Fix ohne neuen Kompromiss - konsistent mit jeder
vorherigen Demo dieser Reihe.

**Nachtrag**: die Demo zeigt inzwischen auch die tatsächliche Lösung dieses eigenen
Auflösungslimits - **CPM** (Constant Potts Model, Traag, Van Dooren & Nesterov, 2011,
*"Narrow scope for resolution-limit-free community detection"*, Physical Review E 84,
016114) ersetzt Modularitäts-Nullmodell (graphgrößen-abhängig) durch einen festen
Dichte-Schwellenwert je Knotenpaar - dieselbe Leiden-Maschinerie, nur eine andere
Gewinnformel. Als „Qualitätsfunktion“-Regler in der Seitenleiste umschaltbar; siehe
Preset "Auflösungslimit richtig behoben (CPM)" für den direkten Beweis auf demselben
Szenario, an dem Modularität bei jedem getesteten γ scheitert.

Ebenfalls ergänzt: **Konsensus-Clustering** (Lancichinetti & Fortunato, 2012,
*"Consensus clustering in complex networks"*, Scientific Reports 2, 336) macht die
Seed-Abhängigkeit des lokalen Verschiebens (zufällige Besuchsreihenfolge → verschiedene,
ähnlich gute lokale Optima) sichtbar UND behebt sie: mehrere unabhängige Läufe werden zu
einer Konsensus-Matrix zusammengefasst, die selbst wie ein neuer gewichteter Graph erneut
geclustert wird, bis alle Läufe exakt übereinstimmen. Funktioniert unverändert mit beiden
Qualitätsfunktionen, da nur die zurückgegebenen Partitionen einfließen. Siehe Preset
"Ergebnis hängt vom Zufall ab (Konsensus hilft)".

## Warum diese Demo anders aufgebaut ist

Die Presets legen mehrere Achsen offen:

- **Einfaches Beispiel**: klar getrennte, moderate Gruppenzahl - Leiden findet die
  korrekte Partition UND die korrekte Gruppenzahl vollautomatisch, ganz ohne
  k-Parameter irgendwo in der App.
- **Kein k nötig - viele Gruppen**: deutlich höhere wahre Gruppenzahl, weiterhin klar
  getrennt - zeigt, dass "kein k nötig" robust bleibt, während jede k-Means-artige Demo
  hier einen Ziel-k-Regler bräuchte.
- **Auflösungsgrenze**: viele kleine, eng gepackte Gruppen (20 Gruppen auf nur 60
  Standorten) - Standard-Modularität (γ=1) verschmilzt einige davon trotz klarer
  visueller Trennung - die ehrliche Kernschwäche.
- **Auflösungsparameter als Kompromiss**: dieselbe Szenerie mit deutlich höherem γ -
  hilft (mehr der kleinen Gruppen werden korrekt getrennt), behebt das Limit aber nicht
  vollständig.
- **Kombinierter Härtefall (Dichte + Brücke)**: dieselben Konstruktionsparameter wie
  hdbscan-demo (`density_imbalance`, `bridge_strength`) - Leiden übersteht beide
  Härtefälle deutlich besser als DBSCAN/Single-Linkage (Rand-Index meist >0.95), neigt
  aber zu leichtem Überclustern statt echtem Falsch-Verschmelzen.
- **Auflösungslimit richtig behoben (CPM)**: identisches Szenario wie "Auflösungsgrenze",
  aber mit CPM statt Modularität - findet die wahren 20 Gruppen fast exakt.
- **Ergebnis hängt vom Zufall ab (Konsensus hilft)**: stärker überlappende Gruppen -
  einzelne Läufe stimmen nur zu ~96% überein und finden teils mehr Gruppen als die
  wahren 6; Konsensus-Clustering liefert eine reproduzierbare, meist bessere Antwort.

## Visualisierung

Leiden läuft in diskreten Pässen (lokales Verschieben → Verfeinerung → Aggregation,
wiederholt) - ein Pass-Schrittregler zeigt den Ähnlichkeitsgraphen und die Punktwolke
bei jedem Level, mit einer mitwachsenden Modularitäts-über-Pässe-Kurve (beweisbar
monoton steigend - Analogon zu divisive-demos SSE-Kurve, nur in die andere Richtung).
Kleinmultiples zeigen dieselben Daten unter verschiedenen Auflösungsparametern
nebeneinander.

## Sicherheitsgrenzen

`N_POINTS_HARD_MAX` (300) begrenzt die naive O(n²)-Ähnlichkeitsberechnung. `n_neighbors`
darf bis knapp an `N_POINTS_MAX-1` heranreichen (ein vollständiger Graph) - die
Ähnlichkeitsberechnung ist ohnehin schon O(n²) unabhängig von `n_neighbors`, gemessen
~0.14s bei n_points=300, n_neighbors=299, kein spürbares Performance-Risiko.

## Verifikation

- **Handgerechnetes Beispiel** (für BEIDE Qualitätsfunktionen): zwei Dreiecke, verbunden
  durch eine Brücke - Modularität der korrekten Zwei-Cluster-Partition von Hand
  nachgerechnet (Q = 0.357142857...), ebenso der CPM-Wert (H = 0, gegenüber H = -8 für
  die Ein-Cluster-Partition).
- **Monotonie** (Struktur-Invariante, exakt und algorithmusunabhängig, für BEIDE
  Qualitätsfunktionen einzeln getestet): jede der drei Leiden-Phasen ist so konstruiert,
  dass die Qualität niemals sinkt - über mehrere Pässe hinweg getestet, inklusive eines
  gezielten Mehrebenen-Aggregations-Regressionstests (siehe unten).
- **Zusammenhangsgarantie**: jede von Leiden zurückgegebene Community induziert einen
  zusammenhängenden Teilgraphen - über mehrere Zufallsszenarien direkt bewiesen. Die
  Verfeinerungsphase ist dabei nachweislich kein Blindgang: auf echten Szenarien liefert
  sie eine ANDERE, feinere Partition als lokales Verschieben allein.
- **Kreuzvergleich** gegen `leidenalg`/`python-igraph` (die Referenzimplementierung des
  Algorithmus selbst, toleranzbasiert wie bei hdbscan-demo/spectral-demo, **für BEIDE
  Qualitätsfunktionen** - leidenalg unterstützt CPM nativ über `CPMVertexPartition`) UND
  gegen `networkx.algorithms.community.quality.modularity` (ein exakter
  Zahlen-Kreuzvergleich der eigenen Modularitätsberechnung, deterministisch - networkx
  kennt kein CPM).
- **Ein echter Bug gefunden und mit Regressionstest abgesichert**: die ursprüngliche
  CPM-Gewinnformel im lokalen Verschieben/Verfeinern ließ den Größenfaktor eines
  aggregierten Knotens weg (`resolution * comm_size[c]` statt `resolution * comm_size[c]
  * node_weights[i]`) - auf Szenarien mit mehreren Aggregationsebenen führte das zu
  einer nachweisbar SINKENDEN CPM-Qualität zwischen zwei Pässen (17.4 → 2.45 in der
  manuellen Reproduktion), obwohl das strukturell unmöglich sein sollte. Gefunden durch
  Kreuzvergleich gegen `leidenalg` auf dem Auflösungslimit-Szenario, nicht durch eigene
  Tests allein - ein Beispiel dafür, warum die unabhängige Referenzimplementierung mehr
  ist als Redundanz.
- **Kern-Behauptungen der Demo direkt getestet**: Leiden findet die wahre Gruppenzahl
  ganz ohne k-Parameter bei klar getrennten Gruppen (auch bei hoher Gruppenzahl);
  Standard-Modularität verschmilzt viele kleine, klar getrennte Gruppen (Auflösungslimit);
  ein höherer Auflösungsparameter hilft, aber nicht vollständig; CPM mit passend
  skaliertem γ löst dasselbe Szenario dagegen fast exakt auf.
- **Alle sieben Presets direkt gegen das tatsächliche App-Verhalten getestet** (Szenario-
  Seed = Algorithmus-Seed, wie `app.py` es macht - etabliertes Muster aus
  dpmm-/spectral-/divisive-demo).
- **`DEFAULT_CPM_RESOLUTION` empirisch statt an einem einzigen Szenario kalibriert**:
  über 1500 Szenarien hinweg (n_points/spread/n_neighbors/k/seed variiert) hatte der
  ursprüngliche Standardwert 0.002 eine spürbare Überclustern-Verzerrung
  (mean(found_k − true_k) = +0.16); 0.001 ist über denselben Sweep nahezu unverzerrt
  (+0.03) bei gleichzeitig höherer Exact-Match-Rate - mit Regressionstest abgesichert.
- **Konsensus-Clustering eigens getestet**: Reproduzierbarkeit bei gleichem Seed,
  tatsächliche Konvergenz (scharfe Konsensus-Matrix) innerhalb der Sicherheitsgrenze,
  Funktionieren mit beiden Qualitätsfunktionen, sowie ein Preset-Nachweis, dass das
  gewählte Szenario echte Seed-Abhängigkeit zeigt (sonst wäre nichts zu konsentieren) UND
  Konsensus-Clustering sie zuverlässig auflöst.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Qualitätsfunktions-Umschalter, Pass-Schrittregler, Qualitäts-Kurve, Methoden-Vergleich, Konsensus-Clustering-Sektion, Formulierungs-Expander |
| `ld_constants.py` | Defaults, Regler-Grenzen (inkl. separater CPM-Auflösungsskala), Sicherheitsgrenzen, Konsensus-Clustering-Parameter, `PRESETS` |
| `ld_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `ld_scenario.py` | Blobs (k bis 20, fester statt mit k mitwachsender Ring-Radius - das Vehikel für die Auflösungslimit-Szenarien) und Halbmonde/Bögen, plus `density_imbalance`/`bridge_strength` (hdbscan-demo-Konstruktionsparameter) |
| `ld_algorithm.py` | Ähnlichkeitsgraph-Bau (wie spectral-demo), Modularität UND CPM mit Auflösungsparameter γ, Leiden (lokales Verschieben, Verfeinerung, Aggregation mit Knotengewichts-Fortführung, vollständiges Pass-Protokoll), Konsensus-Clustering, plus eine testeigene Nur-lokales-Verschieben-Ablation |
| `ld_evaluation.py` | Rand-Index (from scratch, schließt Brückenpunkte aus), kleine k-Means-Referenz (mit internen Neustarts) für den "kein k nötig"-Methodenvergleich |
| `ld_visualization.py` | Ähnlichkeitsgraph-Diagramm, Qualitäts-über-Pässe-Kurve (Achsentitel je nach Qualitätsfunktion), Punktwolke, Kleinmultiples, Methoden-Vergleichsdiagramm (Plotly) |
| `tests/` | Handinstanz, Qualitäts-Monotonie (Modularität + CPM + Mehrebenen-Regression), Zusammenhangsgarantie, `leidenalg`-/`networkx`-Kreuzvergleiche, Kern-Nachweise, Konsensus-Clustering-Tests, Preset-gegen-App-Verhalten-Tests, AppTest-Smoke-Test |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
