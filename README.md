# Leiden-Algorithmus für automatische Depot-Gruppierung – Streamlit-Demo

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

## Warum diese Demo anders aufgebaut ist

Die Presets legen zwei getrennte Achsen offen:

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

## Visualisierung

Leiden läuft in diskreten Pässen (lokales Verschieben → Verfeinerung → Aggregation,
wiederholt) - ein Pass-Schrittregler zeigt den Ähnlichkeitsgraphen und die Punktwolke
bei jedem Level, mit einer mitwachsenden Modularitäts-über-Pässe-Kurve (beweisbar
monoton steigend - Analogon zu divisive-demos SSE-Kurve, nur in die andere Richtung).
Kleinmultiples zeigen dieselben Daten unter verschiedenen Auflösungsparametern
nebeneinander.

## Sicherheitsgrenzen

`N_POINTS_HARD_MAX` (300) begrenzt die naive O(n²)-Ähnlichkeitsberechnung.

## Verifikation

- **Handgerechnetes Beispiel**: zwei Dreiecke, verbunden durch eine Brücke - Modularität
  der korrekten Zwei-Cluster-Partition von Hand nachgerechnet (Q = 0.357142857...).
- **Modularitäts-Monotonie** (Struktur-Invariante, exakt und algorithmusunabhängig):
  jede der drei Leiden-Phasen ist so konstruiert, dass die Modularität niemals sinkt -
  über mehrere Pässe hinweg getestet.
- **Zusammenhangsgarantie**: jede von Leiden zurückgegebene Community induziert einen
  zusammenhängenden Teilgraphen - über mehrere Zufallsszenarien direkt bewiesen. Die
  Verfeinerungsphase ist dabei nachweislich kein Blindgang: auf echten Szenarien liefert
  sie eine ANDERE, feinere Partition als lokales Verschieben allein.
- **Kreuzvergleich** gegen `leidenalg`/`python-igraph` (die Referenzimplementierung des
  Algorithmus selbst, toleranzbasiert wie bei hdbscan-demo/spectral-demo) UND gegen
  `networkx.algorithms.community.quality.modularity` (ein exakter Zahlen-Kreuzvergleich
  der eigenen Modularitätsberechnung, deterministisch).
- **Kern-Behauptungen der Demo direkt getestet**: Leiden findet die wahre Gruppenzahl
  ganz ohne k-Parameter bei klar getrennten Gruppen (auch bei hoher Gruppenzahl);
  Standard-Modularität verschmilzt viele kleine, klar getrennte Gruppen (Auflösungslimit);
  ein höherer Auflösungsparameter hilft, aber nicht vollständig.
- **Alle vier Presets direkt gegen das tatsächliche App-Verhalten getestet** (Szenario-
  Seed = Algorithmus-Seed, wie `app.py` es macht - etabliertes Muster aus
  dpmm-/spectral-/divisive-demo).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Pass-Schrittregler, Modularitäts-Kurve, Methoden-Vergleich, Formulierungs-Expander |
| `ld_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `ld_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `ld_scenario.py` | Blobs (k bis 20, fester statt mit k mitwachsender Ring-Radius - das Vehikel für die Auflösungslimit-Szenarien) und Halbmonde/Bögen |
| `ld_algorithm.py` | Ähnlichkeitsgraph-Bau (wie spectral-demo), Modularität mit Auflösungsparameter γ, Leiden (lokales Verschieben, Verfeinerung, Aggregation, vollständiges Pass-Protokoll), plus eine testeigene Nur-lokales-Verschieben-Ablation |
| `ld_evaluation.py` | Rand-Index (from scratch), kleine k-Means-Referenz (mit internen Neustarts) für den "kein k nötig"-Methodenvergleich |
| `ld_visualization.py` | Ähnlichkeitsgraph-Diagramm, Modularitäts-über-Pässe-Kurve, Punktwolke, Kleinmultiples, Methoden-Vergleichsdiagramm (Plotly) |
| `tests/` | Handinstanz, Modularitäts-Monotonie, Zusammenhangsgarantie, `leidenalg`-/`networkx`-Kreuzvergleiche, Kern-Nachweise, Preset-gegen-App-Verhalten-Tests, AppTest-Smoke-Test |

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
