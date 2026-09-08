"""End-to-end Smoke-Test via Streamlits offizielles AppTest-Framework: laedt app.py mit
den Standardeinstellungen sowie allen Presets und prueft, dass kein Python-Fehler
auftritt - ein Fehler wie `StreamlitDuplicateElementId` (zwei st.plotly_chart-Aufrufe
ohne eindeutiges key=) liegt in app.py's Widget-Verdrahtung selbst und kann nur durch
einen echten End-to-End-Lauf gefunden werden, nicht durch Unit-Tests der
Algorithmus-/Visualisierungs-Module (siehe hdbscan-demo, wo genau dieser Fehlertyp bei
einem echten Nutzer auftrat, 2026-09-05)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import ld_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("preset_name", list(C.PRESETS.keys()))
def test_all_presets_load_without_exception(preset_name):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    buttons = {b.label: b for b in at.button}
    buttons[preset_name].click().run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


def test_cpm_resolution_survives_toggling_quality_function_away_and_back():
    """Regressionstest fuer einen echten Bug: der logarithmische CPM-Auflösungsregler
    verwaltet seine Regler-Position in einem eigenen Session-State-Schluessel
    (`cpm_resolution_log_slider`), der verschwindet, sobald der CPM-Zweig auf einem
    Rerun uebersprungen wird (Qualitätsfunktion kurzzeitig auf Modularität umgestellt).
    Ohne explizite Pruefung auf FEHLEN dieses Schluessels (nicht nur auf geaenderten
    linearen Wert) fiel der Regler beim Zurueckschalten stillschweigend auf
    CPM_RESOLUTION_SLIDER_FLOOR zurueck statt den zuletzt gesetzten Wert (hier 0.5) zu
    behalten - gefunden durch echtes Hin- und Herschalten im Browser, nicht durch
    Unit-Tests der Algorithmus-Module allein."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["quality_function_radio"] = "cpm"
    at.session_state["cpm_resolution_slider"] = 0.5
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["cpm_resolution_slider"] == pytest.approx(0.5)

    at.radio(key="quality_function_radio").set_value("modularity").run(timeout=120)
    at.radio(key="quality_function_radio").set_value("cpm").run(timeout=120)

    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["cpm_resolution_slider"] == pytest.approx(0.5), (
        "CPM resolution reset instead of being preserved across a quality-function toggle"
    )
