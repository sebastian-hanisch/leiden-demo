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
