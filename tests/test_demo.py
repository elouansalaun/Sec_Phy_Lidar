"""Tests de la démo : logique (src/demo.py) et page Streamlit complète (app.py, AppTest)."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/boucle/solutions.csv").exists(),
                                reason="lancer src.boucle d'abord")


@pytest.fixture(scope="module")
def D():
    from src.demo import charger
    return charger(ROOT)


def test_evaluer_redonne_la_boucle(D):
    """L'optimum certifié θ = 0,9, T_R = 10 s tient bien sa promesse dans la démo."""
    from src.demo import evaluer, solutions_certifiees
    S = solutions_certifiees(D)
    for T_R in (0.0, 10.0):
        s = S[(S.theta == 0.9) & (S.T_R == T_R) & (S.fusion == "max")].iloc[0]
        r = evaluer(D, s["T"], D.cfg["intrus"], T_R)
        assert r["cout"] == pytest.approx(s["cout"])
        assert r["p_tilde_F"] >= 0.9 - 1e-9 and not r["violations"]


def test_dispositif_vide(D):
    from src.demo import evaluer
    r = evaluer(D, [], D.cfg["intrus"], 0.0)
    assert r["cout"] == 0 and r["F"] == 0 and r["p_star_F"] == 0


def test_page_complete():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert [m.label for m in at.metric][:3] == ["Coût", "Couverture F", "p* déterminé"]
    at.sidebar.radio[0].set_value("Placement manuel").run()
    assert not at.exception
    at.sidebar.slider[0].set_value(60).run()                 # T_R > T_min : impossibilité
    assert not at.exception and any("aucun dispositif" in e.value for e in at.error)
