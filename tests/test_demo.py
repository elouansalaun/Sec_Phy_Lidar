"""Tests of the demo: logic (src/demo.py) and full Streamlit page (app.py, AppTest)."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/loop/solutions.csv").exists(),
                                reason="run src.adversarial_loop first")


@pytest.fixture(scope="module")
def D():
    from src.demo import load_demo
    return load_demo(ROOT)


def test_evaluate_gives_back_the_loop(D):
    """Every certified optimum of the loop keeps its promise in the demo."""
    from src.demo import certified_solutions, evaluate_layout
    S = certified_solutions(D)
    S = S[S.fusion == "max"]
    assert len(S), "no certified optimum"
    for _, s in S.iterrows():
        r = evaluate_layout(D, s["T"], D.cfg["intruder"], float(s["T_R"]))
        assert r["cost"] == pytest.approx(s["cost"])
        assert r["p_tilde_F"] >= s["theta"] - 1e-9 and not r["violations"]


def test_empty_layout(D):
    from src.demo import evaluate_layout
    r = evaluate_layout(D, [], D.cfg["intruder"], 0.0)
    assert r["cost"] == 0 and r["F"] == 0 and r["p_star_F"] == 0


def test_full_page():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert [m.label for m in at.metric][:3] == ["Cost", "Coverage F", "p* determined"]
    at.sidebar.radio[0].set_value("Manual placement").run()
    assert not at.exception
    at.sidebar.slider[0].set_value(60).run()                 # T_R > T_min: impossibility
    assert not at.exception and any("no layout" in e.value for e in at.error)
