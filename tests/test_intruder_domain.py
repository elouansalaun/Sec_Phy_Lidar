"""Tests of the intruder domain (src/intruder_domain.py) on synthetic grids."""
import numpy as np
import pytest

from src.intruder_domain import build_edges, gate_width
from src.site import Grid


def grid(h, w):
    return Grid(crs="EPSG:2950", xmin=0.0, ymin=0.0, xmax=float(w), ymax=float(h), res=1.0)


def edges_of(mask, inside=None):
    rows, cols = np.nonzero(mask)
    inside = np.zeros(rows.size, bool) if inside is None else inside[rows, cols]
    u, v, l, fr = build_edges(grid(*mask.shape), rows, cols, inside)
    idx = {(r, c): k for k, (r, c) in enumerate(zip(rows, cols))}
    return idx, {frozenset(e) for e in zip(u.tolist(), v.tolist())}, u, v, l, fr


def test_full_8_connectivity():
    """Full n×n grid: 2n(n−1) orthogonal + 2(n−1)² diagonal edges, each exactly once."""
    n = 5
    _, E, u, v, l, _ = edges_of(np.ones((n, n), bool))
    assert len(E) == u.size == 2 * n * (n - 1) + 2 * (n - 1) ** 2
    assert np.isclose(l[l > 1], np.sqrt(2)).all() and (l > 1).sum() == 2 * (n - 1) ** 2


def test_diagonal_between_two_corners_forbidden():
    """Two diagonal cells separated by two obstacles: no edge."""
    m = np.array([[1, 0],
                  [0, 1]], bool)
    idx, E, *_ = edges_of(m)
    assert frozenset((idx[0, 0], idx[1, 1])) not in E


def test_diagonal_with_a_free_neighbour_allowed():
    m = np.array([[1, 1],
                  [0, 1]], bool)
    idx, E, *_ = edges_of(m)
    assert frozenset((idx[0, 0], idx[1, 1])) in E


def test_anti_diagonal():
    """Same rule for the offset (1, −1)."""
    m = np.array([[0, 1],
                  [1, 0]], bool)
    idx, E, *_ = edges_of(m)
    assert frozenset((idx[0, 1], idx[1, 0])) not in E


def test_crossing_is_a_change_of_side():
    """An edge crosses the fence if and only if its ends are on opposite sides."""
    m = np.ones((3, 4), bool)
    inside = np.zeros((3, 4), bool)
    inside[:, 2:] = True
    rows, cols = np.nonzero(m)
    u, v, _, fr = build_edges(grid(3, 4), rows, cols, inside[rows, cols])
    side = inside[rows, cols]
    assert (fr == (side[u] != side[v])).all()
    # column 1 -> column 2: 3 orthogonal + 4 diagonal
    assert fr.sum() == 3 + 4


@pytest.mark.parametrize("s, w", [("Sliding gate 4m", 4.0), ("Gate 1m", 1.0),
                                  ("Gate 2,5 m", 2.5), ("gate 3.5m", 3.5),
                                  ("Portail coulissant 4m", 4.0)])
def test_gate_width(s, w):
    assert gate_width(s) == w


def test_gate_width_missing():
    with pytest.raises(ValueError):
        gate_width("Sliding gate")
