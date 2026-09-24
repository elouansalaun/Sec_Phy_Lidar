"""Tests du domaine de l'intrus (src/intrusion.py) sur des grilles synthétiques."""
import numpy as np
import pytest

from src.intrusion import build_edges, gate_width
from src.site import Grid


def grid(h, w):
    return Grid(crs="EPSG:2950", xmin=0.0, ymin=0.0, xmax=float(w), ymax=float(h), res=1.0)


def edges_of(mask, inside=None):
    rows, cols = np.nonzero(mask)
    inside = np.zeros(rows.size, bool) if inside is None else inside[rows, cols]
    u, v, l, fr = build_edges(grid(*mask.shape), rows, cols, inside)
    idx = {(r, c): k for k, (r, c) in enumerate(zip(rows, cols))}
    return idx, {frozenset(e) for e in zip(u.tolist(), v.tolist())}, u, v, l, fr


def test_plein_8_connexite():
    """Grille pleine n×n : 2n(n−1) orthogonales + 2(n−1)² diagonales, chacune une seule fois."""
    n = 5
    _, E, u, v, l, _ = edges_of(np.ones((n, n), bool))
    assert len(E) == u.size == 2 * n * (n - 1) + 2 * (n - 1) ** 2
    assert np.isclose(l[l > 1], np.sqrt(2)).all() and (l > 1).sum() == 2 * (n - 1) ** 2


def test_diagonale_entre_deux_coins_interdite():
    """Deux cellules en diagonale séparées par deux obstacles : pas d'arête."""
    m = np.array([[1, 0],
                  [0, 1]], bool)
    idx, E, *_ = edges_of(m)
    assert frozenset((idx[0, 0], idx[1, 1])) not in E


def test_diagonale_avec_un_voisin_libre_autorisee():
    m = np.array([[1, 1],
                  [0, 1]], bool)
    idx, E, *_ = edges_of(m)
    assert frozenset((idx[0, 0], idx[1, 1])) in E


def test_diagonale_anti_diagonale():
    """Même règle pour l'offset (1, −1)."""
    m = np.array([[0, 1],
                  [1, 0]], bool)
    idx, E, *_ = edges_of(m)
    assert frozenset((idx[0, 1], idx[1, 0])) not in E


def test_franchissement_changement_de_cote():
    """Une arête franchit la clôture si et seulement si ses extrémités sont de côtés opposés."""
    m = np.ones((3, 4), bool)
    inside = np.zeros((3, 4), bool)
    inside[:, 2:] = True
    rows, cols = np.nonzero(m)
    u, v, _, fr = build_edges(grid(3, 4), rows, cols, inside[rows, cols])
    side = inside[rows, cols]
    assert (fr == (side[u] != side[v])).all()
    # colonne 1 -> colonne 2 : 3 orthogonales + 4 diagonales
    assert fr.sum() == 3 + 4


@pytest.mark.parametrize("s, w", [("Portail coulissant 4m", 4.0), ("Portail 1m", 1.0),
                                  ("Portail 2,5 m", 2.5), ("portail 3.5m", 3.5)])
def test_largeur_portail(s, w):
    assert gate_width(s) == w


def test_largeur_portail_absente():
    with pytest.raises(ValueError):
        gate_width("Portail coulissant")
