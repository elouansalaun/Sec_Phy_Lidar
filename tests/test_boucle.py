"""Tests de la boucle adversariale (src/boucle.py, formulation §9.2) : force brute sur instances jouets."""
import itertools

import numpy as np
import pytest
from scipy import sparse

from src.boucle import boucle, coefficients, niveau
from src.instance import Instance, cost
from src.intrus import detection, seuil_couloir
from src.opportune import temps_restant
from test_intrus import cloture, cols_mask, domaine

IC = dict(vitesse_m_s=1.5, t_cloture_s=10.0, t_portail_s=3.0, tau_obs_s=1.0)


def jouet(seed, nJ=7):
    rng = np.random.default_rng(seed)
    m, ins = cloture(H=7, W=8)
    ins[:, :] = False; ins[:, 4:] = True
    dom, *_ = domaine(m, cols_mask(7, 8, 0), cols_mask(7, 8, 7), inside=ins, portail_rows=[3])
    D = rng.random((nJ, dom.n)) * (rng.random((nJ, dom.n)) < 0.45)
    D[D > 0.85] = 1.0
    A = sparse.csr_matrix(D)
    n_int = int(dom.interieur.sum())
    order = np.r_[np.flatnonzero(dom.interieur), np.flatnonzero(~dom.interieur)]
    assert (order[:n_int] == np.flatnonzero(dom.interieur)).all()
    sup = rng.integers(0, 4, nJ)                             # 4 supports, 2 hauteurs par site
    inst = Instance(A=sparse.csr_matrix(D[:, dom.interieur]), w=np.ones(n_int),
                    support=sup, kind=rng.integers(0, 2, nJ),
                    cost_sensor=np.round(rng.uniform(0.5, 2, nJ), 2),   # au centime, comme le site
                    site=np.array([0, 0, 1, 1]), cost_support=np.round(rng.uniform(0.2, 1, 4), 2),
                    capacity=np.array([2, 1]))
    return dom, A, inst


def force_brute(dom, A, inst, d, theta, T_R, fusion):
    from src.instance import violations
    best = np.inf
    for k in range(inst.n_configs + 1):
        for T in itertools.combinations(range(inst.n_configs), k):
            T = np.array(T, np.int64)
            if violations(inst, T, 1e6):
                continue
            P = (detection(A, T)[0] if fusion == "produit"
                 else (A[T].toarray().max(axis=0) if T.size else np.zeros(dom.n)))
            p = seuil_couloir(dom, np.where(d >= T_R, P, 0.0), "F")[0]
            if p >= theta - 1e-9:
                best = min(best, cost(inst, T))
    return best


@pytest.mark.parametrize("fusion", ["max", "produit"])
def test_equivalence_niveau(fusion):
    rng = np.random.default_rng(3)
    D = rng.random((6, 40)) * (rng.random((6, 40)) < 0.6)
    D[0, :5] = 1.0
    A = sparse.csr_matrix(D)
    for theta in (0.3, 0.6, 0.9):
        C = coefficients(A, theta, fusion)
        for _ in range(20):
            T = np.flatnonzero(rng.random(6) < 0.5)
            P = detection(A, T)[0] if fusion == "produit" else (D[T].max(axis=0) if T.size else 0 * D[0])
            assert ((niveau(C, T) >= 1 - 1e-9) == (P >= theta - 1e-9)).all()


@pytest.mark.parametrize("fusion", ["max", "produit"])
def test_egal_force_brute(fusion):
    n_ok = 0
    for seed in range(12):
        dom, A, inst = jouet(seed)
        d = temps_restant(dom, IC, "F")
        for theta, T_R in ((0.6, 0.0), (0.8, 0.0), (0.6, 6.0)):
            ref = force_brute(dom, A, inst, d, theta, T_R, fusion)
            r = boucle(inst, dom, A, d, theta, T_R, "F", "cout", n_chemins=3, fusion=fusion,
                       temps_max_s=60)
            if np.isinf(ref):
                assert r.statut == "irrealisable"
            else:
                assert r.statut == "optimal" and r.valeur == pytest.approx(ref)
                n_ok += 1
    assert n_ok >= 10                                      # les cas réalisables sont couverts


def test_pas_cout():
    from src.boucle import pas_cout
    _, _, inst = jouet(0)
    assert pas_cout(inst) == 0.01


def test_reservoir_reutilisable():
    """Un réservoir rempli par un cas ne change pas l'optimum d'un autre cas."""
    dom, A, inst = jouet(5)
    d = temps_restant(dom, IC, "F")
    res = []
    boucle(inst, dom, A, d, 0.6, 0.0, "F", "cout", fusion="max", reservoir=res)
    assert res
    for theta, T_R in ((0.8, 0.0), (0.6, 6.0)):
        a = boucle(inst, dom, A, d, theta, T_R, "F", "cout", fusion="max")
        b = boucle(inst, dom, A, d, theta, T_R, "F", "cout", fusion="max", reservoir=list(res))
        assert a.statut == b.statut
        if a.statut == "optimal":
            assert a.valeur == pytest.approx(b.valeur)
