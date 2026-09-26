"""
app.py — Démo interactive : surveillance optimale d'un site à partir de données LiDAR.

    streamlit run app.py

Site fictif superposé à un secteur urbain de Montréal ; aucune information sur un site réel.
La logique est dans src/demo.py ; cette page ne fait qu'afficher.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.demo import (carte, charger, configurations, decrire, evaluer, meilleure_du_front,
                      solutions_certifiees)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Surveillance LiDAR", page_icon="📡", layout="wide")


@st.cache_resource(show_spinner="Chargement du site, du graphe et de la matrice de détection…")
def donnees():
    return charger(ROOT)


D = donnees()
S = solutions_certifiees(D)

# =============================================================================
# Barre latérale : l'intrus, puis le dispositif
# =============================================================================
sb = st.sidebar
sb.caption("Site **fictif** superposé à un secteur urbain quelconque de Montréal. "
           "Données LiDAR © Gouvernement du Québec (MRNF), CC BY 4.0.")

sb.header("Intrus")
T_R = sb.slider("Temps de réponse T_R (s)", 0, 150, 10, 5,
                help="Une détection ne compte que s'il reste au moins T_R secondes à l'intrus "
                     "pour atteindre Bat.3 (détection opportune, modèle EASI).")
vitesse = sb.select_slider("Vitesse (m/s)", [1.0, 1.5, 3.0], 1.5,
                           format_func=lambda v: {1.0: "1,0 lente", 1.5: "1,5 marche",
                                                  3.0: "3,0 course"}[v])
t_c = sb.slider("Retard : escalade de la clôture (s)", 0, 120, 10, 5)
t_p = sb.slider("Retard : passage d'un portail (s)", 0, 120, 3, 1)
ic = dict(D.cfg["intrus"], vitesse_m_s=float(vitesse), t_cloture_s=float(t_c),
          t_portail_s=float(t_p))

sb.header("Dispositif")
mode = sb.radio("Source", ["Optimum certifié", "Front NSGA-II", "Placement manuel"],
                help="Optimum certifié : boucle adversariale exacte (semaine 7). "
                     "Front NSGA-II : couverture / pire cas (semaine 4). "
                     "Placement manuel : ajoutez ou retirez des capteurs, l'intrus répond.")

if "T" not in st.session_state:
    st.session_state["T"] = S[(S.theta == 0.9) & (S.T_R == 10) & (S.fusion == "max")].iloc[0]["T"]

if mode == "Optimum certifié":
    sm = S[S["fusion"] == "max"]
    th = sb.selectbox("Détection opportune exigée θ", sorted(sm["theta"].unique()), index=1)
    tr = sb.selectbox("Pour un temps de réponse de (s)", sorted(sm[sm.theta == th]["T_R"].unique()),
                      index=1, format_func=lambda v: f"{v:g}")
    ligne = sm[(sm.theta == th) & (sm.T_R == tr)].iloc[0]
    st.session_state["T"] = ligne["T"]
    sb.info(f"Coût minimal **certifié** : {ligne['cout']:.1f} unités.")
elif mode == "Front NSGA-II":
    B = sb.slider("Budget", 2.0, 30.0, 16.0, 0.5)
    crit = sb.radio("Critère", ["F", "p*_F"], horizontal=True,
                    help="F : couverture pondérée. p*_F : détection garantie sur le pire couloir.")
    s = meilleure_du_front(D, B, "F" if crit == "F" else "p")
    if s is None:
        sb.warning("Aucune solution du front sous ce budget.")
        st.session_state["T"] = np.zeros(0, np.int64)
    else:
        st.session_state["T"] = s["T"]
else:
    T = list(map(int, st.session_state["T"]))
    with sb.expander("Ajouter un capteur", expanded=True):
        cands = sorted(D.conf["cand"].unique())
        cand = st.selectbox("Emplacement", cands)
        hs = sorted(D.conf.loc[D.conf["cand"] == cand, "h"].unique())
        h = st.selectbox("Hauteur (m)", hs, format_func=lambda v: f"{v:g}")
        caps = sorted(D.conf.loc[(D.conf["cand"] == cand) & np.isclose(D.conf["h"], h),
                                 "capteur"].unique())
        cap = st.selectbox("Capteur", caps)
        opts = configurations(D, cand, h, cap)
        if cap.startswith("radar"):
            j = st.selectbox("Inclinaison", opts.index,
                             format_func=lambda i: f"{D.conf.at[i, 'inclinaison']:g}°")
        else:
            j = st.selectbox("Azimut", opts.index,
                             format_func=lambda i: f"{D.conf.at[i, 'azimut']:g}°")
        if st.button("Ajouter", use_container_width=True):
            T = sorted(set(T) | {int(j)})
    lab = {int(j): f"{D.conf.at[j, 'cand']} · {D.conf.at[j, 'capteur']} · h {D.conf.at[j, 'h']:g}"
           for j in T}
    retirer = sb.multiselect("Retirer", T, format_func=lambda j: lab[j])
    if sb.button("Retirer la sélection", use_container_width=True) and retirer:
        T = [j for j in T if j not in retirer]
    if sb.button("Tout vider", use_container_width=True):
        T = []
    st.session_state["T"] = np.array(T, np.int64)

# =============================================================================
# Page principale
# =============================================================================
st.title("Surveillance optimale d'un site à partir de données LiDAR")
tab_demo, tab_res, tab_meth = st.tabs(["Démo", "Résultats clés", "Méthode"])

with tab_demo:
    r = evaluer(D, st.session_state["T"], ic, float(T_R))
    k = st.columns(6)
    k[0].metric("Coût", f"{r['cout']:.1f}")
    k[1].metric("Couverture F", f"{100 * r['F']:.1f} %")
    k[2].metric("p* déterminé", f"{r['p_star_F']:.2f}",
                help="Détection garantie sur le couloir le moins surveillé (intrus qui franchit "
                     "la clôture n'importe où).")
    k[3].metric(f"p̃* à T_R = {T_R} s", f"{r['p_tilde_F']:.2f}",
                help="Idem, en ne comptant que les détections assez précoces pour la réponse.")
    k[4].metric("p* opportuniste", f"{r['p_star_G']:.2f}", help="Intrus qui passe par un portail.")
    k[5].metric("T_min", f"{r['T_min_F']:.0f} s",
                help="Temps d'intrusion minimal : au-delà, aucun dispositif ne peut détecter à temps.")

    if T_R >= r["T_min_F"]:
        st.error(f"T_R = {T_R} s dépasse le temps d'intrusion minimal ({r['T_min_F']:.0f} s) : "
                 "**aucun dispositif** ne peut garantir une détection à temps (proposition 2). "
                 "Seul un retard plus grand (clôture **et** portails) peut rendre le cas possible.")
    elif r["p_tilde_F"] < 0.5:
        st.warning("L'intrus déterminé trouve un couloir où il n'est pas détecté à temps "
                   "(tracé en pointillés).")
    for v in r["violations"]:
        st.warning(f"Contrainte violée : {v}")

    zone = st.toggle("Montrer la zone où une détection arrive trop tard", value=True)
    st.plotly_chart(carte(D, r, float(T_R), montrer_zone_morte=zone), use_container_width=True)
    with st.expander(f"Dispositif : {len(r['T'])} capteur(s)"):
        st.dataframe(decrire(D, r["T"]), hide_index=True, use_container_width=True)

with tab_res:
    fig = ROOT / D.cfg["optimisation"]["figures_dir"]
    blocs = [
        ("temps_restant.png", "Détecter n'est pas interrompre",
         "Temps restant jusqu'à la cible. Un coin de Bat.3 est à 1,2 m de la clôture : le temps "
         "d'intrusion minimal est de 30 s. Au-delà de ce temps de réponse, aucun capteur ne protège."),
        ("cout_robuste.png", "Coût minimal certifié",
         "Boucle optimiseur ↔ intrus exacte : chaque point est un optimum prouvé. Demander la "
         "détection opportune coûte 15,4 au lieu de 28,6 pour le meilleur point du front."),
        ("frontiere_retard.png", "Le retard est le vrai levier",
         "Avec clôture et portails à 120 s, une réponse en 2 minutes est couverte pour 15,4. Renforcer "
         "la clôture seule ne sert presque à rien : l'intrus passe par un portail."),
        ("regret.png", "Sensibilité par regret de décision",
         "Le feuillage d'été et un intrus qui court suffisent à casser les décisions nominales ; le "
         "surcoût de réparation reste de 0,4 à 3,1 unités."),
        ("chemins_decouplage.png", "Couvrir n'est pas protéger",
         "À B = 16, l'optimum de couverture gagne 0,6 point de F et perd 67 points de détection garantie."),
    ]
    for f, titre, texte in blocs:
        if (fig / f).exists():
            st.subheader(titre)
            st.caption(texte)
            st.image(str(fig / f), use_container_width=True)

with tab_meth:
    st.markdown("""
**Chaîne complète** (détails dans le README et `docs/formulation.md`) :

1. **LiDAR → rasters** : MNT, bâti, hauteur et fraction de trouée de la végétation (loi de Beer-Lambert).
2. **Visibilité exacte** : parcours de toutes les cellules traversées, bâti opaque, végétation semi-transparente ; validée contre des solutions analytiques et un arbitre exact.
3. **Capteurs** : caméra (densité de pixels, EN 62676-4) et radar (Swerling I), sur 4 151 configurations.
4. **Optimisation de la couverture** : glouton, optimum certifié par coupes sous-modulaires, NSGA-II.
5. **L'intrus** : couloir de moindre détection p\\* (union-find), détection opportune p̃\\* (EASI).
6. **Boucle adversariale exacte** : génération de contraintes, l'intrus sert d'oracle ; optima certifiés.
7. **Sensibilité** par regret de décision, protocole préenregistré.

**Éthique.** Le site, ses zones et ses actifs sont inventés. Publier des cartes d'angles morts d'un site réel sensible serait inacceptable.
""")
