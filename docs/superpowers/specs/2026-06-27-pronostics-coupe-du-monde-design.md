# Pronostics Coupe du Monde 2026 — Spécification de conception

**Date :** 2026-06-27
**Statut :** Approuvé (design) — en attente de plan d'implémentation

## 1. Objectif

Application web qui affiche des **pronostics de score** pour les matchs **à venir**
de la Coupe du Monde 2026, fondés sur une **analyse statistique** issue de données
réelles et fiables (API football-data.org).

L'application possède une **mémoire** : elle stocke chaque pronostic émis, le
compare au **score réel** une fois le match joué, et **s'auto-améliore** en
ré-ajustant les paramètres de son modèle à partir de cet historique.

Pour chaque match à venir, l'application produit :

- **Score exact** le plus probable (ex : 2-1)
- **Probabilités 1N2** (victoire domicile / nul / victoire extérieur)
- **Over/Under 2.5 buts** et **BTTS** (les deux équipes marquent)
- **Stats d'appui** : forme récente, confrontations directes, moyennes de buts

Un **disclaimer** clair précise que ce sont des estimations probabilistes, pas des
garanties.

## 2. Source de données

- **football-data.org** (plan gratuit).
- Compétition : **Coupe du Monde**, code `WC`.
- Limite : ~10 requêtes/minute → impose un **cache**.
- Clé API stockée côté serveur dans `.env` (jamais exposée au navigateur).
- À vérifier en début d'implémentation : que la compétition `WC` est bien
  accessible avec le plan gratuit utilisé. Sinon, replier sur une compétition de
  test disponible et le documenter.

## 3. Méthode statistique — Modèle de Poisson (Dixon-Coles simplifié)

### 3.1 Forces des équipes
Pour chaque équipe nationale, à partir des matchs **déjà terminés** dans le
tournoi (phase de groupes), on calcule :

- **Force d'attaque** (domicile / extérieur) = buts marqués relatifs à la moyenne.
- **Force de défense** (domicile / extérieur) = buts encaissés relatifs à la moyenne.

> Note : dans un tournoi sur terrain neutre, la distinction domicile/extérieur est
> faible. Le modèle la conserve par généralité mais, faute de données suffisantes,
> peut utiliser des forces globales (non séparées dom/ext). Par défaut **forces
> globales** pour la Coupe du Monde (terrain neutre).

### 3.2 Buts attendus
- `λ_domicile = baseline_buts_dom × attaque_dom × défense_ext_adverse`
- `λ_extérieur = baseline_buts_ext × attaque_ext × défense_dom_adverse`

`baseline_buts_*` et l'avantage terrain font partie des **hyperparamètres**
ajustables (voir §5).

### 3.3 Matrice des scores
`P(dom=i, ext=j) = Poisson(i; λ_dom) × Poisson(j; λ_ext)` pour i, j ∈ [0, 8].

### 3.4 Dérivations (depuis la matrice unique)
- **Score exact** = case (i, j) de probabilité maximale.
- **1N2** = somme triangle inférieur (dom gagne) / diagonale (nul) / triangle
  supérieur (ext gagne).
- **Over 2.5** = somme des cases où i + j ≥ 3 ; **Under 2.5** = complément.
- **BTTS oui** = somme des cases où i ≥ 1 ET j ≥ 1.

### 3.5 Gestion des données rares (repli / shrinkage)
Si une équipe a joué peu de matchs (seuil hyperparamètre, ex : < 2), ses forces
sont ramenées vers la moyenne du tournoi (régularisation / shrinkage) afin
d'éviter des estimations aberrantes. Un **indice de fiabilité** (faible/moyen/élevé)
est affiché selon le nombre de matchs disponibles.

### 3.6 Phase à élimination directe
Le pronostic 1N2 porte sur le **résultat à la fin du temps réglementaire (90 min)**.
L'interface signale que prolongations / tirs au but sont possibles en phase finale.

## 4. Mémoire & réconciliation des résultats

### 4.1 Principe
Chaque fois qu'un pronostic est calculé pour un match à venir, il est **enregistré**
en base avec la version de modèle utilisée. Après le match, le **score réel** est
récupéré et associé au pronostic. On peut ainsi mesurer la performance dans le temps.

### 4.2 Réconciliation
Un processus `reconciler` :
1. Cherche en base les matchs pronostiqués dont le statut est désormais `FINISHED`
   et qui n'ont pas encore de résultat enregistré.
2. Récupère le score réel via football-data.org.
3. Stocke le résultat et calcule les métriques par pronostic (§6).

Déclenchement : endpoint manuel **et** automatiquement au chargement des matchs à
venir (paresseux), sous réserve du cache pour respecter la limite d'API.

## 5. Auto-amélioration (apprentissage par backtest)

### 5.1 Hyperparamètres ajustables
- `avantage_terrain` (peut être neutralisé pour la CdM)
- `baseline_buts_dom`, `baseline_buts_ext`
- `force_shrinkage` (intensité de la régularisation vers la moyenne)
- `time_decay` (poids des matchs récents ; optionnel, défaut neutre)

### 5.2 Boucle d'apprentissage (`tuner`)
1. Rassembler tous les pronostics ayant un résultat réel.
2. **Backtest walk-forward** : pour chaque match passé, recalculer la prédiction
   avec des hyperparamètres candidats en n'utilisant **que les matchs joués avant**
   ce match (pas de fuite de données / data leakage).
3. Optimiser les hyperparamètres pour **minimiser le RPS moyen** (recherche par
   grille ou `scipy.optimize`).
4. Enregistrer le jeu d'hyperparamètres gagnant comme **nouvelle version de modèle**
   (`model_versions`), avec ses scores de backtest.
5. La **meilleure version** devient celle utilisée pour les futurs pronostics.

Déclenchement : endpoint manuel (`POST /api/tune`) ; peut aussi être lancé après
réconciliation quand assez de nouveaux résultats sont disponibles.

### 5.3 Honnêteté statistique
Une CdM ≈ 64 matchs : l'amélioration sera **réelle mais modeste** sur un seul
tournoi. L'architecture (mémoire + versions) est conçue pour continuer à progresser
si on l'étend à d'autres compétitions. À documenter clairement dans l'UI/README.

## 6. Métriques de performance
Calculées en joignant `predictions` et `results` :

- **RPS** (Ranked Probability Score) sur le 1N2 — métrique principale.
- **Score de Brier** (multi-classe 1N2).
- **Log-loss** 1N2.
- **Taux de score exact** (prédit == réel).
- **Précision 1N2** (issue prédite la plus probable == issue réelle).
- **Courbe de calibration** (probabilité prédite vs fréquence observée).

Agrégées globalement et **par version de modèle** pour visualiser la progression.

## 7. Architecture

```
pronostique_foot/
├── backend/
│   ├── main.py              # App FastAPI + routes
│   ├── config.py            # Clé API, compétition (WC), chemins, constantes
│   ├── football_api.py      # Client football-data.org (matchs, équipes, résultats)
│   ├── poisson_model.py     # Cœur statistique : forces, λ, matrice, dérivations
│   ├── predictor.py         # Orchestration : un match → objet prédiction complet
│   ├── storage.py           # Couche SQLite (predictions, results, model_versions)
│   ├── reconciler.py        # Associe pronostics ↔ scores réels
│   ├── metrics.py           # RPS, Brier, log-loss, calibration, score exact
│   ├── tuner.py             # Backtest walk-forward + optimisation hyperparamètres
│   └── cache.py             # Cache disque (respecter la limite 10 req/min)
├── frontend/
│   ├── index.html           # Matchs à venir + cartes de pronostic
│   ├── performance.html     # Tableau de bord performance & versions
│   ├── style.css            # Design soigné
│   └── app.js               # Appels API backend, rendu
├── data/
│   └── app.db               # SQLite (créée au runtime, ignorée par git)
├── .env                     # FOOTBALL_DATA_API_KEY (jamais commit)
├── .env.example
├── requirements.txt
└── README.md
```

### Découplage des responsabilités
- `football_api` : I/O réseau uniquement (via cache). Aucune logique métier.
- `poisson_model` : mathématiques pures, **testable sans réseau ni base**.
- `storage` : persistance SQLite, requêtes typées. Aucune logique métier.
- `metrics` : fonctions pures (entrée probas + résultat → scores).
- `reconciler` : pont `football_api` ↔ `storage` pour les résultats.
- `tuner` : utilise `storage` (historique) + `poisson_model` pour optimiser.
- `predictor` : orchestration des prédictions et de leur enregistrement.

## 8. Modèle de données (SQLite)

**predictions**
`id, match_id, competition, home_team, away_team, match_utc_date,
model_version_id, created_at, pred_home, pred_away, prob_home, prob_draw,
prob_away, prob_over25, prob_btts, lambda_home, lambda_away, reliability`

**results**
`match_id (PK), actual_home, actual_away, status, reconciled_at`

**model_versions**
`id, created_at, params_json, backtest_rps, backtest_brier, is_active, notes`

Les métriques par pronostic sont calculées à la volée par jointure
`predictions ⋈ results` (pas de table dédiée en v1).

## 9. Flux de données
1. Frontend appelle `GET /api/matches/upcoming`.
2. Backend : réconciliation paresseuse des matchs terminés → récupère (via cache)
   les matchs WC à venir + les matchs terminés (pour calculer les forces).
3. `poisson_model` calcule les forces (version de modèle active) ; `predictor`
   produit la prédiction, **l'enregistre** en base, et la renvoie.
4. Frontend affiche une carte par match.
5. Page performance : `GET /api/performance` agrège les métriques depuis la base.

## 10. Endpoints backend
- `GET /api/matches/upcoming` → matchs à venir avec pronostics (et enregistrement).
- `GET /api/match/{id}` → détail d'un match (stats étendues).
- `GET /api/performance` → métriques agrégées + par version de modèle.
- `POST /api/reconcile` → force la réconciliation des résultats.
- `POST /api/tune` → lance la boucle d'auto-amélioration (nouvelle version).
- `GET /` → frontend ; `GET /performance` → tableau de bord.

## 11. Interface (frontend)
- **Accueil** : liste verticale de **cartes de match** (équipes, date/heure).
  Chaque carte : score exact prédit en évidence, barres de probabilité 1N2,
  badges Over/Under et BTTS, mini-section forme/H2H, indice de fiabilité.
- **Tableau de bord performance** : métriques globales (RPS, Brier, taux de score
  exact, précision 1N2), courbe de calibration, et comparaison des versions de
  modèle dans le temps.
- Disclaimer visible. Design soigné (skill frontend-design à l'implémentation).

## 12. Stratégie de test
- **Tests unitaires `poisson_model`** : λ donnés → matrice correcte, somme ≈ 1,
  dérivations 1N2/OU/BTTS cohérentes, score exact = argmax attendu.
- **Test du repli/shrinkage** : équipe sans match → forces vers la moyenne,
  fiabilité « faible », pas de crash.
- **Tests `metrics`** : RPS/Brier/log-loss sur cas connus (ex : prédiction
  parfaite → RPS 0 ; valeurs de référence vérifiables à la main).
- **Tests `storage`** : insertion/lecture sur base SQLite temporaire.
- **Test `tuner`** : sur un mini-historique simulé, vérifie qu'une nouvelle version
  est créée et que l'absence de fuite de données est respectée (walk-forward).
- **Test client API** avec réponses simulées (mock), sans appel réseau réel.

## 13. Hors périmètre (YAGNI v1)
- Pas de comptes utilisateurs.
- Pas de base relationnelle lourde (SQLite suffit).
- Pas de cotes de bookmakers (non garanties en gratuit).
- Pas d'autres compétitions en v1 (code paramétré par code de compétition → extensible).
- Pas de modèle ML complexe : on optimise les quelques hyperparamètres du Poisson,
  pas un réseau de neurones (données insuffisantes, §5.3).

## 14. Risques / points de vigilance
- **Disponibilité de `WC` en gratuit** : à confirmer tôt.
- **Données rares en phase de groupes** : géré par le repli (§3.5).
- **Fuite de données au backtest** : strictement évitée par le walk-forward (§5.2).
- **Off-season / pas de matchs à venir** : l'UI affiche un message clair si aucun
  match à venir n'est retourné.
- **Limite de débit** : strictement gérée par le cache.
- **Volume de données faible** : amélioration modeste sur un seul tournoi (§5.3).
