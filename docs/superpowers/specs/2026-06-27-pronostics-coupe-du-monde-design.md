# Pronostics Coupe du Monde 2026 — Spécification de conception

**Date :** 2026-06-27
**Statut :** Approuvé (design) — en attente de plan d'implémentation

## 1. Objectif

Application web qui affiche des **pronostics de score** pour les matchs **à venir**
de la Coupe du Monde 2026, fondés sur une **analyse statistique** issue de données
réelles et fiables (API football-data.org).

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
> peut utiliser des forces globales (non séparées dom/ext). Décision tranchée à
> l'implémentation selon la richesse des données ; par défaut **forces globales**
> pour la Coupe du Monde (terrain neutre).

### 3.2 Buts attendus
- `λ_domicile = moyenne_buts_dom_ligue × attaque_dom × défense_ext_adverse`
- `λ_extérieur = moyenne_buts_ext_ligue × attaque_ext × défense_dom_adverse`

### 3.3 Matrice des scores
`P(dom=i, ext=j) = Poisson(i; λ_dom) × Poisson(j; λ_ext)` pour i, j ∈ [0, 8].

### 3.4 Dérivations (depuis la matrice unique)
- **Score exact** = case (i, j) de probabilité maximale.
- **1N2** = somme triangle inférieur (dom gagne) / diagonale (nul) / triangle
  supérieur (ext gagne).
- **Over 2.5** = somme des cases où i + j ≥ 3 ; **Under 2.5** = complément.
- **BTTS oui** = somme des cases où i ≥ 1 ET j ≥ 1.

### 3.5 Gestion des données rares (repli)
Si une équipe a joué peu de matchs (seuil à définir, ex : < 2), ses forces sont
ramenées vers la moyenne du tournoi (régularisation / shrinkage) afin d'éviter des
estimations aberrantes. Un **indice de fiabilité** (faible/moyen/élevé) est affiché
selon le nombre de matchs disponibles.

### 3.6 Pondération temporelle (optionnelle)
Possibilité de pondérer les matchs récents plus fortement (time-decay). Marqué
comme amélioration optionnelle — non bloquant pour la v1.

### 3.7 Phase à élimination directe
Le pronostic 1N2 porte sur le **résultat à la fin du temps réglementaire (90 min)**.
L'interface signale que prolongations / tirs au but sont possibles en phase finale.

## 4. Architecture

```
pronostique_foot/
├── backend/
│   ├── main.py              # App FastAPI + routes
│   ├── config.py            # Clé API, compétition (WC), constantes du modèle
│   ├── football_api.py      # Client football-data.org (matchs, équipes, résultats)
│   ├── poisson_model.py     # Cœur statistique : forces, λ, matrice, dérivations
│   ├── predictor.py         # Orchestration : un match → objet prédiction complet
│   └── cache.py             # Cache disque (respecter la limite 10 req/min)
├── frontend/
│   ├── index.html           # Liste des matchs à venir + cartes de pronostic
│   ├── style.css            # Design soigné
│   └── app.js               # Appels à l'API backend, rendu des cartes
├── .env                     # FOOTBALL_DATA_API_KEY (jamais commit)
├── .env.example
├── requirements.txt
└── README.md
```

### Découplage des responsabilités
- `football_api` : I/O réseau uniquement (via cache). Aucune logique métier.
- `poisson_model` : mathématiques pures, **testable sans réseau**.
- `predictor` : orchestration (récupère données → calcule forces → produit
  prédiction). Pont entre `football_api` et `poisson_model`.
- `cache` : persistance disque avec TTL pour limiter les appels API.

## 5. Flux de données
1. Frontend appelle `GET /api/matches/upcoming`.
2. Backend récupère (via cache) les matchs WC à venir + les matchs terminés
   (pour calculer les forces).
3. `poisson_model` calcule les forces ; `predictor` produit pour chaque match :
   score exact, 1N2, Over/Under, BTTS, stats de forme.
4. Frontend affiche une carte par match.

## 6. Endpoints backend
- `GET /api/matches/upcoming` → liste des matchs à venir avec pronostics complets.
- `GET /api/match/{id}` → détail d'un match (stats étendues).
- `GET /` → sert le frontend statique.

## 7. Interface (frontend)
- Liste verticale de **cartes de match** (équipe A vs équipe B, date/heure).
- Chaque carte : score exact prédit en évidence, barres de probabilité 1N2,
  badges Over/Under et BTTS, mini-section forme/H2H, indice de fiabilité.
- Disclaimer visible.
- Design soigné (voir skill frontend-design à l'implémentation).

## 8. Stratégie de test
- **Tests unitaires `poisson_model`** avec données connues :
  - λ donnés → matrice correcte, somme des probabilités ≈ 1.
  - Dérivations 1N2 / Over-Under / BTTS cohérentes (ex : cas symétrique).
  - Score exact = argmax attendu.
- **Test du repli** : équipe avec peu/pas de matchs → forces ramenées vers la
  moyenne, pas de crash, fiabilité « faible ».
- **Test client API** avec réponses simulées (mock), sans appel réseau réel.

## 9. Hors périmètre (YAGNI v1)
- Pas de comptes utilisateurs, pas de base de données relationnelle (cache fichier
  suffit).
- Pas de cotes de bookmakers (non garanties en gratuit).
- Pas d'autres compétitions en v1 (mais code paramétré par code de compétition,
  donc extensible).
- Pas de pondération temporelle obligatoire (optionnelle).

## 10. Risques / points de vigilance
- **Disponibilité de `WC` en gratuit** : à confirmer tôt.
- **Données rares en phase de groupes** : géré par le repli (§3.5).
- **Off-season / pas de matchs à venir** : l'UI doit afficher un message clair si
  aucun match à venir n'est retourné.
- **Limite de débit** : strictement gérée par le cache.
