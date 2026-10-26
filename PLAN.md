# Plan du projet

**Question de recherche.** L'éducation est-elle le principal déterminant d'un revenu supérieur à
50 000 $, ou d'autres facteurs comme la situation familiale, l'âge ou le temps de travail
pèsent-ils davantage ?

> *Is education the main driver of earning more than $50K a year, or do other factors such as
> marital status, age and hours worked matter more?*

**Hypothèse de départ :** l'éducation est le facteur n°1. Le projet la confirme ou l'infirme.

**Méthode :** un arbre de classification CART codé from scratch (NumPy), comparé à
`sklearn.tree.DecisionTreeClassifier` sur les mêmes splits.

**Dates :** rendu (rapport PDF + code) le **6 novembre 2026**, oral le **13 novembre 2026**.

**Langue :** tous les livrables (notebooks, code, commentaires, rapport, slides) sont en anglais.

---

## 1. Ce que la consigne exige, et où on y répond

| Exigence de la consigne | Où |
|---|---|
| Algorithme et entraînement codés from scratch | `src/cart.py`, notebook 01 |
| Formulation, objectif, optimisation, prédiction expliqués | notebook 01, rapport §3 |
| Préparation documentée, sans fuite de données | `src/data.py`, notebook 02 §3 |
| Comparaison avec sklearn sur les mêmes splits, plusieurs métriques | notebook 02 §6 |
| Interprétation : comportement, limites, coût, améliorations | notebook 02 §7–8, rapport §6–7 |
| Reproductibilité : dépendances, graine, accès aux données, ordre d'exécution | `README.md`, `pyproject.toml`, `uv.lock` |
| Rapport PDF, code, oral | `report/`, repo, slides |

Barème : rapport 20 %, implémentation 25 %, expériences et analyse 25 %, oral 15 %, questions 15 %.
Une bonne note demande de montrer qu'on **maîtrise** l'algorithme : *pourquoi* il marche, *à quel
point* il marche, *quand* il échoue.

---

## 2. Structure du repo

```
src/data.py                     chargement d'Adult, encodage, split 60/20/20 (graine 42)
src/cart.py                     le CART final, importé par les notebooks
notebooks/01_cart_from_scratch.ipynb   construction pas à pas du CART (style Lab 1, checkpoints)
notebooks/02_experiments.ipynb         exploration, expériences, réponse à la question, limites
tests/test_cart.py              vérifications automatiques (uv run pytest)
report/                         rapport PDF (créé au moment du rendu)
```

Proposition à valider : le code du CART est écrit **une seule fois**, dans `src/cart.py`. Le
notebook 01 l'importe, affiche chaque fonction, l'explique et la vérifie. On évite ainsi d'avoir le
même code en double.

---

## 3. Contenu des notebooks

### Notebook 01 : CART from scratch

1. **Le modèle** : des règles `x_j ≤ s`, des feuilles qui prédisent la classe majoritaire, un
   découpage de l'espace en rectangles.
2. **L'objectif** : le Gini G = 1 − Σ p_k², l'entropie, et un graphique des deux.
3. **Le score d'un split** : G_split = (n_L/n)·G_L + (n_R/n)·G_R, avec l'exercice 3B.1 du cours
   recalculé à la main.
4. **La recherche du meilleur split** : une version naïve (boucles), puis une version vectorisée
   (tri + sommes cumulées), et une mesure du temps des deux.
5. **La construction récursive** : critères d'arrêt, algorithme glouton, affichage des règles.
6. **La prédiction** : `predict_proba`, `predict`, et l'accord avec sklearn.
7. **L'importance des variables**, comparée à `feature_importances_` de sklearn.
8. **Récapitulatif** : `CARTClassifier` dans `src/cart.py`.

### Notebook 02 : expériences

1. **Problème et données** : source, variables, colonnes retirées.
2. **Exploration** : déséquilibre de 24 %, valeurs manquantes, taux de >50K selon l'éducation, le
   statut marital, l'âge, les heures et le sexe. On écrit l'hypothèse ici.
3. **Protocole** : split stratifié, encodage calculé sur le train, pas de normalisation. Le test
   n'est utilisé **qu'une seule fois**.
4. **Baseline** : prédire toujours « ≤50K » donne ≈ 76 % d'accuracy et un F1 de 0. Ça justifie les
   métriques choisies.
5. **Choix de la profondeur** : courbe train/validation selon `max_depth`, notre CART et sklearn
   superposés.
6. **Comparaison finale sur le test** : métriques, matrices de confusion, accord des prédictions,
   temps de calcul.
7. **Réponse à la question** : premier split, importances regroupées par variable d'origine, arbre
   lisible de profondeur 3, ablation (on retire l'éducation, puis le statut marital, puis l'âge),
   conclusion nuancée (proxy, corrélation, pas de causalité).
8. **Limites** : instabilité (5 graines), erreurs selon le sexe, seuil de décision, limites
   structurelles, améliorations (élagage, forêts aléatoires, boosting).

---

## 4. Répartition en 3 lots

| Lot | Responsable |
|---|---|
| 1. Le CART from scratch | **Lucas** |
| 2. Données, protocole et comparaison avec sklearn | **Manuel** |
| 3. Réponse à la question, limites et rapport | **Amine** |

Les **interfaces** sont fixées à l'avance pour que tout le monde
avance en parallèle, sans attendre les autres.

### Lot 1 : le CART from scratch (implémentation, 25 % de la note) — **Lucas**

- Dans `src/cart.py` : `gini`, `entropy`, le score d'un split, `best_split` (version naïve **et**
  version vectorisée), la construction récursive et ses critères d'arrêt, `predict_proba`,
  `predict`, `feature_importances_`, la classe `CARTClassifier`.
- Notebook 01 en entier : explications, exercice 3B.1 refait à la main, checkpoints, mesure du temps.
- Tests : valeurs du Gini, naïf = vectorisé, même premier split que sklearn, arbre profond =
  train appris à 100 %, prédictions et importances identiques à sklearn.
- Rapport : §3 (méthode, pseudo-code).
- **Interface fournie aux autres :** `CARTClassifier(max_depth=..., min_samples_leaf=...)` avec
  `fit`, `predict`, `predict_proba`, `feature_importances_`, comme sklearn.
- Charge : forte jusqu'au 29 octobre, plus légère ensuite.

### Lot 2 : données, protocole et comparaison avec sklearn — **Manuel**

- Notebook 02, sections 1 à 6 : problème et données, exploration, protocole, baseline, choix de
  la profondeur, comparaison finale sur le test (métriques, matrices de confusion, accord des
  prédictions, temps de calcul).
- `src/data.py` traduit en anglais, `README.md`, test de reproductibilité sur un clone propre.
- Rapport : §1–2 (introduction, données), §4 (protocole), §5 (résultats).
- Peut démarrer tout de suite avec `DecisionTreeClassifier` de sklearn, puis on remplace par
  `CARTClassifier` quand il est prêt : l'interface est la même.
- Charge : régulière.

### Lot 3 : réponse à la question, limites et rapport — **Amine**

- Notebook 02, sections 7 à 9 : premier split, importances regroupées par variable d'origine,
  arbre lisible de profondeur 3, ablation, conclusion ; limites (instabilité sur 5 graines,
  erreurs selon le sexe, seuil de décision, améliorations).
- **Coordination du rapport** : squelette, intégration des parties de chacun, bibliographie,
  mise en page des figures. Rapport : §6–8 (réponse, discussion, conclusion).
- Peut démarrer tout de suite avec sklearn, comme le lot 2.
- Charge : légère au début, forte à la fin.

**Pour tout le monde :** chacun rédige sa partie du rapport et prépare ses slides. **Tout le
monde** doit savoir refaire à la main un calcul de Gini et le choix d'un split : la partie
questions de l'oral est individuelle.

---

## 5. Planning

| Dates | Lot 1 : Lucas (CART) | Lot 2 : Manuel (données, comparaison) | Lot 3 : Amine (question, limites, rapport) |
|---|---|---|---|
| **26–27 oct** | Gini, score d'un split, `best_split` naïf | exploration, baseline | squelette du rapport, arbre lisible (sklearn) |
| **28–29 oct** | version vectorisée, construction, prédiction, importances | protocole, courbe de profondeur (sklearn) | importances regroupées, ablation (sklearn) |
| **29 oct** | ✅ **le CART passe tous les tests** | on passe à notre CART | on passe à notre CART |
| **30 oct–2 nov** | notebook 01 rédigé, rapport §3 | comparaison finale sur le test, §1–2, §4–5 | limites, §6–8, intégration |
| **2–5 nov** | relecture croisée, test de reproductibilité, export PDF || |
| **6 nov** | **RENDU** || |
| **7–12 nov** | slides, répétitions chronométrées, questions probables || |

---

## 6. Règles d'équipe

- Une branche par tâche, puis une pull request relue par un autre membre avant la fusion.
- **Un notebook = un responsable à la fois** : on ne modifie pas en même temps le même notebook,
  car les conflits git sur les notebooks sont pénibles à régler.
- Avant de pousser, on vérifie que `uv run pytest` passe et que le notebook tourne du début à la fin.
- Graine `SEED = 42` partout. On ne touche **jamais** au jeu de test avant le notebook 02 §6.
- Dépendances : `uv add <paquet>`, jamais `pip install`.
- On ne committe pas les données ni les PDF du cours.

---

## 7. Questions probables à l'oral (à préparer par tous)

- Pourquoi le Gini plutôt que l'entropie ? Que se passe-t-il pour un nœud 50/50 ?
- Pourquoi un algorithme glouton ? Trouve-t-il le meilleur arbre ?
- Pourquoi tester les seuils au milieu de deux valeurs ? Pourquoi est-ce suffisant ?
- Comment fonctionne votre version vectorisée ? Quelle est sa complexité ?
- Pourquoi aucune normalisation ?
- Comment savez-vous qu'il n'y a pas de fuite de données ?
- Pourquoi l'accuracy ne suffit-elle pas ici ?
- Pourquoi votre version est-elle plus lente que sklearn ?
- Pourquoi regrouper les importances par variable d'origine ?
- Le statut marital *cause*-t-il un revenu élevé ?
- Quand votre modèle échoue-t-il ?
