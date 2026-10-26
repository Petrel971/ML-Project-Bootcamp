# Contexte du projet

Ce fichier rassemble **tout le fond du projet** : la consigne, les choix faits et leurs raisons,
ce qui a été vérifié, où on en est. Il sert à deux choses :

- **reprendre le projet** sans avoir suivi les discussions précédentes ;
- **le donner à un assistant IA** (« lis `CONTEXT.md` ») pour poser des questions ou travailler
  sur le projet avec tout le contexte.

Pour le détail des tâches et du planning, voir [PLAN.md](PLAN.md).

> **À tenir à jour :** quand une décision est prise ou qu'une étape est finie, mettre à jour les
> sections 4 (décisions) et 8 (état d'avancement).

---

## 1. Le cours et l'évaluation

- Cours : *Mathematics & Statistics for Machine Learning*, Alassane Diallo, Albert School, MSc 1
  (MSA-DATI07-04).
- Équipe : **Lucas, Manuel, Amine** (3 membres).
- **Rendu (rapport PDF + code) : 6 novembre 2026 au plus tard.** Oral : **13 novembre 2026**.
  Le rendu doit arriver 7 jours avant l'oral : le prof relit et exécute le code avant.
- Barème :

| Critère | Poids |
|---|---|
| Rapport : clarté, organisation, qualité scientifique, graphiques | 20 % |
| Implémentation from scratch : justesse, lisibilité, modularité, transparence | 25 % |
| Expériences : protocole, comparaison, interprétation, analyse critique | 25 % |
| Oral : clarté, structure, timing | 15 % |
| Réponses aux questions méthodologiques et techniques | 15 % |

**Ce que le prof veut voir :** une vraie maîtrise de l'algorithme et de son entraînement, pas
seulement un modèle existant qu'on fait tourner. Un bon projet explique *pourquoi* la méthode
marche, *à quel point* elle marche, et *quand* elle échoue.

## 2. Ce que la consigne impose

1. Un dataset Kaggle et une question de recherche formulée par le groupe.
2. Un algorithme codé **from scratch**, avec son entraînement : formulation, fonction objectif,
   optimisation ou construction, prédiction.
3. **Librairies.** NumPy, pandas, Matplotlib et Seaborn sont autorisés partout. scikit-learn ne
   sert **qu'à** la comparaison, aux métriques et au découpage des données. Tout code externe doit
   être cité.
4. Une comparaison avec la version d'une librairie, **sur les mêmes splits**, avec **plusieurs
   métriques** justifiées.
5. Une préparation des données documentée (valeurs manquantes, encodage, split) **sans fuite de
   données**.
6. Une analyse critique : comportement, limites, coût de calcul, améliorations.
7. Reproductibilité : dépendances, graines, accès aux données (donner l'URL Kaggle, ne pas
   redistribuer les données), ordre d'exécution.
8. Livrables : rapport PDF (problème, données, méthode mathématique, protocole, figures et
   tableaux, résultats comparés, discussion, références), code, oral suivi de questions.

## 3. Le projet en bref

- **Dataset :** Adult / Census Income (recensement américain de 1994).
  Kaggle : https://www.kaggle.com/datasets/uciml/adult-census-income
- **Tâche :** classification binaire, prédire si une personne gagne plus de 50 000 $ par an.
- **Méthode :** un arbre de classification **CART** codé from scratch avec NumPy, comparé à
  `sklearn.tree.DecisionTreeClassifier` avec les mêmes hyperparamètres.
- **Question de recherche :**
  > L'éducation est-elle le principal déterminant d'un revenu supérieur à 50 000 $, ou d'autres
  > facteurs comme la situation familiale, l'âge ou le temps de travail pèsent-ils davantage ?
  >
  > *Is education the main driver of earning more than $50K a year, or do other factors such as
  > marital status, age and hours worked matter more?*
- **Hypothèse de départ :** l'éducation est le facteur n°1. Le projet la confirme ou l'infirme.

## 4. Décisions et leurs raisons

| Décision | Pourquoi |
|---|---|
| **Dataset Adult** | Taille raisonnable (~48 800 lignes), mélange de variables numériques et catégorielles, cible binaire claire, déséquilibre modéré (24 %). **Écartés :** *Bank Marketing* (plus déséquilibré, et la variable `duration` crée une fuite de données) ; *Diabetes 130 Hospitals* (nettoyage lourd, 3 classes, patients répétés, signal faible). |
| **CART** | Il est au programme (Lecture 3B, partie I). Il gère les variables numériques et catégorielles sans normalisation. Il est faisable à la main en deux semaines. Il est **vérifiable** : avec les mêmes réglages, nos splits doivent être identiques à ceux de sklearn. Il est **lisible**, donc il répond directement à la question de recherche. |
| **Question sur l'éducation** | Elle part d'une hypothèse qu'on peut tester. Sur Adult, c'est souvent la **situation familiale** (`relationship`, `marital-status`) qui sort en premier, pas l'éducation : un résultat surprenant, donc intéressant à discuter. La comparaison avec sklearn n'est pas dans la question, car la consigne l'impose de toute façon. |
| **Colonnes retirées** | `fnlwgt` est un poids d'échantillonnage du recensement, pas une caractéristique de la personne. `education` est redondante avec `education-num` (même information, codée en entier). |
| **Valeurs manquantes = modalité « Unknown »** | Elles ne touchent que des variables catégorielles (`workclass`, `occupation`, `native-country`). En faire une catégorie garde toutes les lignes, et l'arbre peut apprendre si « inconnu » est informatif. À justifier dans le rapport. |
| **One-hot encoding calculé sur le train** | Les colonnes sont fixées à partir du train puis appliquées à la validation et au test. Aucune information des autres jeux n'entre dans la préparation. |
| **Pas de normalisation** | Un arbre ne compare une variable qu'à un seuil : changer l'échelle ne change pas les splits. |
| **Split stratifié 60/20/20, graine 42** | La stratification garde 24 % de >50K dans chaque jeu. Train pour apprendre, validation pour régler les hyperparamètres, test utilisé **une seule fois** à la fin. La graine fixe rend tout reproductible. |
| **Métriques** | Prédire toujours « ≤50K » donne déjà ≈ 76 % d'accuracy avec un F1 de 0 : l'accuracy seule est trompeuse. On suit aussi précision, rappel, F1 et la matrice de confusion. |
| **Deux notebooks** | La consigne sépare « implémentation » et « expériences » ; le prof retrouve sa grille. Deux notebooks évitent aussi les conflits git quand plusieurs personnes travaillent en même temps. |
| **Le CART vit dans `src/cart.py`** | Le code est écrit une seule fois. Les notebooks l'importent et les tests le vérifient : pas de copies qui divergent. |
| **Livrables en anglais** | Comme le cours. Les discussions du groupe restent en français. |
| **uv pour l'environnement** | `uv.lock` fixe les versions exactes : tout le monde a le même environnement, et le prof aussi. |

## 5. Ce qui a été vérifié sur les vraies données

Avec `uv run python -m src.data` (le 22/10/2026) :

| Jeu | Lignes | Variables après one-hot | Part de >50K |
|---|---|---|---|
| train | 29 304 | 91 | 23,9 % |
| validation | 9 769 | 91 | 23,9 % |
| test | 9 769 | 91 | 23,9 % |

Total : 48 842 lignes, 14 variables d'origine (6 numériques, 8 catégorielles).

## 6. Les maths du CART (Lecture 3B)

- **Modèle.** Chaque nœud teste une règle `x_j ≤ s`. Chaque feuille contient les proportions p_k
  de chaque classe et prédit la classe majoritaire. L'arbre découpe l'espace en rectangles.
- **Impureté de Gini.** G = 1 − Σ p_k². Elle vaut 0 pour un nœud pur et 0,5 au maximum avec deux
  classes. À la racine d'Adult, G ≈ 1 − (0,76² + 0,24²) ≈ 0,36.
- **Score d'un split.** G_split = (n_L/n)·G_L + (n_R/n)·G_R. On garde le split qui minimise
  G_split (donc qui maximise le gain G_parent − G_split). On pondère par la taille pour qu'un
  split qui isole une seule observation pure ne paraisse pas excellent.
- **Seuils testés.** Les milieux entre deux valeurs distinctes consécutives, comme sklearn : entre
  deux valeurs observées, tous les seuils donnent la même partition.
- **Recherche vectorisée.** On trie les valeurs d'une variable, puis des sommes cumulées des
  classes donnent les comptes à gauche pour tous les seuils d'un coup (à droite : total − gauche).
  Coût O(d · n log n) au lieu de O(d · n²) pour la version naïve.
- **Construction.** Gloutonne et récursive : meilleur split, puis on recommence dans chaque groupe
  jusqu'à une règle d'arrêt (`max_depth`, `min_samples_split`, `min_samples_leaf`, nœud pur,
  `min_impurity_decrease`). Glouton, parce que trouver l'arbre optimal est NP-difficile.
- **Importance d'une variable.** La somme des baisses d'impureté pondérées (n_t/N)·(G_t − G_split)
  sur les nœuds qui utilisent cette variable, normalisée à 1. Elle avantage les variables qui ont
  beaucoup de seuils possibles.
- **Élagage (option).** `ccp_alpha` : on coupe après coup les branches qui n'apportent pas assez.
- **Messages à faire passer.** Un arbre profond sur-apprend (courbe train/validation selon la
  profondeur). Un arbre seul est instable. Les forêts aléatoires et le boosting font mieux : ce
  sont les pistes d'amélioration.

## 7. Le code

```
notebooks/01_cart_from_scratch.ipynb   construction du CART pas à pas (style Lab 1, checkpoints)
notebooks/02_experiments.ipynb         exploration, expériences, réponse à la question, limites
src/data.py                            chargement d'Adult, encodage, split (SEED = 42)
src/cart.py                            le CART (NumPy uniquement)
tests/test_cart.py                     tests du CART
```

```bash
uv sync                          # installe l'environnement
uv run python -m src.data        # télécharge Adult et affiche les tailles des splits
uv run pytest                    # tests du CART
```

Dans VS Code : ouvrir le dossier, puis choisir le kernel `.venv` dans les notebooks.

**Style attendu** (comme le Lab 1 du cours) : NumPy uniquement pour le modèle, pas de boucle
Python sur les lignes quand on peut vectoriser, un *checkpoint* (`assert` + `OK`) après chaque
fonction, `SEED = 42`.

## 8. État d'avancement

*Dernière mise à jour : 26/10/2026.*

- ✅ Structure du repo, chargement et préparation des données (vérifiés sur les vraies données).
- ✅ Squelettes des deux notebooks (titres, setup, cellules TODO), utilisables dans VS Code.
- ⬜ **CART** : `src/cart.py` n'est qu'un squelette, toutes ses fonctions lèvent
  `NotImplementedError`. Les 3 tests échouent pour cette raison, c'est normal.
- ⬜ Expériences du notebook 02.
- ⬜ Rapport, slides.
- ⬜ Répartition des 3 lots entre Lucas, Manuel et Amine (voir [PLAN.md](PLAN.md)).

**Prochaine étape :** coder `gini`, le score d'un split, puis la version naïve de `best_split`
(notebook 01, sections 2 à 4).

## 9. Règles pour travailler sur le repo

- **Une branche par tâche**, puis une pull request relue par un autre membre avant la fusion.
- **Un notebook = un responsable à la fois**, pour éviter les conflits git sur les notebooks.
- Avant de pousser : `uv run pytest` passe et le notebook tourne du début à la fin.
- **Ne jamais utiliser le jeu de test** avant la comparaison finale (notebook 02, section 6).
  Les hyperparamètres se choisissent sur la validation.
- Dépendances : `uv add <paquet>`, jamais `pip install`.
- Ne pas committer les données (`data/` est ignoré) ni les PDF du cours (`docs/`, ignoré).

## 10. Pour un assistant IA qui reprend ce projet

- Réponds en **français** ; le code, les notebooks et le rapport sont en **anglais**.
- Le groupe apprend le ML : **explique le pourquoi** de chaque choix avec les notions et les
  notations du cours, pour que chacun puisse le défendre à l'oral. Sur `src/cart.py`, réfléchis
  d'abord à l'approche avec la personne avant de coder.
- **Respecte la consigne** : rien de scikit-learn dans le modèle lui-même, pas de fuite de
  données, test utilisé une seule fois.
- **Vérifie en exécutant** (`uv run pytest`, le notebook) avant de dire que quelque chose marche.
- Ne fais de commit ou de push que si on te le demande, et mets à jour les sections 4 et 8 de ce
  fichier quand une décision est prise ou une étape terminée.

## 11. Questions probables à l'oral

- Pourquoi le Gini plutôt que l'entropie ? Que vaut-il pour un nœud 50/50 ?
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
