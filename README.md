
#  Projet : Scoring et Analyse Avancée des Actifs du CAC 40

Ce projet vise à évaluer et à segmenter les actifs du CAC 40 en utilisant des techniques d'apprentissage non supervisé (clustering) et à explorer la prédiction de leurs caractéristiques financières clés (volatilité et rendement).

---

##  Objectifs Principaux

* **Segmentation Stratégique :** Identifier des groupes d'actifs du CAC 40 (clusters) partageant des caractéristiques financières et de performance similaires pour optimiser les stratégies d'investissement.
* **Modélisation Prédictive :** Développer des modèles pour anticiper la **volatilité** et le **rendement** futurs des actifs.

---

## Outils et Méthodes Utilisées

### 1. Clustering (Segmentation des Actifs)

L'approche de clustering est utilisée pour regrouper les titres en fonction de leurs métriques fondamentales et de performance.

| Algorithme | Objectif Principal | Rôle dans le Projet |
| :--- | :--- | :--- |
| **K-means** | Partitionnement basé sur la distance euclidienne. | Détermination de groupes bien définis, fournissant une segmentation de base. |
| **DBSCAN** | Regroupement basé sur la densité. | Identification de clusters de forme arbitraire et distinction des **valeurs aberrantes (bruit)**. |
| **KNN (K-Nearest Neighbors)** | Classification/Régression par proximité. | *Usage ici : Évaluation de la robustesse des clusters et potentiellement classification de nouveaux actifs.* |

### 2. Prédiction et Analyse

* **Prédiction de Volatilité :** Modélisation et prévision de la volatilité des actifs (par exemple, utilisation de modèles **GARCH** ou d'autres techniques de séries temporelles).
* **Prédiction de Rendement :** Développement de modèles de régression pour anticiper les rendements futurs des titres.

---

##  Structure du Projet et Pipeline d'Exécution

Le projet est organisé par étapes séquentielles, chaque notebook contribuant à la construction du modèle final.

### Dossiers Clés

* `data/raw/` : Données historiques brutes du CAC 40.
* `data/processed/` : Fichiers intermédiaires (données normalisées, composantes PCA, résultats de clustering).
* `src/` : Modules Python contenant les fonctions et classes réutilisables.

### Pipeline d'Analyse (*Notebooks*)

| Fichier | Description | Statut |
| :--- | :--- | :--- |
| `01_EDA.ipynb` | Exploration des données et Nettoyage. | Complété |
| `02_Feature_Engineering.ipynb` | Standardisation / Normalisation des données et Application de l'**ACP (PCA)**. | Complété |
| `03_Clustering_Experiments.ipynb` | Expérimentation et détermination des hyperparamètres optimaux pour K-means et DBSCAN (ex: méthode du coude). | En cours |
| `04_DBSCAN_Analysis.ipynb` | Mise en œuvre finale du clustering **DBSCAN** et évaluation des résultats (Coefficient de Silhouette). | Complété |
| `05_Prediction_Modeling.ipynb` | Construction et évaluation des modèles de prédiction pour le rendement et la volatilité. | À Faire |
| `06_Final_Report.Rmd` | Synthèse des résultats de clustering et de prédiction. | À Faire |

```{python}

def regarder_mon_projet(Cluster, ML):
    if Cluster and ML:
        return "RAS - Projet de Clustering & ML complet et vérifié."
    else:
        return "RAS - En attente de complétion des étapes de Clustering ou ML."