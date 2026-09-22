# Système de Recommandation TripAdvisor

Ce projet implémente un système de recommandation basé exclusivement sur les avis textuels des utilisateurs (sans utiliser les métadonnées de lieux). L'objectif est de recommander les endroits les plus similaires à un lieu donné en s'appuyant sur l'empreinte textuelle laissée par les visiteurs.

## 👥 Auteurs
- Timothée Joliot
- Ovia Chanemouganandam

## 🎯 Objectif et Hypothèse
L'hypothèse principale est que des expériences similaires (par exemple, une même cuisine, un type d'attraction similaire) sont exprimées avec un vocabulaire similaire dans les avis des utilisateurs. En capturant cette similarité textuelle, nous pouvons recommander des lieux pertinents.

## ⚙️ Méthodologie

### Prétraitement des Données
- **Filtrage par langue** : Seuls les avis en anglais (`langue == 'en'`) sont conservés pour garantir un espace de vocabulaire commun.
- **Équilibrage du corpus** : Pour éviter qu'un lieu ayant des milliers d'avis ne domine l'espace vectoriel, nous limitons l'agrégation aux 20 premiers avis par lieu. Ces avis sont ensuite concaténés pour former un profil de lieu unique ("Place Profile").

### Protocole d'Évaluation
Le modèle est évalué sur un split train/test de 50/50, avec le set d'entraînement servant de requêtes et le set de test de base de données.
Deux niveaux d'erreur de classement sont calculés (plus le score est bas, meilleur est le résultat) :
- **Niveau 1** : Mesure la similarité globale (ex: l'endroit retourné est-il bien un Hôtel, un Restaurant ou une Attraction ?).
- **Niveau 2** : Mesure la similarité fine (ex: pour un restaurant, s'agit-il du même type de cuisine ?).

## 🚀 Modèles et Résultats

### 1. Modèle de Base (BM25)
Le modèle BM25 est une approche probabiliste souvent utilisée pour la recherche d'informations (requête courte vers un document long). Dans notre cas (profil contre profil), ce modèle montre certaines limites.
- **Erreur Niveau 1** : 0.58
- **Erreur Niveau 2** : 5.94

### 2. Modèle Amélioré (TF-IDF + Cosine Similarity)
Nous avons implémenté un Vectorizer TF-IDF restreint aux 3000 mots les plus fréquents (avec filtrage des stop-words anglais), couplé à une similarité cosinus. Cette approche s'est révélée beaucoup plus adaptée pour comparer des documents longs entre eux et a permis d'améliorer considérablement la recherche de granularité fine.
- **Erreur Niveau 1** : 0.67
- **Erreur Niveau 2** : 4.66

## 📈 Conclusion et Perspectives
Notre modèle TF-IDF avec similarité cosinus a considérablement réduit l'erreur de Niveau 2 (passant de 5.94 à 4.66), validant ainsi notre hypothèse de départ : il est tout à fait possible de capturer des similarités complexes (comme le type de cuisine ou l'ambiance) uniquement grâce au traitement du langage naturel sur les avis utilisateurs.

**Limites** : L'erreur de Niveau 1 a légèrement augmenté avec TF-IDF. Cela s'explique par le fait que BM25 est très performant pour associer des mots très fréquents qui identifient des catégories larges (Level 1), tandis que TF-IDF excelle dans la reconnaissance de mots spécifiques (Level 2).

**Améliorations futures** : Intégrer la Modélisation de Sujets (Topic Modeling avec LDA) pour extraire des thèmes précis (Atmosphère, Qualité de la nourriture, Prix) et ne comparer que les avis pertinents selon le contexte.

## 🛠️ Exécuter le Projet
Le projet peut être exécuté via le fichier principal `pipeline.py` :
```bash
python pipeline.py
```
Les résultats d'évaluation seront générés dans le terminal et sauvegardés dans un fichier `results.txt`.
