# Analyse de sentiments binaires avec un Long Short Term Memory (LSTM) avec Pytorch

Un analyseur binaire (positif ou négatif) de sentiment avec un LSTM implémenté en PyTorch.

## Utilisation du dataset IMDB

IMDB est un ensemble de données textuel de critiques de films, composé d'un ensemble d'apprentissage de 25 000 exemples et d'un ensemble de test de 25 000 exemples.

Chaque exemple est une critique associée à un label : critique positive (1) ou négative (0).

## Étapes du Notebook :

- Import du dataset.
- Mise en forme de la data.
- Tokenisation des données textuelles.
- Création des mini-batches sur la data.
- Création du modèle basé sur l'architecture LSTM.
- Phase d'entraînement.
- Visualisation des courbes de loss et d'accuracy.
- Sauvegarde du modèle.
- Chargement du modèle sauvegardé.
- Réalisation des prédictions.

