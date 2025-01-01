# Detection du Niveau de Concentration

## Description
Ce projet vise à détecter le niveau de concentration à travers l'analyse des expressions faciales à l'aide de modèles d'apprentissage profond. Deux architectures de réseaux de neurones convolutifs (CNN) sont utilisées : un modèle CNN personnalisé et un modèle pré-entraîné VGG16. Le projet analyse les émotions faciales capturées en temps réel ou via des images pour estimer l'état de concentration.

### Explication du Projet
Avec la montée de l'enseignement à distance, il devient essentiel de surveiller et d'évaluer le niveau de concentration des étudiants en temps réel pour améliorer leur engagement et leur apprentissage. Ce projet combine plusieurs approches innovantes pour détecter et analyser les niveaux de concentration à l'aide de données multi-sources :

- **Expressions faciales** : Analyse des émotions (joie, tristesse, neutralité, etc.) pour identifier l'état émotionnel de l'étudiant.
- **État des yeux** : Suivi des états ouverts/fermés et de la direction du regard pour détecter des signes de distraction ou de somnolence.
- **Indicateurs physiologiques et environnementaux** : Bien que non intégrés dans cette version, ces données (rythme cardiaque, EEG, niveau de bruit, etc.) apportent des insights supplémentaires sur l'état mental de l'étudiant.

L'objectif est de catégoriser les étudiants en trois niveaux de concentration (hautement concentré, modérément concentré, et déconcentré) afin de permettre aux enseignants d'identifier rapidement les besoins individuels et d'ajuster leurs méthodes pédagogiques en conséquence.

## Structure du Projet
```
IsmailDr13f-Detection_du_niveau_de_concentration/
├── referances                            # Ressources et articles utilisés comme références
├── emotion-classification-cnn-using-keras.ipynb  # Notebook pour l'entraîment et les tests avec CNN
├── haarcascade_frontalface_default.xml   # Classificateur Haar pour la détection de visages
├── emotion-classification-cnnvgg16-using-keras.py  # Script pour l'entraîment avec VGG16
├── cnn_model.h5                          # Modèle pré-entraîné CNN enregistré
├── main_cnn.py                           # Script principal pour exécuter la détection avec le modèle CNN
├── main_vgg16.py                         # Script principal pour exécuter la détection avec le modèle VGG16
```

## Prérequis
Pour exécuter ce projet, vous devez disposer de l'environnement suivant :
- Python 3.6
- TensorFlow 2.16.1
- Keras
- OpenCV
- NumPy
- Matplotlib

## Installation
. Clonez ce répertoire :
   ```bash
   git clone https://github.com/IsmailDr13f/Detection_du_niveau_de_concentration.git
   ```

## Utilisation

### 1. Entraîner le Modèle
Pour entraîner le modèle personnalisé CNN, utilisez le notebook :
```bash
jupyter notebook emotion-classification-cnn-using-keras.ipynb
```
Pour entraîner le modèle VGG16, exécutez le script correspondant :
```bash
python emotion-classification-cnnvgg16-using-keras.py
```

### 2. Exécution du Modèle
Pour utiliser le modèle CNN pour la détection :
```bash
python main_cnn.py
```
Pour utiliser le modèle VGG16 pour la détection :
```bash
python main_vgg16.py
```

### 3. Détection en Temps Réel
Assurez-vous que votre webcam est connectée et exécutez l'un des scripts principaux. Le classificateur Haar sera utilisé pour détecter les visages avant l'analyse des émotions.

## Résultats
Les résultats de la détection afficheront :
- Les émotions détectées (ex. : Heureux, Triste, Neutre, etc.)
- Une estimation de l'état de concentration à partir des émotions.

## Ressources
Le dossier `referances` contient des articles et des documents relatifs à la classification des émotions et à l'utilisation des modèles CNN/VGG16.

## Auteur
Ce projet a été réalisé par **Ismail Drief**, étudiant en Intelligence Artificielle et Data Science.



